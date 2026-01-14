########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from lightning_utilities.core.rank_zero import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime
from lightning import Trainer

from configs import TrainerCLI_Config

class ConcatMMapIndexedDataset:
    """
    Runtime concat of multiple MMapIndexedDataset shards.
    IMPORTANT: we do NOT allow a read to cross shard boundaries.
    If offset lands too close to the end of a shard, we shift it backward
    so (local_offset + length) fits inside that shard.
    """
    def __init__(self, data_prefixes):
        if isinstance(data_prefixes, str):
            data_prefixes = [data_prefixes]
        self.datasets = [MMapIndexedDataset(p) for p in data_prefixes]

        # token sizes per shard
        self.sizes = [len(ds._bin_buffer) // ds._index._dtype_size for ds in self.datasets]

        # dtype size must match across shards
        dtype_sizes = [ds._index._dtype_size for ds in self.datasets]
        if len(set(dtype_sizes)) != 1:
            raise ValueError(f"Mixed _dtype_size across datasets: {dtype_sizes}")

        self.cumulative_sizes = np.cumsum([0] + self.sizes)  # len = n_shards+1
        self.total_data_size = int(self.cumulative_sizes[-1])

    def get(self, idx, offset, length):
        assert idx == 0

        # ----- (1) clamp in GLOBAL concatenated space -----
        max_off = self.total_data_size - length
        if max_off < 0:
            raise ValueError(f"Requested length={length} exceeds total_data_size={self.total_data_size}")
        if offset < 0:
            offset = 0
        if offset > max_off:
            offset = max_off

        # determine which shard the offset falls into
        dataset_idx = int(np.searchsorted(self.cumulative_sizes, offset, side="right") - 1)
        dataset_idx = max(0, min(dataset_idx, len(self.datasets) - 1))

        # local offset in that shard (in TOKENS)
        local_offset = int(offset - self.cumulative_sizes[dataset_idx])
        shard_tokens = int(self.sizes[dataset_idx])

        # ----- (2) clamp in SHARD space so we never request past shard end -----
        max_local = shard_tokens - length
        if max_local < 0:
            raise ValueError(f"Shard {dataset_idx} too small: {shard_tokens} tokens < length {length}")
        if local_offset > max_local:
            local_offset = max_local
        if local_offset < 0:
            local_offset = 0

        return self.datasets[dataset_idx].get(idx=0, offset=local_offset, length=length)


class MyDatasetV2(Dataset):
    """
    Support for ConcatMMapIndexedDataset with safe sampling.
    Key fix: you read req_len=ctx_len+1, so the last aligned ctx_len slot
    might be invalid. We compute max_i and use effective_slot for prime checks.
    """
    def __init__(self, config: TrainerCLI_Config, trainer: Trainer):
        self.config = config
        self.trainer = trainer

        assert config.train.data_type == "binidx"
        self.vocab_size = config.model.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = ConcatMMapIndexedDataset(config.train.data_file)
        self.data_size = self.data.total_data_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = config.runtime.epoch_global_steps * config.runtime.global_step_bsz
        rank_zero_info(f"########## training stage {config.train.train_stage} ##########")

        self.ctx_len = config.model.ctx_len
        self.req_len = self.ctx_len + 1

        # ----- (3) compute max valid aligned offset for req_len -----
        # aligned offsets are multiples of ctx_len; require i + req_len <= data_size
        self.max_i = ((self.data_size - self.req_len) // self.ctx_len) * self.ctx_len
        if self.max_i < 0:
            raise ValueError(f"Dataset too small: data_size={self.data_size}, req_len={self.req_len}")

        # number of valid aligned slots
        effective_slot = (self.max_i // self.ctx_len) + 1

        assert config.train.my_exit_tokens <= self.data_size
        assert MaybeIsPrime(config.train.magic_prime)
        assert config.train.magic_prime % 3 == 2
        assert config.train.magic_prime / effective_slot > 0.99 and config.train.magic_prime / effective_slot <= 1

        rank_zero_info(
            f"ctx_len={self.ctx_len} req_len={self.req_len} "
            f"max_i={self.max_i} effective_slot={effective_slot} magic_prime={config.train.magic_prime}"
        )

    def __len__(self):
        return self.config.runtime.epoch_global_steps * self.config.train.micro_bsz * self.trainer.accumulate_grad_batches

    def __getitem__(self, idx):
        config = self.config
        rank = self.trainer.global_rank
        epoch = self.trainer.current_epoch
        world_size = self.trainer.world_size

        magic_prime = config.train.magic_prime
        assert config.train.train_stage >= 0

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * self.ctx_len

        # extra safety: clamp to max valid aligned offset
        if i > self.max_i:
            i = self.max_i
        if i < 0:
            i = 0

        dix = self.data.get(idx=0, offset=i, length=self.req_len).astype(int)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

class MyDataset(Dataset):
    def __init__(self, config:TrainerCLI_Config, trainer:Trainer):
        self.config = config
        self.trainer = trainer

        assert config.train.data_type == "binidx"
        self.vocab_size = config.model.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(config.train.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.samples_per_epoch = config.runtime.epoch_global_steps * config.runtime.global_step_bsz
        #assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## training stage {config.train.train_stage} ##########")
        dataset_slot = self.data_size // config.model.ctx_len
        assert config.train.my_exit_tokens <= self.data_size
        assert MaybeIsPrime(config.train.magic_prime)
        assert config.train.magic_prime % 3 == 2
        assert config.train.magic_prime / dataset_slot > 0.99 and config.train.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.config.runtime.epoch_global_steps * self.config.train.micro_bsz * self.trainer.accumulate_grad_batches

    def __getitem__(self, idx):
        config = self.config
        rank = self.trainer.global_rank
        epoch = self.trainer.current_epoch
        world_size = self.trainer.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = config.model.ctx_len
        req_len = ctx_len + 1
        magic_prime = config.train.magic_prime
        data = self.data

        assert config.train.train_stage >= 0
        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")

        dix = data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y

class MMapDataset(Dataset):
    def __init__(self, data_prefix, ctx_len):
        self.data = MMapIndexedDataset(data_prefix)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        self.req_len = ctx_len + 1
        self.count = self.data_size // self.req_len

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        data_chunk = self.data.get(idx=0, offset=idx * self.req_len, length=self.req_len).astype(int)
        input_ids = torch.tensor(data_chunk, dtype=torch.long)
        x = input_ids[:-1]
        y = input_ids[1:]

        return x, y
