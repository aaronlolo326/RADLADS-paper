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
    def __init__(self, data_prefixes):
        if isinstance(data_prefixes, str):
            data_prefixes = [data_prefixes]
        self.datasets = [MMapIndexedDataset(p) for p in data_prefixes]

        #Calculate individual token sizes
        self.sizes = [len(ds._bin_buffer) // ds._index._dtype_size for ds in self.datasets]
        self.cumulative_sizes = np.cumsum([0] + self.sizes)
        self.total_data_size = sum(self.sizes)

    def get(self, idx, offset, length):
        #Determine which dataset the requested offset falls into
        dataset_idx = np.searchsorted(self.cumulative_sizes, offset, side='right') - 1
        dataset_idx = max(0, min(dataset_idx, len(self.datasets) - 1))

        # Calculate local offset within that specific dataset
        local_offset = offset - self.cumulative_sizes[dataset_idx]

        #Boundary check
        return self.datasets[dataset_idx].get(idx=0, offset=local_offset, length=length)

class MyDatasetV2(Dataset):
    '''
    Support for ConcatMMapIndexedDataset
    '''
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
        
        dataset_slot = self.data_size // config.model.ctx_len
        assert config.train.my_exit_tokens <= self.data_size
        assert MaybeIsPrime(config.train.magic_prime)
        assert config.train.magic_prime % 3 == 2
        
        # Verify that the magic_prime matches the new total concatenated size
        assert config.train.magic_prime / dataset_slot > 0.99 and config.train.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.config.runtime.epoch_global_steps * self.config.train.micro_bsz * self.trainer.accumulate_grad_batches
    
    def __getitem__(self, idx):
        config = self.config
        rank = self.trainer.global_rank
        epoch = self.trainer.current_epoch
        world_size = self.trainer.world_size

        ctx_len = config.model.ctx_len
        req_len = ctx_len + 1
        magic_prime = config.train.magic_prime
        data = self.data

        assert config.train.train_stage >= 0
        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)
        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len

        # This now calls the ConcatMMapIndexedDataset.get()
        dix = data.get(idx=0, offset=i, length=req_len).astype(int)

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
