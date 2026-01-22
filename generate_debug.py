########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
#
import os, sys, types, json, math, time
import warnings
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

from pprint import pformat, pprint
#import transformers # just for a bugfix for 0.4.2 of lm_eval
from transformers import AutoModelForCausalLM

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

# Suppress warning from fla.ops.rwkv7.fused_recurrent about input tensor shape
warnings.filterwarnings('ignore', message='Input tensor shape suggests potential format mismatch')

from pydoc import locate

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

from transformers.modeling_utils import load_state_dict, load_sharded_checkpoint

########################################################################################################

from dataclasses import dataclass
import typing

@dataclass(kw_only=True)
class CLI_Config:
    path: str
    tokenizer_path: str = 'Qwen/Qwen2.5-72B-Instruct'
    # prompt:str = "The Chinese University of Hong"
    prompt:str = "Lard is a semi-solid white fat product obtained by rendering the fatty"
    max_len:int = 50
    attempts:int = 1
    precision: int | str = 'bf16'
    num_fewshot: int = 0
    seed: int | None = None
    train:typing.Any = None
    model: Model_Config
    is_instruct: int = 0

config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
if errors != '':
    print(errors)
    exit()
config.train = None # to avoid clashes with training configs

os.environ["RWKV_MODEL_TYPE"] = config.model.tmix
os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)
attention_type = str(config.model.attention_type)
if attention_type == 'rwkv7':
    attention_type = 'rwkv7_fla_fused_recurrent'
os.environ["RWKV_ATTENTION_TYPE"] = attention_type

model_path = config.path

# Setup the model
from src.model import Transformer
from safetensors.torch import load_file

# avoid 1000 huggingface warnings "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...""
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print(f'Loading model - {model_path}')
classname = config.model.classname
if config.path.lower().endswith('.safetensors'):
    load_dict = load_file(config.path)
else:
    load_dict = torch.load(model_path, mmap=True)
# pprint (list(load_dict.keys()))
# pprint (load_dict)
# pprint (f"{config.model.n_embd=}")
if any([
    (classname.startswith('qwen2') or config.model.tmix.startswith('qwen2')) and config.model.n_embd < 3584,
    (classname.startswith('qwen3') or config.model.tmix.startswith('qwen3')) and config.model.n_embd < 4096,
]):
    if 'lm_head.weight' not in load_dict:
        load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
    
with torch.device('meta'):
    if classname != '':
        model_classpath = f'models.{classname}.Model_{classname}'
        model_factory = locate(model_classpath)
        if model_factory is None:
            print(f"Unsupported model type: {model_classpath}")
            exit(0)
        model = model_factory(config)
        print (f"Loaded {model_classpath=}")
    #elif config.model.tmix.startswith('qwen2'):
    #    model = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in config.model.tmix, **qwen_cfg), config)
    else:
        model = Transformer(config)

# pprint(config)
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
tokenizer.padding_side = 'left'

if hasattr(model, 'configure_model'):
    model.configure_model()
model.load_state_dict(load_dict, assign=True, strict=False)

match config.precision:
    case 32:
        dtype = torch.float32
    case '32':
        dtype = torch.float32
    case 16:
        dtype = torch.float16
    case '16':
        dtype = torch.float16
    case 'bf16':
        dtype = torch.bfloat16
    case _:
        print("Bad precision type specified")
        exit()

device = 'cuda'
model = model.to(device=device, dtype=dtype)
model.eval()

if config.seed is None:
    config.seed = 1234 

from transformers import AutoTokenizer, Qwen2ForCausalLM, set_seed

set_seed(config.seed)

if config.is_instruct:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": config.prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
else:
    text = config.prompt

use_cache = True
precision = '32'
use_batch_all_time = True
pad8mul = True
bsz = 24
max_gen_toks = 32
device = int(os.environ.get("CUDA_VISIBLE_DEVICES", None))  # Choose device/config by setting $DEVICE=1..5
print (f"{device=}")


@torch.no_grad()
def greedy_generate(model, tokenizer, ctx, max_gen_toks=max_gen_toks, state=None, key_values=None):

    logits_list = []

    if True:

        last_model_state = state
        past_key_values = key_values

        STOP_TOKEN = [tokenizer.eos_token_id]

        out_strs = []

        for input_text in ctx:
            out_str = ''
            all_tokens = []
            out_last = 0
            # max_gen_toks = 1
            for i in range(max_gen_toks):
                tokens = tokenizer.encode(input_text) if i == 0 else [token]
                chunk_size = len(tokens)
                # breakpoint()
                # hei: below loop is for chunk_parallel?
                while len(tokens) > 0:
                    if i == 0:
                        print (f"greedy_generate {tokens[:chunk_size]=}")
                    results = model.forward(tokens[:chunk_size], last_model_state=state, past_key_values=past_key_values, use_cache=use_cache)
                    if isinstance(results, tuple):
                        logits = results[0]
                        last_model_state = results[1]
                        past_key_values = results[-1]
                    elif isinstance(results, torch.Tensor):
                        logits = results
                        #next_model_state = last_model_state
                    else:
                        logits = results.logits
                        last_model_state = results.model_state
                        past_key_values = results.key_values
                    tokens = tokens[chunk_size:]
                logits_list.append(logits[0,-1])
                token = logits[0,-1:].argmax().item()
                if token in STOP_TOKEN:
                    break
                all_tokens += [token]
                tmp = tokenizer.decode(all_tokens[out_last:])
                if '\ufffd' not in tmp: # is valid utf-8 string?
                    out_str += tmp
                    out_last = i + 1
            out_strs.append(out_str)
    # print (f"{ctx=}")
    # print (out_str)
    return out_strs, logits_list

@torch.no_grad()
def batch_greedy_generate(model, tokenizer, input_texts, max_gen_toks=max_gen_toks, state=None, key_values=None):


    logits_list = []
    print (tokenizer.padding_side)
    inputs = tokenizer(input_texts, padding=True, return_tensors='pt').to(device)
    if True:

        last_model_state = state
        past_key_values = key_values

        STOP_TOKEN = [tokenizer.eos_token_id]
        batch_input_ids = inputs['input_ids']

        is_eos_generated = [False for _ in range(len(batch_input_ids))]

        all_tokens_batch = [[] for _ in range(len(batch_input_ids))]
        out_str_batch = ['' for _ in range(len(batch_input_ids))]
        out_last = 0

        input_ids_padded_batch_list = []
        attention_mask_batch_list = []
        # stack and pad to longest
        maxlen = max([len(x) for x in batch_input_ids])
        original_maxlen = maxlen
        if pad8mul:
            maxlen = (maxlen + 7) // 8 * 8 # round pad size up to nearest 8 for better GPU usage
        print(f"DEBUG: batch_size={len(batch_input_ids)}, original_maxlen={original_maxlen}, padded_maxlen={maxlen}")
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for i in range(len(batch_input_ids)):
            padded_len = (maxlen - len(batch_input_ids[i]))
            # breakpoint()
            input_ids_padded_batch_list.append(
                F.pad(
                    batch_input_ids[i],
                    (padded_len, 0),
                    value=pad_token_id
                )
            )
            attention_mask_batch_list.append([0] * padded_len + [1] * len(batch_input_ids[i]))
        input_ids_padded_batch = torch.stack(input_ids_padded_batch_list, dim=0)
        input_ids_padded_batch = input_ids_padded_batch.to(device)
        attention_mask_batch = torch.tensor(attention_mask_batch_list, dtype=torch.long)
        attention_mask_batch = attention_mask_batch.to(device)
        last_token_batch = None
        
        # max_gen_toks = 1
        for i in range(max_gen_toks):
            tokens = input_ids_padded_batch if i == 0 else last_token_batch
            # breakpoint()
            if i == 0:
                print (f"batch_greedy_generate {tokens=}")
            # For generation steps (i > 0), we need a new attention mask for the single new token
            if i == 0:
                current_attention_mask = attention_mask_batch
            else:
                # For subsequent steps, all tokens are valid (newly generated)
                current_attention_mask = torch.ones((len(batch_input_ids), tokens.shape[1]), dtype=torch.long, device=device)
            results = model.forward(tokens.to(device), last_model_state=last_model_state, past_key_values=past_key_values, attention_mask=current_attention_mask, use_cache=use_cache)
            if isinstance(results, tuple):
                logits = results[0]
                last_model_state = results[1]
                past_key_values = results[-1]
            elif isinstance(results, torch.Tensor):
                logits = results
                #next_model_state = last_model_state
            else:
                logits = results.logits
                last_model_state = results.model_state
                past_key_values = results.key_values
            logits_list.append(logits[:,-1,:])  # Save logits for all instances in batch [batch_size, vocab_size]
            last_token_batch = logits[:,-1:,:].argmax(dim=-1) #, keepdims=True)
            is_eos_generated = [
                is_eos_generated[i] or last_token_batch[i][-1].detach().cpu() in STOP_TOKEN
                for i in range(len(last_token_batch))
            ]
            if all(is_eos_generated):
                break
            for i in range(len(out_str_batch)):
                # all_tokens_batch[i].append(last_token_batch[i])
                if not is_eos_generated[i]:
                    tmp = tokenizer.decode(last_token_batch[i][-1])
                    if '\ufffd' not in tmp: # is valid utf-8 string?
                        out_str_batch[i] += tmp

    # print (f"{out_str_batch=}")
    # breakpoint()
    return out_str_batch, logits_list

input_prompts = [
    "The most important thing in life is",
    "The most important thing in life is not",
    "Lard is a semi-solid white fat product obtained by rendering the fatty tissue of pigs, but lard is a semi-solid white fat product obtained by rendering the fatty"
]

from pprint import pprint

out_str_batch, logits_list = batch_greedy_generate(model, tokenizer, input_prompts[:2])
pprint (out_str_batch)
print (logits_list[0])

input_prompts = [
    "The most important thing in life is",
    "The most important thing in life is not",
    "Lard is a semi-solid white fat product obtained by rendering the fatty tissue of pigs, but lard is a semi-solid white fat product obtained by rendering the fatty",
    
    "Lard is a semi-solid white fat product obtained by rendering the fatty tissue of pigs, but lard is a semi-solid white fat product obtained by rendering the fatty",
    
    "Lard is a semi-solid white fat product obtained by rendering the fatty tissue of pigs, but lard is a semi-solid white fat product obtained by rendering the fatty"
]
out_str_batch2, logits_list2 = batch_greedy_generate(model, tokenizer, input_prompts)
pprint (out_str_batch2)
print (logits_list2[0])

print ("======================================")

out_str, logits_list = greedy_generate(model, tokenizer, input_prompts)
pprint (out_str)

breakpoint()

