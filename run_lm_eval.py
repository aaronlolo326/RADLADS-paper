########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
#
import os, sys, types, json, math, time
import numpy as np
from pprint import pprint
np.set_printoptions(precision=4, suppress=True, linewidth=200)

#import transformers # just for a bugfix for 0.4.2 of lm_eval
from transformers import AutoModelForCausalLM

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

from pydoc import locate

from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

#from src.pipeline import PIPELINE, PIPELINE_ARGS

from transformers.modeling_utils import load_state_dict, load_sharded_checkpoint

from lm_eval import tasks, evaluator, utils
from lm_eval.api.model import TemplateLM

import datasets
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

from tqdm import tqdm

########################################################################################################

from dataclasses import dataclass
import typing

from huggingface_hub import login

# Optionally get your HuggingFace token from environment or prompt
hf_token = os.environ.get("HF_TOKEN")
if hf_token is not None:
    login(token=hf_token)
else:
    try:
        login()
    except Exception as e:
        print("Warning: HuggingFace login failed or was skipped. You may need to manually authenticate if you download private datasets.")



use_cache = True
precision = '32'

# Switch-case style logic for CUDA_VISIBLE_DEVICES and configs (1-5)
device = int(os.environ.get("CUDA_VISIBLE_DEVICES", None))  # Choose device/config by setting $DEVICE=1..5
print (f"{device=}")
# if device == 1:
#     use_batch_all_time = True
#     pad8mul = True
#     bsz = 16
# elif device == 2:
#     use_batch_all_time = True
#     pad8mul = False
#     bsz = 16
# elif device == 3:
#     use_batch_all_time = True
#     pad8mul = True
#     bsz = 1
# elif device == 4:
#     use_batch_all_time = True
#     pad8mul = False
#     bsz = 1
# elif device == 5:
#     use_batch_all_time = False
#     pad8mul = False
#     bsz = 1
# else:
use_batch_all_time = True
pad8mul = True
bsz = 24

@dataclass(kw_only=True)
class CLI_Config:
    path: str
    tasks: str = 'lambada_openai' # arc_challenge, arc_easy, headqa, openbookqa, hellaswag, winogrande, piqa, record, copa, storycloze_2016
    bsz: int = bsz
    precision: int | str = precision
    num_fewshot: int | None = None
    seed: int | None = None
    recurrent: int = 1
    train:typing.Any = None
    model: Model_Config | None = None
    is_pretrained: str = "no"
    tokenizer_name: str | None = None
    limit: int | None = None
    log_path: str | None = None
    log_activations: int = 0
    activation_layers: str | None = None  # Comma-separated list of layer indices, e.g., "0,1,2" or "all"
    activation_last_token_only: bool = True  # Only capture last token activations to save memory

print (sys.argv)
config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
if errors != '':
    print(errors)
    exit()
pprint (config)

model_path = config.path

## check existing results file
result_subdir = "__".join(model_path.replace(".pth", "").split("/")[-2:])
results_dir = f'results/{result_subdir}'
os.makedirs(results_dir, exist_ok=True)
if config.log_path:
    logits_dir = f'{config.log_path}'
    os.makedirs(logits_dir, exist_ok=True)

# Load existing results if they exist, then merge with new results
# if device == 0:
results_file = f'{results_dir}/lm_eval_results.json'
results_file_full = f'{results_dir}/lm_eval_results_full.json'
# else:
#     results_file = f'{results_dir}/20250107_results_batched{use_batch_all_time}_bsz{config.bsz}_8mul{pad8mul}_cache{use_cache}_prec{config.precision}.json'
#     results_file_full = f'{results_dir}/20250107_full_batched{use_batch_all_time}_bsz{config.bsz}_8mul{pad8mul}_cache{use_cache}_prec{config.precision}.json'

existing_results = {}
if config.limit is None and os.path.exists(results_file):
    with open(results_file, 'r') as f:
        existing_results = json.load(f)
eval_tasks = config.tasks.split(',')

print (f"{eval_tasks=}")
tasks_existed = set(existing_results.keys())
eval_tasks = list(set(eval_tasks) - tasks_existed)
print (f"{tasks_existed=}")
print (f"{eval_tasks=}")
if not eval_tasks:
    print ("all tasks evaluated in results dir; nothing to evaluate")
    exit()

config.train = None # to avoid clashes with training configs

## Set for linearization ##
if config.is_pretrained == "no":
    os.environ["RWKV_MODEL_TYPE"] = config.model.tmix
    os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)
    attention_type = str(config.model.attention_type)
    if attention_type == 'rwkv7':
        attention_type = 'rwkv7_fla_fused_recurrent'
    os.environ["RWKV_ATTENTION_TYPE"] = attention_type


# Setup the model
from src.model import Transformer
from safetensors.torch import load_file

# avoid 1000 huggingface warnings "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...""
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print(f'Loading model - {model_path}')
if config.is_pretrained == "yes":
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,          # or float16
        attn_implementation="flash_attention_2",
        device_map="cuda")
elif config.is_pretrained == "no":
    classname = config.model.classname
    if config.path.lower().endswith('.safetensors'):
        load_dict = load_file(config.path)
    else:
        load_dict = torch.load(model_path, mmap=True)
    if any([
        (classname.startswith('qwen2') or config.model.tmix.startswith('qwen2')) and config.model.n_embd < 3584,
        (classname.startswith('qwen3') or config.model.tmix.startswith('qwen3')) and config.model.n_embd < 4096,
    ]):
        load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
        
    with torch.device('meta'):
        if classname != '':
            model_classpath = f'models.{classname}.Model_{classname}'
            model_factory = locate(model_classpath)
            if model_factory is None:
                print(f"Unsupported model type: {model_classpath}")
                exit(0)
            model = model_factory(config)
        #elif config.model.tmix.startswith('qwen2'):
        #    model = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in config.model.tmix, **qwen_cfg), config)
        else:
            model = Transformer(config)

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
model = model.to(device=device, dtype=torch.bfloat16)
model.eval()

#pipeline = PIPELINE(model, "rwkv_vocab_v20230424")



#RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
#STOP_TOKEN = RWKV_PAD + pipeline.tokenizer.encode('\n\n') # we will use '\n\n' as STOP
# RWKV_PAD = [0] # you can try using [0] as pad

RWKV_PAD = [] # means do not pad

print('RWKV_PAD', RWKV_PAD)
#print('STOP_TOKEN', STOP_TOKEN)

########################################################################################################

logitBuf = {}
correctBuf = {}

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        #self.eos_token_id = 0
        self.eos_token_id = tokenizer.eos_token_id

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

class EvalHarnessAdapter(TemplateLM):
    # bugfix for lm_eval 0.4.2
    AUTO_MODEL_CLASS = AutoModelForCausalLM

    def __init__(self, batch_size_per_gpu, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size_per_gpu
        self.request_counter = 0

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "`loglikelihood_rolling` is currently not supported"
        )
    
    @torch.no_grad()
    def greedy_generate(self, ctx, state=None, key_values=None):

        logits_list = []
        if config.is_pretrained == "yes":
            inputs = self.tokenizer([ctx], return_tensors="pt").to(model.device)
            inputs_len = inputs['input_ids'].shape[1]
            outputs = model.generate(**inputs, max_new_tokens=self.max_gen_toks)
            out_str = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
        else:

            last_model_state = state
            past_key_values = key_values

            STOP_TOKEN = [self.tokenizer.eos_token_id]

            all_tokens = []
            out_last = 0
            out_str = ''

            
            max_gen_toks = self.max_gen_toks
            # max_gen_toks = 1
            for i in range(max_gen_toks):
                tokens = self.tokenizer.encode(ctx) if i == 0 else [token]
                chunk_size = self.max_length # original
                chunk_size = len(tokens)
                # breakpoint()
                # hei: below loop is for chunk_parallel?
                while len(tokens) > 0:
                    # print (f"greedy_generate {len(tokens)=} {tokens[:self.max_length]=}")
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
                tmp = self.tokenizer.decode(all_tokens[out_last:])
                if '\ufffd' not in tmp: # is valid utf-8 string?
                    out_str += tmp
                    out_last = i + 1
        # print (f"{ctx=}")
        # print (out_str)
        return out_str, logits_list

    @torch.no_grad()
    def batch_greedy_generate(self, input_texts, state=None, key_values=None):


        logits_list = []
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt').to(device)
        if config.is_pretrained == "yes":
            # breakpoint()
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_gen_toks,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
            # breakpoint()
            out_str_batch = [
                tokenizer.decode(outputs[i][len(inputs['input_ids'][i]):], skip_special_tokens=True)
                for i in range(len(input_texts))
            ]
        else:

            last_model_state = state
            past_key_values = key_values

            STOP_TOKEN = [self.tokenizer.eos_token_id]
            batch_input_ids = inputs['input_ids']

            is_eos_generated = [False for _ in range(len(batch_input_ids))]

            all_tokens_batch = [[] for _ in range(len(batch_input_ids))]
            out_str_batch = ['' for _ in range(len(batch_input_ids))]
            out_last = 0

            input_ids_padded_batch_list = []
            attention_mask_batch_list = []
            # stack and pad to longest
            maxlen = max([len(x) for x in batch_input_ids])
            if pad8mul:
                maxlen = (maxlen + 7) // 8 * 8 # round pad size up to nearest 8 for better GPU usage
            for i in range(len(batch_input_ids)):
                padded_len = (maxlen - len(batch_input_ids[i]))
                # breakpoint()
                input_ids_padded_batch_list.append(
                    F.pad(
                        batch_input_ids[i],
                        (padded_len, 0),
                        value=model.config.model.vocab_padding_idx # self.tokenizer.special_tokens_map["pad_token"]
                    )
                )
                attention_mask_batch_list.append([0] * padded_len + [1] * len(batch_input_ids[i]))
            input_ids_padded_batch = torch.stack(input_ids_padded_batch_list, dim=0)
            input_ids_padded_batch = input_ids_padded_batch.to(device)
            attention_mask_batch = torch.tensor(attention_mask_batch_list, dtype=torch.long)
            attention_mask_batch = attention_mask_batch.to(device)
            last_token_batch = None
            
            max_gen_toks = self.max_gen_toks
            # max_gen_toks = 1
            for i in range(max_gen_toks):
                tokens = input_ids_padded_batch if i == 0 else last_token_batch
                # breakpoint()
                # print (f"batch_greedy_generate {tokens=}")
                results = model.forward(tokens.to(device), last_model_state=last_model_state, past_key_values=past_key_values, attention_mask=attention_mask_batch, use_cache=use_cache)
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
                        tmp = self.tokenizer.decode(last_token_batch[i][-1])
                        if '\ufffd' not in tmp: # is valid utf-8 string?
                            out_str_batch[i] += tmp

        # print (f"{out_str_batch=}")
        # breakpoint()
        return out_str_batch, logits_list
    
    @torch.no_grad()
    def generate_until(self, requests):
        """
        Generate until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.

        TODO: batched / data parallel generation

        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """


        res = []
        ###batch
        self.tokenizer.padding_side = 'left'
        # get only the args from each Instance object
        reqs = [req.args for req in requests]
        def _collate(x):
            encoded = self.tokenizer.encode(x[0])
            return (-len(encoded), x[0])
        
        reord = utils.Reorderer(reqs, _collate)
        sorted_reqs = reord.get_reordered()
        sorted_input_texts, sorted_gen_kwargs = zip(*sorted_reqs)
        # sort requests by descending total length, so we batch together groups that have similar padded sizes, descending so we OOM early if at all
        B = self.batch_size_per_gpu

        gen_switch = (B >= 1) if use_batch_all_time else (B > 1)
        if gen_switch:

            for nb in tqdm(range(0, len(sorted_reqs), B), "Running batched greedy generation"):
                ne = min(nb+B, len(sorted_reqs))
                out_str_batch, logits_list_batch = self.batch_greedy_generate(sorted_input_texts[nb:ne])
                
                for i in range(nb, ne):
                    res.append(out_str_batch[i - nb])
                    for term in sorted_gen_kwargs[i]['until']:
                        res[i] = res[i].split(term)[0]
        ###
        
        # print ("\n"*6)
        # print ("================="*30)
        # print ("\n"*6)

        ## below is original inference impl
        else:
            for context, gen_kwargs in tqdm(reord.get_reordered(), "Running greedy generation"):
                out_str, logits_list = self.greedy_generate(context)
                
                for term in gen_kwargs['until']:
                    out_str = out_str.split(term)[0]
                res.append(out_str)

        return reord.get_original(res)


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # FIXME - is this correct? is it even used? didn't seem to be
        # FIXME - we really should support recurrent inference
        return config.model.ctx_len
        # try:
        #     return self.gpt2.config.n_ctx
        # except AttributeError:
        #     # gptneoconfig doesn't have n_ctx apparently
        #     return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        # FIXME - is this correct? is it even used? didn't seem to be, since the model itself complained at 512 when returning 256 here
        # FIXME - we really should support recurrent inference
        return config.model.ctx_len

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        res = []

        # sort requests by descending total length, so we batch together groups that have similar padded sizes, descending so we OOM early if at all
        rq_indices = sorted(range(len(requests)),key=lambda i: len(requests[i][1])+len(requests[i][2]), reverse=True)

        res = [None for _ in range(len(requests))]

        B = self.batch_size_per_gpu
        for nb in range(0, len(requests), B):
            ne = min(nb+B, len(requests))

            # stack and pad to longest
            batched_inputs = []
            batch_info = []
            maxlen = 0
            for i in range(nb, ne):
                rq_index = rq_indices[i]
                request = requests[rq_index]
                q = RWKV_PAD + request[1]
                src = q + request[2]
                input = torch.tensor(src, dtype=torch.long, device=device, requires_grad=False)
                batched_inputs.append( input )
                batch_info.append((len(q), len(src), rq_index))
                maxlen = max(maxlen, len(src))

            maxlen = (maxlen + 7) // 8 * 8 # round pad size up to nearest 8 for better GPU usage
            for i in range(len(batched_inputs)):
                batched_inputs[i] = F.pad(batched_inputs[i], (0, maxlen - batched_inputs[i].size(0)))
            batched_inputs = torch.stack(batched_inputs, dim=0)

            results = model.forward(batched_inputs, None)
            if isinstance(results, tuple):
                logits = results[0]
                #next_model_state = results[1]
            elif isinstance(results, torch.Tensor):
                logits = results
                #next_model_state = last_model_state
            else:
                logits = results.logits
                #next_model_state = last_model_state
                

            batched_logprobs = F.log_softmax(logits, dim=-1)
            # Check if per-token argmax is exactly equal to continuation
            batched_greedy_toks = batched_logprobs.argmax(dim=-1)
            
            for i, info in enumerate(batch_info):
                q_len, src_len, rq_index = info
                a_len = src_len - q_len
                logprobs, a_toks, greedy_toks = batched_logprobs[i, q_len-1 : src_len-1], batched_inputs[i, q_len : src_len], batched_greedy_toks[i, q_len-1 : src_len-1]
                assert logprobs.size(0) == a_len
                assert a_toks.size(0) == a_len
                assert greedy_toks.size(0) == a_len
                max_equal = (greedy_toks == a_toks).all()
        
                # Obtain log-probs at the corresponding continuation ('answer') token indices
                logprobs = torch.gather(logprobs, 1, a_toks.unsqueeze(-1)).squeeze(-1)
                assert logprobs.size(0) == a_len
            
                # Answer: (log prob, is-exact-match)
                answer = (float(logprobs.sum()), bool(max_equal))

                # place the answer into the slot that matches the original request (this is important or lm_eval_harness will return bad results!!!)
                res[rq_index] = answer

            FREQ = 10 * B
            if nb % FREQ == 0:
                print(f'{nb//FREQ}/{len(requests)//FREQ}', end = ' ', flush=True)

        return res

if config.seed is None:
    config.seed = 1234 

# tokenizer = TokenizerWrapper(pipeline.tokenizer) # RWKV tokenizer
from transformers import AutoTokenizer
if config.is_pretrained == "yes":
    tokenizer = AutoTokenizer.from_pretrained(model_path)
elif config.tokenizer_name is not None:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
else:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')
print (f"{tokenizer=}")

RWKV_PAD = []

adapter = EvalHarnessAdapter(batch_size_per_gpu=config.bsz, tokenizer=tokenizer)
limit = config.limit
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=dtype):
        print (f"{dtype=}")
        results = evaluator.simple_evaluate(
	        model=adapter,
            # model_args="trust_remote_code=True",
	        tasks=eval_tasks,
	        #provide_description=False,
	        num_fewshot=config.num_fewshot,
	        limit=limit,
	        bootstrap_iters=10000,
	        numpy_random_seed = config.seed,
	        torch_random_seed = config.seed,
	        fewshot_random_seed = config.seed,
	    )

pprint ({
    k: v
    for k, v in results["results"].items()
    if k in eval_tasks
})

# Merge new results into existing (new results override existing entries)
existing_results.update(results['results'])

if limit is None:
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)

if limit is not None and limit <= 100:
    with open(results_file_full, 'w') as f:
        json.dump(results, f, indent=2)
