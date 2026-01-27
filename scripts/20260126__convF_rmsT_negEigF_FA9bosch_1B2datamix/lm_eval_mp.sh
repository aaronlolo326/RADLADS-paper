#!/bin/bash
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7
export BASE="${1:-/work/${USERNAME}}/radlads"
export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503

checkpoints=()
# checkpoints+=('init')
# start=0
# end=20
# stride=1
# for i in $(seq $start $stride $end); do
#     checkpoints+=("$i")
# done
checkpoints+=('final')

# tasks=mmlu,lambada_openai,hellaswag

# tasks=gsm8k,winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
checkpoint='final'
eval_pth_path=${STEP2_PTH_PATH}/ckpt-${checkpoint}.pth
eval_hf_path=${STEP2_HF_PATH}/ckpt-${checkpoint}
config_dir="configs/${RUN_NAME}"

# if [ ! -e "${eval_hf_path}" ]; then
  python convert_pth_to_hf.py \
    --input_path ${eval_pth_path} \
    --output_path ${eval_hf_path} \
    --config_dir ${config_dir} \
    --base_model ${base_model}
# fi

tasks=winogrande,arc_easy,piqa,arc_challenge,mmlu,gsm8k,mathqa,race
accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
   --model hf \
   --model_args pretrained=${eval_hf_path},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=32768 \
   --tasks ${tasks} \
   --device cuda \
   --trust_remote_code \
   --batch_size 1 \
   --output_path $lm_eval_output_path \
   --log_samples \
   --trust_remote_code

accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
   --model hf \
   --model_args pretrained=${eval_hf_path},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=32768 \
   --tasks longbench \
   --trust_remote_code \
   --device cuda \
   --num_fewshot 0 \
   --batch_size 1 \
   --output_path $lm_eval_output_path \
   --log_samples \
   --seed 1234

accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
   --model hf \
   --model_args pretrained=${eval_hf_path},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,torch_dtype=bfloat16,max_length=32768 \
   --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3 \
   --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
   --device cuda \
   --trust_remote_code \
   --batch_size 1 \
   --output_path $lm_eval_output_path \
   --log_samples \
   --seed 1234
   # --gen_kwargs '{"max_new_tokens": 128}' \




# python ~/exp.py --gpus ${CUDA_VISIBLE_DEVICES}

