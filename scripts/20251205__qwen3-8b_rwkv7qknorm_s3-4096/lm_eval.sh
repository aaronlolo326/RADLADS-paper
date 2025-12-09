#!/bin/bash
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
export CUDA_VISIBLE_DEVICES=1 #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7
export USERNAME=$(whoami)
export BASE="${1:-/work/${USERNAME}}/radlads"
export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503

# logs_dir="logs/logs_1201"
# mkdir -p ${logs_dir}

# ## Originally pretrained ##
# ORIGINAL_MODEL_NAME="Qwen2.5-7B-Instruct"
# ORIGINAL_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
# MODEL_PATH="${BASE}/pths/${ORIGINAL_MODEL_NAME}/pretrained.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python run_lm_eval.py \
#     --path ${ORIGINAL_MODEL_PATH} \
#     --is_pretrained yes \
#     --bsz 24 \
#     --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
#     # gsm8k
# #     # --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa



MODEL_PATH="${BASE}/out/pths/${RUN_NAME}/rwkv-final.pth"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python run_lm_eval.py \
    -c configs/qwen7b.yaml \
    -c configs/${RUN_NAME}/qwerky7.yaml \
    --path ${MODEL_PATH} \
    --is_pretrained no \
    --bsz 16 \
    --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
    # gsm8klambada_openai,mmlu

MODEL_NAME="${RUN_NAME}-2"
MODEL_PATH="${BASE}/out/pths/${MODEL_NAME}/rwkv-final.pth"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python run_lm_eval.py \
    -c configs/qwen7b.yaml \
    -c configs/${RUN_NAME}/qwerky7.yaml \
    --path ${MODEL_PATH} \
    --is_pretrained no \
    --bsz 16 \
    --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
    # gsm8klambada_openai,mmlu


MODEL_NAME="${RUN_NAME}-4-4k"
MODEL_PATH="${BASE}/out/pths/${MODEL_NAME}/rwkv-final.pth"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python run_lm_eval.py \
    -c configs/qwen7b.yaml \
    -c configs/${RUN_NAME}/qwerky7.yaml \
    --path ${MODEL_PATH} \
    --is_pretrained no \
    --bsz 16 \
    --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
    # gsm8k








# bs=48
# TASK="harness_6tasks"
# ORIGINAL_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
# ORIGINAL_MODEL_NAME="Qwen2.5-7B-Instruct"
# export OUTPUT_DIR="results/${MODEL_NAME}/${TASK}"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
#     --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#     --model hf \
#     --model_args pretrained=${ORIGINAL_MODEL_PATH},trust_remote_code=True,dtype=bfloat16  \
#     --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa \
#     --device cuda \
#     --trust_remote_code \
#     --batch_size ${bs} \
#     --output_path $OUTPUT_DIR \
#     --log_samples \
#     --trust_remote_code


# bs=48
# TASK="harness_6tasks"
# MODEL_NAME="${BASE_MODEL_NAME}-4"
# MODEL_PATH="${BASE}/out/pths/${MODEL_NAME}/rwkv-final.pth"
# export OUTPUT_DIR="results/${MODEL_NAME}/${TASK}"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
#     --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#     --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16 \
#     --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa \
#     --device cuda \
#     --trust_remote_code \
#     --batch_size ${bs} \
#     --output_path $OUTPUT_DIR \
#     --log_samples \
#     --trust_remote_code
