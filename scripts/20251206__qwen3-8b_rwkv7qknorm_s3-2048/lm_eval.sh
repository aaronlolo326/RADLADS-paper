#!/bin/bash
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
export CUDA_VISIBLE_DEVICES=0 #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7
export BASE="${1:-/work/${USERNAME}}/radlads"
export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503

# logs_dir="logs/logs_1201"
# mkdir -p ${logs_dir}

# ## Originally pretrained ##
# ORIGINAL_MODEL_NAME="Qwen3-8B-Base"
# ORIGINAL_MODEL_PATH="Qwen/${ORIGINAL_MODEL_NAME}"
# MODEL_PATH="${BASE}/pths/${ORIGINAL_MODEL_NAME}/pretrained.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python run_lm_eval.py \
#     --path ${ORIGINAL_MODEL_PATH} \
#     --is_pretrained yes \
#     --bsz 16 \
#     --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
#     # gsm8k
# #     # --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa



start=0
end=20
stride=1
checkpoints=('init')
for i in $(seq $start $stride $end); do
    checkpoints+=("$i")
done
checkpoints+=('final')

for i in "${checkpoints[@]}"; do
    CKPT_PATH="${STEP0_PTH_PATH}/rwkv-${i}.pth"
    if [ -f "$CKPT_PATH" ]; then
        echo "Evaluating checkpoint: $CKPT_PATH"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        python run_lm_eval.py \
            -c ${qwen_yaml} \
            -c ${qwerky7_yaml} \
            --path "${CKPT_PATH}" \
            --is_pretrained no \
            --bsz 16 \
            --tokenizer_name ${tokenizer} \
            --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
            # gsm8k
    else
        echo "Checkpoint does not exist: $CKPT_PATH, skipping."
    fi
done

for i in "${checkpoints[@]}"; do
    CKPT_PATH="${STEP1_PTH_PATH}/rwkv-${i}.pth"
    if [ -f "$CKPT_PATH" ]; then
        echo "Evaluating checkpoint: $CKPT_PATH"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        python run_lm_eval.py \
            -c ${qwen_yaml} \
            -c ${qwerky7_yaml} \
            --path "${CKPT_PATH}" \
            --is_pretrained no \
            --bsz 16 \
            --tokenizer_name ${tokenizer} \
            --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
            # gsm8k
    else
        echo "Checkpoint does not exist: $CKPT_PATH, skipping."
    fi
done

for i in "${checkpoints[@]}"; do
    CKPT_PATH="${STEP2_PTH_PATH}/rwkv-${i}.pth"
    if [ -f "$CKPT_PATH" ]; then
        echo "Evaluating checkpoint: $CKPT_PATH"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        python run_lm_eval.py \
            -c ${qwen_yaml} \
            -c ${qwerky7_yaml} \
            --path "${CKPT_PATH}" \
            --is_pretrained no \
            --bsz 16 \
            --tokenizer_name ${tokenizer} \
            --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
            # gsm8k
    else
        echo "Checkpoint does not exist: $CKPT_PATH, skipping."
    fi
done

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python run_lm_eval.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path "${STEP0_PTH_PATH}/rwkv-final.pth" \
#     --is_pretrained no \
#     --bsz 16 \
#     --tokenizer_name ${tokenizer} \
#     --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
#     # gsm8k

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python run_lm_eval.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path "${STEP1_PTH_PATH}/rwkv-final.pth" \
#     --is_pretrained no \
#     --bsz 16 \
#     --tokenizer_name ${tokenizer} \
#     --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
#     # gsm8k

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python run_lm_eval.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path "${STEP2_PTH_PATH}/rwkv-final.pth" \
#     --is_pretrained no \
#     --bsz 16 \
#     --tokenizer_name ${tokenizer} \
#     --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race
#     # gsm8k






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
