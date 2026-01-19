#!/bin/bash
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
export gpu_num=1
export CUDA_VISIBLE_DEVICES=${gpu_num} #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7
export BASE="${1:-/work/${USERNAME}}/radlads"
export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503

bsz=4

# tasks=gsm8k
# tasks=winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race,gsm8k
tasks=winogrande
# tasks=mmlu_pro,cmmlu,ceval,gpqa_diamond_zeroshot,aime24,aime25
# tasks=\
# gsm8k,\
# winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race,\
# mmlu_pro,cmmlu,ceval,gpqa_diamond_zeroshot,aime24,aime25


# # Originally pretrained ##
# ORIGINAL_MODEL_NAME="openPangu-R-7B-2512"
# ORIGINAL_MODEL_PATH="Qwen/${ORIGINAL_MODEL_NAME}"
# MODEL_PATH="${BASE}/pths/${ORIGINAL_MODEL_NAME}/pretrained.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python run_lm_eval.py \
#     --path ${ORIGINAL_MODEL_PATH} \
#     -c ${qwen_yaml} \
#     --is_pretrained yes \
#     --bsz ${bsz} \
#     --tasks ${tasks}
#     # --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa

# Originally pretrained ##
ORIGINAL_MODEL_NAME="openPangu-R-7B-2512"
# ORIGINAL_MODEL_PATH="Qwen/${ORIGINAL_MODEL_NAME}"
MODEL_PATH="/work/aman/openPangu-R-7B-2512"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python run_lm_eval.py \
    --path ${MODEL_PATH} \
    -c ${qwen_yaml} \
    --is_pretrained yes \
    --bsz ${bsz} \
    --tasks ${tasks}
    # --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa









# checkpoints=()
# # checkpoints+=('init')
# # start=45
# # end=47
# # stride=12
# # for i in $(seq $start $stride $end); do
# #     checkpoints+=("$i")
# # done
# # checkpoints+=('45')
# checkpoints+=('final')
# # tasks=mmlu,lambada_openai,hellaswag


# for i in "${checkpoints[@]}"; do
#     CKPT_PATH="${STEP2_PTH_PATH}/ckpt-${i}.pth"
#     if [ -f "$CKPT_PATH" ]; then
#         echo "Evaluating checkpoint: $CKPT_PATH"
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         python run_lm_eval.py \
#             -c ${qwen_yaml} \
#             -c ${la_yaml} \
#             --path "${CKPT_PATH}" \
#             --is_pretrained no \
#             --bsz ${bsz} \
#             --tokenizer_name ${tokenizer} \
#             --tasks ${tasks}
#             # gsm8k
#     else
#         echo "Checkpoint does not exist: $CKPT_PATH, skipping."
#     fi
# done

# # for i in "${checkpoints[@]}"; do
# #     CKPT_PATH="${STEP0_PTH_PATH}/ckpt-${i}.pth"
# #     if [ -f "$CKPT_PATH" ]; then
# #         echo "Evaluating checkpoint: $CKPT_PATH"
# #         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# #         python run_lm_eval.py \
# #             -c ${qwen_yaml} \
# #             -c ${la_yaml} \
# #             --path "${CKPT_PATH}" \
# #             --is_pretrained no \
# #             --bsz ${bsz} \
# #             --tokenizer_name ${tokenizer} \
# #             --tasks ${tasks}
# #             # gsm8k
# #     else
# #         echo "Checkpoint does not exist: $CKPT_PATH, skipping."
# #     fi
# # done


# # for i in "${checkpoints[@]}"; do
# #     CKPT_PATH="${STEP1_PTH_PATH}/ckpt-${i}.pth"
# #     if [ -f "$CKPT_PATH" ]; then
# #         echo "Evaluating checkpoint: $CKPT_PATH"
# #         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# #         python run_lm_eval.py \
# #             -c ${qwen_yaml} \
# #             -c ${la_yaml} \
# #             --path "${CKPT_PATH}" \
# #             --is_pretrained no \
# #             --bsz ${bsz} \
# #             --tokenizer_name ${tokenizer} \
# #             --tasks ${tasks}
# #             # gsm8k
# #     else
# #         echo "Checkpoint does not exist: $CKPT_PATH, skipping."
# #     fi
# # done



# python ~/exp.py --gpus ${gpu_num}
