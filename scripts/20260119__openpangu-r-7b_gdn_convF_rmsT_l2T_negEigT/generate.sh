source "$(dirname "$0")/vars.sh"
echo $RUN_NAME



# Original model
# INIT_PTH_PATH="${out_pths_dir}/${STEP0_DIR}/rwkv-init.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python generate.py \
#     -c ${qwen_yaml} \
#     -c ${all_orig_attn} \
#     --path  ${INIT_PTH_PATH} \
#     --tokenizer_path ${tokenizer}



# INIT_PTH_PATH="${out_pths_dir}/${STEP0_DIR}/rwkv-init.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python generate.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path  ${INIT_PTH_PATH} \
#     --tokenizer_path ${tokenizer}

# STEP0_PTH_PATH="${out_pths_dir}/${STEP0_DIR}/rwkv-final.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python generate.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path  ${STEP0_PTH_PATH} \
#     --tokenizer_path ${tokenizer}

# STEP1_PTH_PATH="${out_pths_dir}/${STEP1_DIR}/rwkv-final.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python generate.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path  ${STEP1_PTH_PATH} \
#     --tokenizer_path ${tokenizer}

# STEP2_PTH_PATH="${out_pths_dir}/${STEP2_DIR}/rwkv-final.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python generate.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path  ${STEP2_PTH_PATH} \
#     --tokenizer_path ${tokenizer}


CUDA_VISIBLE_DEVICES="0"

# start=0
# end=20
# stride=1
# checkpoints=('init')
# for i in $(seq $start $stride $end); do
#     checkpoints+=("$i")
# done
# checkpoints+=('final')

# # for idx in "${checkpoints[@]}"; do
# #     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# #     python generate.py \
# #         -c ${qwen_yaml} \
# #         -c ${la_yaml} \
# #         --path "${STEP0_PTH_PATH}/ckpt-${idx}.pth" \
# #         --tokenizer_path ${tokenizer}
# # done

# for idx in "${checkpoints[@]}"; do
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#     python generate.py \
#         -c ${qwen_yaml} \
#         -c ${la_yaml} \
#         --path "${STEP1_PTH_PATH}/ckpt-${idx}.pth" \
#         --tokenizer_path ${tokenizer}
# done
# path=/work/aman/hub/models--Qwen--Qwen3-8B-Base
path=/work/jijun/model/openPangu-R-7B-2512
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python generate.py \
        -c ${openpangu_yaml} \
        -c ${openpangu_all_orig_attn_yaml} \
        --path "${path}" \
        --tokenizer_path ${tokenizer}
