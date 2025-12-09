source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

qwen_yaml=configs/qwen3-8b.yaml
qwerky7_yaml="configs/${RUN_NAME}/qwerky7.yaml"
all_orig_attn="configs/${RUN_NAME}/qwen3-8b_all_orig_attn.yaml"
tokenizer="Qwen/Qwen3-8B-Base"

CUDA_VISIBLE_DEVICES="0"


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


start=4
end=19
stride=5
checkpoints=('init')
for i in $(seq $start $stride $end); do
    checkpoints+=("$i")
done
checkpoints+=('final')

for idx in "${checkpoints[@]}"; do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python generate.py \
        -c ${qwen_yaml} \
        -c ${qwerky7_yaml} \
        --path "${STEP1_PTH_PATH}/rwkv-${idx}.pth" \
        --tokenizer_path ${tokenizer}
done