source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

STEP0_DIR="${RUN_NAME}-1"
STEP1_DIR="${RUN_NAME}-2"
STEP2_DIR="${RUN_NAME}-4"

qwen_yaml=configs/qwen7b.yaml
qwerky7_yaml="configs/20251127/qwerky7.yaml"
tokenizer="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES="1"

# INIT_PTH_PATH="${out_pths_dir}/${STEP0_DIR}/rwkv-init.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python generate.py \
#     -c ${qwen_yaml} \
#     -c ${qwerky7_yaml} \
#     --path  ${INIT_PTH_PATH} \
    # --tokenizer_path ${tokenizer}

STEP0_PTH_PATH="${out_pths_dir}/${STEP0_DIR}/rwkv-final.pth"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python generate.py \
    -c ${qwen_yaml} \
    -c ${qwerky7_yaml} \
    --path  ${STEP0_PTH_PATH} \
    --tokenizer_path ${tokenizer} \
    --is_instruct 1

STEP1_PTH_PATH="${out_pths_dir}/${STEP1_DIR}/rwkv-final.pth"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python generate.py \
    -c ${qwen_yaml} \
    -c ${qwerky7_yaml} \
    --path  ${STEP1_PTH_PATH} \
    --tokenizer_path ${tokenizer} \
    --is_instruct 1

STEP2_PTH_PATH="${out_pths_dir}/${STEP2_DIR}/rwkv-final.pth"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python generate.py \
    -c ${qwen_yaml} \
    -c ${qwerky7_yaml} \
    --path  ${STEP2_PTH_PATH} \
    --tokenizer_path ${tokenizer} \
    --is_instruct 1
