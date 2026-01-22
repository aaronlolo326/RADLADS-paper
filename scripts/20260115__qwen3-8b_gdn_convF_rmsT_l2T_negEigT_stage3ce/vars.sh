CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 #0,1,3,4,5,6,7

export USERNAME=$(whoami)
pths_dir=/work/${USERNAME}/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

RUN_NAME="$(basename "$(dirname "$0")")"

STEP0_DIR="${RUN_NAME}-1"
STEP1_DIR="${RUN_NAME}-2"
STEP2_DIR="${RUN_NAME}-4-2k"


STEP0_PTH_PATH="${out_pths_dir}/${STEP0_DIR}"
STEP1_PTH_PATH="${out_pths_dir}/${STEP1_DIR}"
STEP2_PTH_PATH="${out_pths_dir}/${STEP2_DIR}"

qwen_yaml=configs/qwen3-8b.yaml
la_yaml="configs/${RUN_NAME}/gdn.yaml"
all_orig_attn="configs/${RUN_NAME}/qwen3-8b_all_orig_attn.yaml"
tokenizer="Qwen/Qwen3-8B-Base"