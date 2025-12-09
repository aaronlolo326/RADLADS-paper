RUN_NAME="20251205__qwen3-8b_rwkv7qknorm_s3-4096"

export USERNAME=$(whoami)
pths_dir=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

STEP0_DIR="${RUN_NAME}-1"
STEP1_DIR="${RUN_NAME}-2"
STEP2_DIR="${RUN_NAME}-4-2k"


STEP0_PTH_PATH="${out_pths_dir}/${STEP0_DIR}"
STEP1_PTH_PATH="${out_pths_dir}/${STEP1_DIR}"
STEP2_PTH_PATH="${out_pths_dir}/${STEP2_DIR}"