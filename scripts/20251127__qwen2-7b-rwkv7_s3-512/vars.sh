export USERNAME=$(whoami)
pths_dir=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

RUN_NAME="20251127__qwen2-7b-rwkv7_s3-512"

STEP0_DIR="${RUN_NAME}-1"
STEP1_DIR="${RUN_NAME}-2"
STEP2_DIR="${RUN_NAME}-4"


STEP0_PTH_PATH="${out_pths_dir}/${STEP0_DIR}"
STEP1_PTH_PATH="${out_pths_dir}/${STEP1_DIR}"
STEP2_PTH_PATH="${out_pths_dir}/${STEP2_DIR}"