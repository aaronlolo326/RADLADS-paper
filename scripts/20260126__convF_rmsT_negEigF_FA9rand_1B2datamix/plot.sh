#!/bin/bash
source "$(dirname "$0")/vars.sh"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base+

DATE_STR="20260104"

python plot.py \
    --date_str ${DATE_STR} \
    --runs_name "" \
    --exc_runs 20251205__qwen3-8b_rwkv7_s3-2048,L28-D3584-qwerky7_qwen2,20251212__qwen3-8b_gdn,qwen2-7b_rwkv7_s3-512,20251215__qwen3-8b_gdn_conv0_rmsF_l2T,20251215__qwen3-8b_gdn_convF_rmsF_l2T,20251215__qwen3-8b_gdn_convF_rmsT_l2T \
    --final_only \
    --csv \
    --step2_only

# python plot.py \
#     --date_str ${DATE_STR} \
#     --runs_name ${RUN_NAME} \
#     --exc_runs ""