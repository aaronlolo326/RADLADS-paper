#!/bin/bash
source "$(dirname "$0")/vars.sh"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base+

DATE_STR=20251219

# python plot.py \
#     --date_str ${DATE_STR} \
#     --runs_name "" \
#     --exc_runs 20251205__qwen3-8b_rwkv7_s3-2048,L28-D3584-qwerky7_qwen2 \
#     --final_only \
#     --step2_only

python plot.py \
    --date_str ${DATE_STR} \
    --runs_name ${RUN_NAME} \
    --exc_runs ""