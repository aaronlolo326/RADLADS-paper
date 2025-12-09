#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base+

DATE_STR=20251208

# python plot.py \
#     --date_str ${DATE_STR} \
#     --runs_name "" \
#     --exc_runs 20251205__qwen3-8b_rwkv7_s3-2048,L28-D3584-qwerky7_qwen2 \
#     --final_only \
#     --step2_only

# python plot.py \
#     --date_str ${DATE_STR} \
#     --runs_name "20251205__qwen3-8b_rwkv7_s3-2048" \
#     --exc_runs ""

# python plot.py \
#     --date_str ${DATE_STR} \
#     --runs_name "20251206__qwen3-8b_rwkv7qknorm_s3-2048" \
#     --exc_runs ""

python plot.py \
    --date_str ${DATE_STR} \
    --runs_name "20251127__qwen2-7b_rwkv7_s3-512" \
    --exc_runs ""