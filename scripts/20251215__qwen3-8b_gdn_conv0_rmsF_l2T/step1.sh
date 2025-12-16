source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c ${qwen_yaml} \
    -c ${la_yaml} \
    -c configs/${RUN_NAME}/qwen3-8b-instructteacher.yaml \
    -c configs/${RUN_NAME}/distill2.yaml \
    --train.load_model ${out_pths_dir}/${RUN_NAME}-1/ckpt-final.pth \
    --train.proj_name ${RUN_NAME} #\
    # > logs/20251215.log 2>&1

# wandb: Syncing run qwerky7_qwen2 L28 D3584 ctx512  2025-11-27-19-56-49
# wandb: ⭐️ View project at https://wandb.ai/aaronlolo326-individual/qwen2rwkv
# wandb: 🚀 View run at https://wandb.ai/aaronlolo326-individual/qwen2rwkv/runs/cofq6v8a
