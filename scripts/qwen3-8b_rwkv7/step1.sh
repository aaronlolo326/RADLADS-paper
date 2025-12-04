export USERNAME=$(whoami)
pth=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

MODEL_PREFIX="L36-D4096-qwerky7_qwen3-8b" # gen from code
RUN_NAME="qwen3-8b_rwkv7"

CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c configs/qwen3-8b.yaml \
    -c configs/${RUN_NAME}/qwerky7.yaml \
    -c configs/${RUN_NAME}/qwen8binstructteacher.yaml \
    -c configs/${RUN_NAME}/distill2.yaml \
    --train.load_model ${out_pths_dir}/${MODEL_PREFIX}-1/rwkv-final.pth

# wandb: Syncing run qwerky7_qwen2 L28 D3584 ctx512  2025-11-27-19-56-49
# wandb: ⭐️ View project at https://wandb.ai/aaronlolo326-individual/qwen2rwkv
# wandb: 🚀 View run at https://wandb.ai/aaronlolo326-individual/qwen2rwkv/runs/cofq6v8a
