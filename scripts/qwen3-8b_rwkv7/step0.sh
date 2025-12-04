export USERNAME=$(whoami)
pths_dir=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

RUN_NAME="qwen3-8b_rwkv7"

CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python train.py \
    -c configs/qwen3-8b.yaml \
    -c configs/qwen3-8b_rwkv7/qwerky7.yaml \
    -c configs/qwen3-8b_rwkv7/distill1.yaml \
    --train.load_model ${pths_dir}/Qwen3-8B-Base.pth \
    --run_name ${RUN_NAME}

# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
# RWKV_TORCH_COMPILE=0 \
# RWKV_JIT_ON=0 \
# python -m pdb train.py \
#     -c configs/qwen3-8b.yaml \
#     -c configs/qwen3-8b_rwkv7/qwerky7.yaml \
#     -c configs/qwen3-8b_rwkv7/distill1.yaml \
#     --train.load_model /work/hei/radlads/pths/Qwen3-8B-Base.pth