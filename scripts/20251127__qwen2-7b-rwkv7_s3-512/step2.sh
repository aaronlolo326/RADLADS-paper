export USERNAME=$(whoami)
pth=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c configs/qwen7b.yaml \
    -c configs/20251127/qwerky7.yaml \
    -c configs/20251127/qwen7binstructteacher.yaml \
    -c configs/20251127/distill3.yaml \
    --train.load_model ${out_pths_dir}/L28-D3584-qwerky7_qwen2-2/rwkv-final.pth