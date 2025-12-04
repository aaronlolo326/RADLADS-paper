export USERNAME=$(whoami)
pths_dir=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf


CUDA_VISIBLE_DEVICES="4,5,6,7" \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c configs/qwen7b.yaml \
    -c configs/20251127/qwerky7.yaml \
    -c configs/20251127/distill1.yaml \
    --train.load_model ${pths_dir}/Qwen2.5-7B-Instruct.pth