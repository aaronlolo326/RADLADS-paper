export USERNAME=$(whoami)
pth=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

CUDA_VISIBLE_DEVICES="0" \
python generate.py \
    -c configs/qwen7b.yaml \
    -c configs/20251127_qwerky7.yaml \
    --path  ${out_hf_dir}/L28-D3584-qwerky7_qwen2-1/rwkv-final.pth
