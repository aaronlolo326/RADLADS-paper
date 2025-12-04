export USERNAME=$(whoami)
model_name="L28-D3584-qwerky7_qwen2"
pth=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

python convert_to_safetensors.py \
    ${out_pths_dir}/${model_name}-1/rwkv-final.pth \
    ${out_hf_dir}/${model_name}-1.safetensors

python convert_to_safetensors.py \
    ${out_pths_dir}/${model_name}-2/rwkv-final.pth \
    ${out_hf_dir}/${model_name}-2.safetensors

python convert_to_safetensors.py \
    ${out_pths_dir}/${model_name}-4/rwkv-final.pth \
    ${out_hf_dir}/${model_name}-4.safetensors
