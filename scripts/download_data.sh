data_dir=/work/hei/data
mkdir -p ${data_dir}
wget --continue -O ${data_dir}/dataset_mix_1.7B_20Btokens.idx https://huggingface.co/datasets/xfxcwynlc/dataset_mix_1.7B_20Btokens/resolve/main/dataset_mix_1.7B_20Btokens.idx?download=true
wget --continue -O ${data_dir}/dataset_mix_1.7B_20Btokens.bin https://huggingface.co/datasets/xfxcwynlc/dataset_mix_1.7B_20Btokens/resolve/main/dataset_mix_1.7B_20Btokens.bin?download=true
