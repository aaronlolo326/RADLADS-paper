# huggingface-cli download Qwen/Qwen2.5-7B-Instruct
# python3 convert_hf_to_pth.py YOUR_CACHED_HF_QWEN_MODEL_LOCATION out/Qwen2.5-7B-Instruct.pth

export USERNAME=$(whoami)
hf_cache_dir="/work/hei/.cache/huggingface"
pths_dir=/work/${USERNAME}/radlads/pths

download_and_convert_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(echo ${MODEL_PATH} | cut -d'/' -f2)
    local MODEL_CACHE_NAME=$(echo ${MODEL_PATH} | sed 's/\//--/g')

    # Download the model
    huggingface-cli download ${MODEL_PATH}

    # Find the snapshot directory (most recent one)
    local YOUR_CACHED_HF_QWEN_MODEL_LOCATION=$(find ${hf_cache_dir}/hub/models--${MODEL_CACHE_NAME}/snapshots -mindepth 1 -maxdepth 1 -type d | sort -r | head -n 1)

    if [ -z "${YOUR_CACHED_HF_QWEN_MODEL_LOCATION}" ]; then
        echo "Error: Could not find cached model location for ${MODEL_PATH}"
        return 1
    fi

    # Convert to pth format
    mkdir -p ${pths_dir}/${MODEL_NAME}
    python3 convert_hf_to_pth.py ${YOUR_CACHED_HF_QWEN_MODEL_LOCATION} ${pths_dir}/${MODEL_NAME}/pretrained.pth
}

# MODEL_PATH="Qwen/Qwen3-8B-Base"
# download_and_convert_model ${MODEL_PATH}

# MODEL_PATH="Qwen/Qwen3-1.7B-Base"
# download_and_convert_model ${MODEL_PATH}

MODEL_PATH="Qwen/Qwen3-8B"
download_and_convert_model ${MODEL_PATH}