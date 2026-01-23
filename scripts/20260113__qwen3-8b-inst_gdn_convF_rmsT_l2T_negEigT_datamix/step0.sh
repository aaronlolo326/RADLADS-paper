source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c ${qwen_yaml} \
    -c ${la_yaml} \
    -c configs/${RUN_NAME}/distill1.yaml \
    --train.load_model ${pths_dir}/Qwen3-8B/pretrained.pth \
    --train.proj_name ${RUN_NAME}