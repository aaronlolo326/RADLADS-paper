source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

CUDA_VISIBLE_DEVICES="4,5,6,7" \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c ${qwen_yaml} \
    -c ${la_yaml} \
    -c configs/${RUN_NAME}/qwen3-8b-instructteacher.yaml \
    -c configs/${RUN_NAME}/distill3.yaml \
    --train.load_model ${out_pths_dir}/${RUN_NAME}-2/ckpt-final.pth \
    --train.proj_name ${RUN_NAME}