CUDA_VISIBLE_DEVICES="7"

pth=/work/hei/pths

RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c configs/qwen7b.yaml \
    -c configs/qwerky7.yaml \
    -c configs/qwen7binstructteacher.yaml \
    -c configs/distill2.yaml \
    --train.load_model out/L28-D3584-qwerky7_qwen2-1/rwkv-final.pth