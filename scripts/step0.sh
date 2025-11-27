CUDA_VISIBLE_DEVICES="7"

pth=/work/hei/pths

RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c configs/qwen7b.yaml \
    -c configs/20251127_qwerky7.yaml \
    -c configs/20251126_distill1.yaml \
    --train.load_model ${pth}/Qwen2.5-7B-Instruct.pth