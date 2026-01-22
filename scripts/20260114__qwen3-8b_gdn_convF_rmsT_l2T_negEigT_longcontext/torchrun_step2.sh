source "$(dirname "$0")/vars.sh"
echo "$RUN_NAME"

export MASTER_ADDR=192.168.240.63
export MASTER_PORT=29503
export NNODES=3
export NPROC_PER_NODE=8
export NODE_RANK=0

RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train.py \
    -c ${qwen_yaml} \
    -c ${la_yaml} \
    -c configs/${RUN_NAME}/qwen3-8b-instructteacher.yaml \
    -c configs/${RUN_NAME}/distill3.yaml \
    --train.load_model "/work/yanan/radlads/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT-2/ckpt-final.pth" \
    --train.proj_name ${RUN_NAME} \
    --train.num_nodes 3 \
    --train.devices $NPROC_PER_NODE
