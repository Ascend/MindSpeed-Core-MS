#!/bin/bash

source "tests_extend/system_tests/env_npu.sh"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_DIR=./ckpt_llama
DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"
TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --no-overlap-p2p-communication \
    --no-delay-grad-reduce \
    --delay-param-gather \
    --no-scatter-gather-tensors-in-pipeline \
    --sequence-parallel \
    --num-layers 8 \
    --hidden-size 5120 \
    --ffn-hidden-size 13664 \
    --num-attention-heads 40 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --no-load-optim \
    --no-load-rng \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 2 \
    --global-batch-size 2 \
    --make-vocab-size-divisible-by 1 \
    --lr 0.375e-5 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 0.375e-6 \
    --weight-decay 0.1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --norm-epsilon 1e-5 \
    --rotary-percent 1 \
    --hysteresis 2 \
    --initial-loss-scale 65536 \
    --min-loss-scale 1
    --loss-scale-window 1000 \
    --use-flash-attn \
    --no-gradient-accumulation-fusion \
    --lr-decay-iters 320000 \
    --init-method-std 0.006 \
    --pre-tockens 65536 \
    --next-tockens 0 \
    --shape-order SBH \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_DIR \
    --load $CKPT_DIR \
    --finetune \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args

set +x
