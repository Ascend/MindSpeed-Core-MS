#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
source "tests_extend/system_tests/env_npu.sh"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./ckpt_gpt
VOCAB_FILE=/home/dataset/enwiki/gpt2-vocab.json
MERGE_FILE=/home/dataset/enwiki/gpt2-merges.txt
DATA_PATH=/home/dataset/enwiki/my-t5_text_sentence

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 32 \
    --hidden-size 2560 \
    --num-attention-heads 32 \
    --micro-batch-size 2 \
    --global-batch-size 6 \
    --seq-length 8192 \
    --ffn-hidden-size 13664 \
    --max-position-embeddings 8192 \
    --train-iters 5000 \
    --lr-decay-iters 320000 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --lr 0.375e-5 \
    --lr-decay-style cosine \
    --min-lr 0.375e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --no-gradient-accumulation-fusion \
    --use-flash-attn \
    --pre-tockens 65536 \
    --next-tockens 0 \
    --shape-order SBH \
    --sequence-parallel \
    --no-bias-gelu-fusion \
    --jit-compile \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
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
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
