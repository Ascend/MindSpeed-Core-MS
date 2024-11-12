#!/bin/bash
set -e

export HCCL_BUFFERSIZE=200
export HCCL_EXEC_TIMEOUT=300

PORT=8848
WORKER_NUM=1
LOG_PATH="rotary_embedding_log"

MODEL_ARGS="
    --num-layers 8 \
    --seq-length 64 \
    --vocab-size 1024 \
    --hidden-size 512 \
    --ffn-hidden-size 1 \
    --num-attention-heads 32 \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 128 \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --no-load-optim \
    --no-load-rng \
    --no-gradient-accumulation-fusion \
    --bf16\
    --rotary-interleaved \
    --no-rope-fusion \
    --use-mcore-models \
"

DATA_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 1 \
"

PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
"

msrun --worker_num "$WORKER_NUM" \
      --local_worker_num="$WORKER_NUM" \
      --master_port=$PORT \
      --log_dir="$LOG_PATH" \
      --join=True \
      --cluster_time_out=300 \
      run_rotary_embedding.py \
      $MODEL_ARGS \
      $DATA_ARGS \
      $PARALLEL_ARGS
