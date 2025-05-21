#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600

source ../../../scripts/set_path.sh
MindSpeed_LLM_PATH=../../../../MindSpeed-LLM

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=8187
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_ds3_sft/output_ds3_sft"
DATA_PATH="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_ds3_sft/finetune_dataset/alpaca"
TOKENIZER_PATH="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_ds3_sft/tokenizer"
CKPT_LOAD_DIR="/home/workspace/mindspore_dataset/msadapter/test_input/net/test_ds3_sft/load"

TP=1
PP=2
EP=4
CP=1
CP_TYPE='ulysses_cp_algo'

NUM_LAYERS=4
SEQ_LEN=4096
MBS=1
GBS=8

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
    --local_worker_num $GPUS_PER_NODE \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log_sft_deepseek \
    --join=True \
    --cluster_time_out=300 \
    --bind_core=True \
"

MLA_ARGS="
    --multi-head-latent-attention \
    --qk-rope-head-dim 64 \
    --qk-nope-head-dim 128 \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --qk-layernorm \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 16 \
    --moe-router-topk 8 \
    --moe-intermediate-size 2048 \
    --moe-router-load-balancing-type noaux_tc \
    --topk-group 4 \
    --routed-scaling-factor 2.5 \
    --seq-aux \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --use-fused-moe-token-permute-and-unpermute \
"

ROPE_ARGS="
    --rope-scaling-beta-fast 32 \
    --rope-scaling-beta-slow 1 \
    --rope-scaling-factor 40 \
    --rope-scaling-mscale 1.0 \
    --rope-scaling-mscale-all-dim  1.0 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.deepseek_spec layer_spec \
    --num-layer-list 2,2 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 8 \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --use-flash-attn \
    --shape-order BNSD \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 7168 \
    --ffn-hidden-size 18432 \
    --num-attention-heads 128 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-5 \
    --train-iters 10 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 1e-2 \
    --lr-warmup-iters 1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
    --vocab-size 129280 \
    --padded-vocab-size 129280 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --distributed-timeout-minutes 120 \
    --no-gradient-accumulation-fusion \
    --no-check-for-nan-in-loss-and-grad \
    --fix-router \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng \
    --load ${CKPT_LOAD_DIR} \
"

FINETUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --prompt-type deepseek3 \
"

msrun $DISTRIBUTED_ARGS ${MindSpeed_LLM_PATH}/posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MLA_ARGS \
    $ROPE_ARGS \
    $MOE_ARGS \
    $FINETUNE_ARGS \
    --distributed-backend nccl \
    | tee tune_deepseek3.txt