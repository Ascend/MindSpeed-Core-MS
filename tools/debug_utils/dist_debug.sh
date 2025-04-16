#!/bin/bash
export MS_DEBUG_RANK=0                         # TO SET
export MS_WORKER_NUM=$WORLD_SIZE
export CLUSTER_TIME_OUT=76800

# since localhost can't be identified, replace by 127.0.0.1
if [ "$MS_SCHED_HOST" = "localhost" ]; then
    export MS_SCHED_HOST="127.0.0.1"
fi

# run scheduler
export MS_SCHED_HOST=$MASTER_ADDR              # TO SET IN MULTI NODES
export MS_SCHED_PORT=$MASTER_PORT
export MS_ROLE=MS_SCHED
python pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --distributed-backend nccl \
    > scheduler.log 2>&1 &

# run worker
export MS_ROLE=MS_WORKER
START_RANK=$(( NODE_RANK * NPUS_PER_NODE ))
END_RANK=$(( START_RANK + NPUS_PER_NODE ))
for ((worker_rank=START_RANK; worker_rank<END_RANK; worker_rank++)); do
    if [ $worker_rank -eq $MS_DEBUG_RANK ]; then
        continue
    fi
    export MS_NODE_ID=$worker_rank
    echo "running rank ${MS_NODE_ID} in background."
    python pretrain_gpt.py \
        ${GPT_ARGS} \
        ${DATA_ARGS} \
        ${OUTPUT_ARGS} \
        --distributed-backend nccl \
        > worker_${worker_rank}.log 2>&1 &
done

if [ $MS_DEBUG_RANK -ge $START_RANK ] && [ $MS_DEBUG_RANK -lt $END_RANK ]; then
    export MS_NODE_ID=$MS_DEBUG_RANK
    echo "DEBUGGING worker in current process, global rank=${MS_DEBUG_RANK}"
    python pretrain_gpt.py \
        ${GPT_ARGS} \
        ${DATA_ARGS} \
        ${OUTPUT_ARGS} \
        --distributed-backend nccl
fi
