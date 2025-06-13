#!/bin/bash
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:$THIS_SCRIPT_DIR
export RANK_TO_DEBUG=0                         # TO SET !!!!!!!!!
export MS_WORKER_NUM=$WORLD_SIZE
export CLUSTER_TIME_OUT=76800

export MS_SCHED_HOST=$MASTER_ADDR              # TO SET IN MULTI NODES
export MS_SCHED_PORT=$MASTER_PORT

# since localhost can't be identified, replace by 127.0.0.1
if [ "$MS_SCHED_HOST" = "localhost" ]; then
    export MS_SCHED_HOST="127.0.0.1"
fi

# original command, replace `msrun` with `python`
run_pretrain() {                                    # TO SET !!!!!!!!! e.g. posttrain_gpt.py
    python pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $MLA_ARGS \
        $ROPE_ARGS \
        $MOE_ARGS \
        --distributed-backend nccl \
        --ai-framework mindspore
}

# run scheduler
export MS_ROLE=MS_SCHED

run_pretrain > scheduler.log 2>&1 &

# run worker
export MS_ROLE=MS_WORKER
START_RANK=$(( NODE_RANK * NPUS_PER_NODE ))
END_RANK=$(( START_RANK + NPUS_PER_NODE ))

for ((worker_rank=START_RANK; worker_rank<END_RANK; worker_rank++)); do
    if [[ $worker_rank -eq $RANK_TO_DEBUG ]]; then
        continue
    fi
    export MS_NODE_ID=$worker_rank
    echo "running rank ${MS_NODE_ID} in background."
    run_pretrain \
     > worker_${worker_rank}.log 2>&1 &
done

if [[ $RANK_TO_DEBUG -ge $START_RANK ]] && [[ $RANK_TO_DEBUG -lt $END_RANK ]]; then
    export MS_NODE_ID=$RANK_TO_DEBUG
    echo "DEBUGGING worker in current process, global rank=${RANK_TO_DEBUG}"
    run_pretrain
fi