#!/bin/bash
THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH=$PYTHONPATH:$THIS_SCRIPT_DIR
export RANK_TO_DEBUG=0                         # TO SET !!!!!!!!!
export HCCL_EXEC_TIMEOUT=76800

# to avoid OpenMP thread and multi-process resource conflict.
if [ -z "${OMP_NUM_THREADS}" ] && [ "${NPUS_PER_NODE}" -gt 1 ]; then
  export OMP_NUM_THREADS=1
fi

export WORLD_SIZE=$(( NPUS_PER_NODE * NNODES ))       # global process num
export LOCAL_WORLD_SIZE=$NPUS_PER_NODE                # local process num
export NODE_RANK=$NODE_RANK

# MASTER env, to set due to torch multi processing logic
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

export TORCHELASTIC_RESTART_COUNT="${TORCHELASTIC_RESTART_COUNT:-0}"   # already restart count
export TORCHELASTIC_MAX_RESTARTS="${TORCHELASTIC_MAX_RESTARTS:-0}"     # max restart count
export TORCHELASTIC_RUN_ID="${TORCHELASTIC_RUN_ID:-$(uuidgen)}"        # unique ID

# since localhost can't be identified, replace by 127.0.0.1
if [ "$MASTER_ADDR" = "localhost" ]; then
    export MASTER_ADDR="127.0.0.1"
fi

# original command, replace `torch.distributed.launch` with `python`
run_pretrain() {                                    # TO SET !!!!!!!!! e.g. posttrain_gpt.py
    python pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $MLA_ARGS \
        $ROPE_ARGS \
        $MOE_ARGS \
        --distributed-backend nccl
}


# run worker
START_RANK=$(( NODE_RANK * NPUS_PER_NODE ))
END_RANK=$(( START_RANK + NPUS_PER_NODE ))

for ((worker_rank=START_RANK; worker_rank<END_RANK; worker_rank++)); do
    if [[ $worker_rank -eq $RANK_TO_DEBUG ]]; then
        continue
    fi
    echo "running rank ${worker_rank} in background."
    export RANK=$worker_rank
    export LOCAL_RANK=$(( worker_rank % NPUS_PER_NODE ))
    export GROUP_RANK=$NODE_RANK                                        # 组内 rank（单组时等于节点号）
    export ROLE_RANK=$RANK                                              # 单角色训练下等同全局 rank
    export ROLE_WORLD_SIZE=$WORLD_SIZE
    run_pretrain \
     > worker_${worker_rank}.log 2>&1 &
done

if [[ $RANK_TO_DEBUG -ge $START_RANK ]] && [[ $RANK_TO_DEBUG -lt $END_RANK ]]; then
    export RANK=$RANK_TO_DEBUG
    export LOCAL_RANK=$(( RANK_TO_DEBUG % NPUS_PER_NODE ))
    export GROUP_RANK=$NODE_RANK
    export ROLE_RANK=$RANK_TO_DEBUG
    echo "DEBUGGING worker in current process, global rank=${RANK_TO_DEBUG}"
    run_pretrain
fi