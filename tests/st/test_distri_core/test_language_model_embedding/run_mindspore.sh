#!/bin/bash

set -e

export ASCEND_RT_VISIBLE_DEVICES=${1:-"0"}
runmode=${2:-"test_legacy"}
data_dir=${3:-"data/alone/random_data/"}
ckpt_dir=${4:-"data/alone/random_ckpt/"}
output_dir=${5:-"data/alone/output/"}
log_path=${6:-"msrun_log"}

export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=600
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1

port=8811

# kill process
PIDS=$(lsof -i :$port | awk 'NR>1 {print $2}')
if [ -n "$PIDS" ]; then
    for pid in $PIDS; do
        kill -9 $pid
        echo "Killed process $pid"
    done
else
    echo "No processes found listening on port $port."
fi

project_dir=$(dirname "$0")
yaml_file="$project_dir/test_embedding.yaml"

dirs=("${data_dir}"
      "${ckpt_dir}"
      "${output_dir}/legacy_forward"
      "${output_dir}/legacy_backward"
      "${output_dir}/mcore_forward"
      "${output_dir}/mcore_backward")
for dir in "${dirs[@]}"; do
    mkdir -p $dir
done

rm -rf "${log_path}"
mkdir "${log_path}"
echo "train start, log path: ${log_path}"

# 计算设备数量
IFS=',' read -r -a devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
work_num=${#devices[@]}

msrun --worker_num "$work_num" \
      --local_worker_num="$work_num" \
      --master_port=$port --log_dir="$log_path" \
      --join=True \
      --cluster_time_out=300 \
      $project_dir/run_embedding.py \
      --run_mode "$runmode" \
      --ckpt_dir "$ckpt_dir" \
      --data_dir "$data_dir" \
      --output_dir "$output_dir" \
      --yaml-cfg "$yaml_file"
