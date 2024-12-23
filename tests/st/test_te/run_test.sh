#!/bin/bash
set -e
rm -rf msrun_log
mkdir msrun_log
echo "start training"

# msrun --worker_num=2 --local_worker_num=2 --master_port=8197 --log_dir=msrun_log --join=True --cluster_time_out=100 test_fp8_row.py
msrun --worker_num=2 --local_worker_num=2 --master_port=8127 --log_dir=msrun_log --join=True --cluster_time_out=100 test_fp8_column.py