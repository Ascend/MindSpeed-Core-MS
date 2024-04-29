#! /bin/bash

cann_dir=$1
pipeline_dir=$2
sub_task=$3
DATESTR=$(date +"%m-%d")
log_dir=$pipeline_dir/logs/$DATESTR
megatron_dir=$pipeline_dir/Megatron-LM

source $cann_dir/ascend-toolkit/set_env.sh
cd $megatron_dir

mkdir -p $log_dir

for file_dir in $megatron_dir/tests_extend/system_tests/${sub_task}/*;
do
    file_name=${file_dir##*/}
    task_name="${file_name%.*}"
    echo "------------------ Task $task_name Start ------------------"
    echo "====================$task_name====================" >> ${log_dir}/result.log
    echo "[Start time] $(date "+%Y/%m/%d %H:%M:%S")" >> ${log_dir}/result.log
    run_cmd="bash tests_extend/system_tests/${sub_task}/${task_name}.sh 2>&1 | tee ${log_dir}/${task_name}.log"
    rm -rf ckpt_llama
    eval ${run_cmd}
    
    # TPS
    step_time=`grep -a "elapsed time per iteration" ${log_dir}/${task_name}.log|awk -F "|" '{print $3}'|awk -F ":" '{print $2}'|tail -n -20|awk '{sum+=$1} END {print"",sum/NR}'`
    global_batch_size=`grep -w "global_batch_size" ${log_dir}/${task_name}.log|awk -F " " '{print $3}'`
    seq_length=`grep -w "seq_length" ${log_dir}/${task_name}.log|awk -F " " '{print $3}'`
    world_size=`grep -w "world_size" ${log_dir}/${task_name}.log|awk -F " " '{print $3}'`
    TPS=$(echo "$global_batch_size * $seq_length * 1000 / $world_size / $step_time"|bc)
    
    # Loss
    last_loss=`grep "elapsed time per iteration" ${log_dir}/${task_name}.log|awk -F "|" '{print $6}'|awk -F ":" '{print $2}'|tail -n -1`
    
    # Gather the results
    echo "[ActualFPS] ${TPS}" >> ${log_dir}/result.log
    echo "[ActualLoss] ${last_loss}" >> ${log_dir}/result.log
    echo -e "[End time] $(date "+%Y/%m/%d %H:%M:%S")\n" >> ${log_dir}/result.log
    echo -e "------------------ Task $task_name Done ------------------\n"
done

set +x
