#!/bin/bash
backup() {
    fname=$1
    cp $fname $fname'_back'
    echo '======'$fname 'backuped!'
}

recover() {
    fname=$1
    cp $fname'_back' $fname
    echo '======'$fname 'recovered!!!!'
}

memRecord() {
    recordFile=$1
    bash mem.sh $recordFile > mem.txt 2>&1&
}


# 关闭确定性计算跑一遍
export HCCL_DETERMINISTIC=false # HCCL确定性
export ASCEND_LAUNCH_BLOCKING=  # 硬件确定性
export NCCL_DETERMINISTIC=
bash test_ds3_sft.sh > ms_non_det.txt
cat ms_non_det.txt