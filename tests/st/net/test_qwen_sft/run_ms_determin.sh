#!/bin/bash

MindSpeed_LLM_PATH=../../../../MindSpeed-LLM
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

addSeedAll() {
    fname=$1
    lineNumMain=$(grep -n '__main__' ${fname} | cut -d: -f1)
    echo deterministic
    sed -i $((lineNumMain + 1))'i\ \ \ \ seed_all()' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch_npu.npu.manual_seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch_npu.npu.manual_seed_all(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch.use_deterministic_algorithms(True)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch.manual_seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ np.random.seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ os.environ["PYTHONHASHSEED"] = str(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ random.seed(seed)' $fname
    sed -i $((lineNumMain - 1))'idef seed_all(seed=42):' $fname
    sed -i $((lineNumMain - 1))'iimport torch_npu' $fname
    sed -i $((lineNumMain - 1))'iimport torch' $fname
    sed -i $((lineNumMain - 1))'iimport numpy as np' $fname
    sed -i $((lineNumMain - 1))'iimport random' $fname
    sed -i $((lineNumMain - 1))'iimport os' $fname
}


# 开确定性计算跑一遍
backup ${MindSpeed_LLM_PATH}/posttrain_gpt.py
addSeedAll ${MindSpeed_LLM_PATH}/posttrain_gpt.py
export HCCL_DETERMINISTIC=true  # HCCL确定性
export ASCEND_LAUNCH_BLOCKING=1  # 硬件确定性
export NCCL_DETERMINISTIC=1
bash test_qwen_sft.sh > ms_det.txt
cat ms_det.txt
recover ${MindSpeed_LLM_PATH}/posttrain_gpt.py