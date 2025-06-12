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

# 打印16位日志
modifyTrainingLogs() {
    fname=$1
    echo "Modifying training log precision..."
    # 替换 log_string += ' {}: {:.6E} |'.format(key, avg)
    sed -i 's/log_string += '\'' {}: {:.6E} |'\''.format(key, avg)/log_string += '\'' {}: {:.16f} |'\''.format(key, avg)/g' "$fname"
    # 替换 log_string += ' grad norm: {:.3f} |'.format(grad_norm)
    sed -i 's/log_string += '\'' grad norm: {:.3f} |'\''.format(grad_norm)/log_string += '\'' grad norm: {:.16f} |'\''.format(grad_norm)/g' "$fname"
    # 替换 log_string += ' params norm: {:.3f} |'.format(params_norm)
    sed -i 's/log_string += '\'' params norm: {:.3f} |'\''.format(params_norm)/log_string += '\'' params norm: {:.16f} |'\''.format(params_norm)/g' "$fname"
    echo "Log precision has been updated to 16 decimal places in $fname"
}

# 开确定性计算跑一遍
backup ${MindSpeed_LLM_PATH}/posttrain_gpt.py
backup ${MindSpeed_LLM_PATH}/mindspeed_llm/training/training.py
addSeedAll ${MindSpeed_LLM_PATH}/posttrain_gpt.py
modifyTrainingLogs ${MindSpeed_LLM_PATH}/mindspeed_llm/training/training.py
export HCCL_DETERMINISTIC=true  # HCCL确定性
export ASCEND_LAUNCH_BLOCKING=1  # 硬件确定性
export NCCL_DETERMINISTIC=1
bash test_qwen_sft.sh > ms_det.txt
cat ms_det.txt
recover ${MindSpeed_LLM_PATH}/posttrain_gpt.py
recover ${MindSpeed_LLM_PATH}/mindspeed_llm/training/training.py