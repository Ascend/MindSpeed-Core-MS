#!/bin/bash
source ../../../scripts/set_path_rl.sh
MindSpeed_RL_PATH=../../../../RL/MindSpeed-RL

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

addSeedAll() {
    fname=$1
    # check file exist
    if [ ! -f "$fname" ]; then
        echo "错误：文件 '$fname' 不存在"
        return 1
    fi

    lineNumMain=$(grep -n 'no_shuffle: false' ${fname} | cut -d: -f1)
    sed -i 's/no_shuffle:[[:space:]]*false/no_shuffle: true/' "$fname"
    sed -i $((lineNumMain + 1))'i\ \ use_deter_comp: true' $fname
    sed -i 's/guarantee_order:[[:space:]]*false/guarantee_order: true/' "$fname"
    echo "------- $fname addSeedAll!!"
}

modify_layer() {
    fname=$1
    lineNum1=$(grep -n -F 'fc2_name = megatron_name.replace("linear_fc1' "${fname}" | cut -d: -f1)
    sed -i $((lineNum1 + 1))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ try:' $fname
    sed -i $((lineNum1 + 2))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ megatron_param_fc1 = dict(true_megatron_model.named_parameters())[megatron_name]' $fname
    sed -i $((lineNum1 + 3))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ except KeyError:' $fname
    sed -i $((lineNum1 + 4))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ print(f"[WARNING] megatron_name: {megatron_name} is not Found. Skip...")' $fname
    sed -i $((lineNum1 + 5))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ continue' $fname
    sed -i $((lineNum1 + 6))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ try:' $fname
    sed -i $((lineNum1 + 7))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ megatron_param_fc2 = dict(true_megatron_model.named_parameters())[fc2_name]' $fname
    sed -i $((lineNum1 + 8))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ except KeyError:' $fname
    sed -i $((lineNum1 + 9))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ print(f"[WARNING] fc2_name: {fc2_name} is not Found. Skip...")' $fname
    sed -i $((lineNum1 + 10))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ continue' $fname
    sed -i $((lineNum1 + 11))'d' "$fname"
    sed -i $((lineNum1 + 11))'d' "$fname"
    lineNum2=$(grep -n -F 'megatron_param = megatron_params_dict[megatron_name]' "${fname}" | cut -d: -f1)
    sed -i $((lineNum2 + 1))'i\ \ \ \ \ \ \ \ \ \ \ \ try:' $fname
    sed -i $((lineNum2 + 2))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ megatron_param = megatron_params_dict[megatron_name]' $fname
    sed -i $((lineNum2 + 3))'i\ \ \ \ \ \ \ \ \ \ \ \ except KeyError:' $fname
    sed -i $((lineNum2 + 4))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ print(f"[WARNING] megatron_name: {megatron_name} is not Found. Skip...")' $fname
    sed -i $((lineNum2 + 5))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ continue' $fname
    sed -i "${lineNum2}d" "$fname"
    echo '======'$fname 'modify_layer!'
}

modify_grpo_trainer() {
    fname=$1
    lineNum=$(grep -n -F 'self.compute_advantage(blocking=False, guarantee_order=self.guarantee_order)' "${fname}" | cut -d: -f1)
    sed -i $((lineNum + 1))'i\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ self.compute_advantage(blocking=True, guarantee_order=self.guarantee_order)' $fname
    sed -i "${lineNum}d" "$fname"
    echo '======'$fname 'modify_grpo_trainer!'
}

filepath1=./configs/grpo_qwen25_7b_A3.yaml
filepath2=${MindSpeed_RL_PATH}/mindspeed_rl/workers/resharding/vllm_weight_container.py
filepath3=${MindSpeed_RL_PATH}/mindspeed_rl/trainer/grpo_trainer_hybrid.py
addSeedAll ${filepath1}
backup ${filepath2}
backup ${filepath3}
modify_layer ${filepath2}
modify_grpo_trainer  ${filepath3}



rm -rf ${MindSpeed_RL_PATH}/configs/grpo_qwen25_7b_A3.yaml
rm -rf ${MindSpeed_RL_PATH}/configs/model/qwen25_7b.yaml
cp -r $filepath1  ${MindSpeed_RL_PATH}/configs/
cp -r ./configs/model/qwen25_7b.yaml ${MindSpeed_RL_PATH}/configs/model/
ray stop
python3 ${MindSpeed_RL_PATH}/cli/train_grpo.py --config-name grpo_qwen25_7b_A3 2>&1 | tee ms_det.txt || true
cat ms_det.txt
recover ${filepath2}
recover ${filepath3}
ray stop