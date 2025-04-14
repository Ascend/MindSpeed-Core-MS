#!/bin/bash
source ../../../../scripts/set_path_rl.sh
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
    # 在文件顶部插入
    echo deterministic
    sed -i '1iimport os' $fname
    sed -i '2iimport random' $fname
    sed -i '3iimport numpy as np' $fname
    sed -i '4iimport torch' $fname
    sed -i '5iimport torch_npu' $fname
    sed -i '6idef seed_all(seed=42):' $fname
    sed -i '7i\    random.seed(seed)' $fname
    sed -i '8i\    os.environ["PYTHONHASHSEED"] = str(seed)' $fname
    sed -i '9i\    np.random.seed(seed)' $fname
    sed -i '10i\    torch.manual_seed(seed)' $fname
    sed -i '11i\    torch.use_deterministic_algorithms(True)' $fname
    sed -i '12i\    torch_npu.npu.manual_seed_all(seed)' $fname
    sed -i '13i\    torch_npu.npu.manual_seed(seed)' $fname

    lineNumMain=$(grep -n 'def initialize(self):' ${fname} | cut -d: -f1)
    [ -n "$lineNumMain" ] && sed -i $((lineNumMain + 1))'i\ \ \ \ \ \ \ \ seed_all()' $fname
    lineNumMain=$(grep -n 'def fit(self, data_loader: DataLoader):' ${fname} | cut -d: -f1)
    [ -n "$lineNumMain" ] && sed -i $((lineNumMain + 1))'i\ \ \ \ \ \ \ \ seed_all()' $fname
}


# 开确定性计算跑一遍
filepath1=${MindSpeed_RL_PATH}/mindspeed_rl/trainer/grpo_trainer_hybrid.py
filepath2=${MindSpeed_RL_PATH}/mindspeed_rl/workers/actor_hybrid_worker.py
filepath3=${MindSpeed_RL_PATH}/mindspeed_rl/workers/base_worker.py
filepath4=${MindSpeed_RL_PATH}/mindspeed_rl/workers/reference_woker.py
filepath5=${MindSpeed_RL_PATH}/mindspeed_rl/workers/reward_woker.py
backup ${filepath1}
backup ${filepath2}
backup ${filepath3}
backup ${filepath4}
backup ${filepath5}
addSeedAll ${filepath1}
addSeedAll ${filepath2}
addSeedAll ${filepath3}
addSeedAll ${filepath4}
addSeedAll ${filepath5}

#export HCCL_DETERMINISTIC=true  # HCCL确定性
#export ASCEND_LAUNCH_BLOCKING=1  # 硬件确定性
#export NCCL_DETERMINISTIC=1

cp -r /home/workspace/mindspore_dataset/msadapter/test_input/net/test_ds3_grpo/DeepSeek-V3-hf-8p/ .
cp -r /home/workspace/mindspore_dataset/msadapter/test_input/net/test_ds3_grpo/pe-nlp/ .
rm -rf ${MindSpeed_RL_PATH}/configs/grpo_trainer_deepseekv3_671b.yaml
rm -rf ${MindSpeed_RL_PATH}/configs/model/deepseekv3_671b.yaml
cp -r ./configs/grpo_trainer_deepseekv3_3b.yaml ${MindSpeed_RL_PATH}/configs/
cp -r ./configs/model/deepseekv3-3b.yaml ${MindSpeed_RL_PATH}/configs/model/
ray stop
python3 ${MindSpeed_RL_PATH}/cli/train_grpo.py --config-name grpo_trainer_deepseekv3_3b 2>&1 | tee ms_det.txt || true
cat ms_det.txt
recover ${filepath1}
recover ${filepath2}
recover ${filepath3}
recover ${filepath4}
recover ${filepath5}
ray stop