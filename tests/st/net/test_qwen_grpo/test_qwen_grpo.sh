#!/bin/bash
source ../../../scripts/set_path_rl.sh
MindSpeed_RL_PATH=../../../../RL/MindSpeed-RL

pip install yq
yq --version

sed -i '5s/^/#/' ${MindSpeed_RL_PATH}/mindspeed_rl/trainer/base.py

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

modifyConfig() {
  filepath1="$1"
  sed -i \
      -e '33s/no_shuffle: false/no_shuffle: true/' \
      -e '57s/guarantee_order: false/guarantee_order: true' \
      -e '34i use_deter_comp: true' \
      -e '35i seed: 1234' \
      $filepath1
}

filepath1=./configs/grpo_qwen25_7b_A3.yaml

modifyConfig ${filepath1}


rm -rf ${MindSpeed_RL_PATH}/configs/grpo_qwen25_7b_A3.yaml
rm -rf ${MindSpeed_RL_PATH}/configs/model/qwen25_7b.yaml
cp -r $filepath1  ${MindSpeed_RL_PATH}/configs/
cp -r ./configs/model/qwen25_7b.yaml ${MindSpeed_RL_PATH}/configs/model/
ray stop
python3 ${MindSpeed_RL_PATH}/cli/train_grpo.py --config-name grpo_qwen25_7b_A3 2>&1 | tee ms_det.txt || true
cat ms_det.txt
ray stop