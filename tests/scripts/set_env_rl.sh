#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
mkdir ${MindSpeed_Core_MS_PATH}/RL
cp -r ${MindSpeed_Core_MS_PATH}/test_convert_rl.sh ${MindSpeed_Core_MS_PATH}/RL
cp -r ${MindSpeed_Core_MS_PATH}/tools ${MindSpeed_Core_MS_PATH}/RL
cd RL
bash test_convert_rl.sh
echo "..............................................done set RL_env"
