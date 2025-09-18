#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
mkdir ${MindSpeed_Core_MS_PATH}/RL
cp -r ${MindSpeed_Core_MS_PATH}/auto_convert_rl.sh ${MindSpeed_Core_MS_PATH}/RL
cp -r ${MindSpeed_Core_MS_PATH}/tools ${MindSpeed_Core_MS_PATH}/RL
cd RL
sed -i '/#safetensors/,/#huggingface_hub/ s/^/#/' auto_convert_rl.sh
sed -i '/#huggingface_hub/,/echo "..............................................start code_convert"/ s/^/#/' auto_convert_rl.sh
sed -i '/echo "..............................................start code_convert"/ s/^#//' auto_convert_rl.sh
bash auto_convert_rl.sh is_rl_gongka
echo "..............................................done set RL_env"
