#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
mkdir ${MindSpeed_Core_MS_PATH}/MM
cp -r ${MindSpeed_Core_MS_PATH}/auto_convert.sh ${MindSpeed_Core_MS_PATH}/MM
cd ${MindSpeed_Core_MS_PATH}/MM
sed -i '/echo ${MindSpeed_Core_MS_PATH}/i rm -rf MindSpeed-MM\/tests\nrm -rf MindSpeed\/tests_extend\nrm -rf Megatron-LM\/tests\nrm -rf msadapter\/tests' auto_convert.sh
bash auto_convert.sh mm msa_latest
echo "..............................................done set MM_env"
