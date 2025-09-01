#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
mkdir ${MindSpeed_Core_MS_PATH}/MM
cp -r ${MindSpeed_Core_MS_PATH}/auto_convert_mm.sh ${MindSpeed_Core_MS_PATH}/MM
cp -r ${MindSpeed_Core_MS_PATH}/test_convert_mm.sh ${MindSpeed_Core_MS_PATH}/MM
cd ${MindSpeed_Core_MS_PATH}/MM
sed -i '/MindSpeed_Core_MS_PATH=$(pwd)/i rm -rf MindSpeed-MM\/tests\nrm -rf MindSpeed\/tests_extend\nrm -rf Megatron-LM\/tests\nrm -rf MSAdapter\/tests' auto_convert_mm.sh
sed -i '/MindSpeed_Core_MS_PATH=$(pwd)/i rm -rf MindSpeed-MM\/tests\nrm -rf MindSpeed\/tests_extend\nrm -rf Megatron-LM\/tests\nrm -rf msadapter\/tests' test_convert_mm.sh
bash test_convert_mm.sh
echo "..............................................done set MM_env"
