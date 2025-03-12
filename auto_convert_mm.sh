#!/bin/bash

echo "===========================================start convert==========================================="
rm -rf MindSpeed-MM/
git clone https://gitee.com/ascend/MindSpeed-MM.git
if [ $? -ne 0 ]; then
    echo "Error:git clone MindSpeed-MM failed"
    exit 1
fi
cd MindSpeed-MM/
git checkout 9526e823
cd ..
echo "------------------------------------done MindSpeed-MM"

rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git
if [ $? -ne 0 ]; then
    echo "Error:git clone MindSpeed failed"
    exit 1
fi
cd MindSpeed
git checkout ab39de78
cd ..
echo "...............................................done MindSpeed"

rm -rf Megatron-LM/
git clone -b core_r0.6.0 https://github.com/NVIDIA/Megatron-LM.git --depth=1
if [ $? -ne 0 ]; then
    echo "Error:git clone Megatron-LM failed"
    exit 1
fi
echo "------------------------------------done Megatron-LM"

rm -rf transformers/
git clone -b v4.47.0 https://github.com/huggingface/transformers.git --depth=1
if [ $? -ne 0 ]; then
    echo "Error:git clone transformers failed"
    exit 1
fi
echo "------------------------------------done transformers"

pip install -r requirements.txt
echo "..............................................done install requirements"

python tools/transfer.py --megatron_path Megatron-LM/ --mindspeed_path MindSpeed/ --mindspeed_mm_path MindSpeed-MM/ \
--transformers_path transformers/  --peft_path peft/ --mindspeed_type MM

MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadaptor:${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/MindSpeed/:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
echo "..............................................done set PYTHONPATH"

echo "===========================================finish convert==========================================="

