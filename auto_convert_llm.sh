#!/bin/bash

echo "===========================================start convert==========================================="
rm -rf MindSpeed-LLM/
git clone -b 1.0.0 https://gitee.com/ascend/MindSpeed-LLM.git --depth=1
if [ $? -ne 0 ]; then
    echo "Error:git clone MindSpeed-LLM failed"
    exit 1
fi
rm -rf MindSpeed-LLM/tests
echo "------------------------------------done MindSpeed-LLM"

rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git
if [ $? -ne 0 ]; then
    echo "Error:git clone MindSpeed failed"
    exit 1
fi
cd MindSpeed
git checkout 969686ff
rm -rf tests_extend
cd ..
echo "...............................................done apply MindSpeed"

rm -rf Megatron-LM/
git clone -b core_r0.6.0 https://github.com/NVIDIA/Megatron-LM.git --depth=1
if [ $? -ne 0 ]; then
    echo "Error:git clone Megatron-LM failed"
    exit 1
fi
rm -rf Megatron-LM/tests
echo "------------------------------------done Megatron-LM"

#msadapter
rm -rf msadapter
git clone https://gitee.com/mindspore/msadapter.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
rm -rf msadapter/tests
echo "..............................................done msadapter"

rm -rf transformers/
git clone -b v4.47.0 https://github.com/huggingface/transformers.git --depth=1
if [ $? -ne 0 ]; then
    echo "Error:git clone transformers failed"
    exit 1
fi
rm -rf transformers/tests
echo "------------------------------------done transformers"

rm -rf peft/
git clone -b v0.7.1 https://github.com/huggingface/peft.git --depth=1
if [ $? -ne 0 ]; then
    echo "Error:git clone peft failed"
    exit 1
fi
echo "------------------------------------done peft"

pip install -r requirements.txt
echo "..............................................done install requirements"

python tools/transfer.py --megatron_path Megatron-LM/ --mindspeed_path MindSpeed/ --mindspeed_llm_path MindSpeed-LLM/ \
--transformers_path transformers/ --mindspeed_type LLM

MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/MindSpeed/:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/peft/src/:$PYTHONPATH
echo "..............................................done set PYTHONPATH"

echo "===========================================finish convert==========================================="
