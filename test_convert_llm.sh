#!/bin/bash

#MindSpeed-LLM
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi
rm -rf MindSpeed-LLM/tests
echo "------------------------------------done MindSpeed-LLM"

#MindSpeed
rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed"
    exit 1
fi
rm -rf MindSpeed/tests_extend
echo "...............................................done MindSpeed"

#Megatron-LM
rm -rf Megatron-LM/
git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1
if [ $? -ne 0 ]; then
    echo "Error: git clone Megatron-LM"
    exit 1
fi
rm -rf Megatron-LM/tests
echo "..............................................done Megatron-LM"

#msadapter
rm -rf msadapter
git clone https://gitee.com/mindspore/msadapter.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
cd msadapter
rm -rf tests
cd ..
echo "..............................................done msadapter"

MindSpeed_Core_MS_PATH=$(pwd)
echo ${MindSpeed_Core_MS_PATH}

export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter:${MindSpeed_Core_MS_PATH}/msadapter/msa_thirdparty:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:$PYTHONPATH
echo $PYTHONPATH

pip uninstall -y bitsandbytes-npu-beta