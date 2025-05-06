#!/bin/bash

#MindSpeed-LLM
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi
cd MindSpeed-LLM
git checkout 84a34ee5e10da1ef5beff0787e94605aa961d3ad
git checkout master -- tests/mindspore
cd ..
echo "------------------------------------done MindSpeed-LLM"

#MindSpeed
rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed"
    exit 1
fi
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df
rm -rf tests_extend
cd ..
echo "...............................................done MindSpeed"

#Megatron-LM
rm -rf Megatron-LM/
git clone https://gitee.com/mirrors/Megatron-LM.git
if [ $? -ne 0 ]; then
    echo "Error: git clone Megatron-LM"
    exit 1
fi
cd Megatron-LM
git checkout core_r0.8.0
rm -rf tests
cd ..
echo "..............................................done Megatron-LM"

#msadaptor
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

#transformers
rm -rf transformers/
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 97e57b2545bfc302cc75f7b4f720a7f77992b69e
rm -rf tests
cd ..
echo "..............................................done apply transformers"

echo "..............................................start code_convert"
MindSpeed_Core_MS_PATH=$(pwd)
echo ${MindSpeed_Core_MS_PATH}

python3 tools/transfer.py \
--megatron_path ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/ \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/

export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
echo $PYTHONPATH
echo "..............................................done code_convert"

