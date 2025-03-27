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
cd ..
echo "..............................................done Megatron-LM"

#msadaptor
rm -rf msadapter
git clone https://gitee.com/mindspore/msadapter.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
echo "..............................................done msadapter"

#transformers
rm -rf transformers/
git clone https://gitee.com/mirrors/huggingface_transformers.git -b v4.47.0
if [ $? -ne 0 ]; then
    echo "Error: git clone msadaptor"
    exit 1
fi
mv huggingface_transformers transformers
cd transformers
git apply ../tools/rules/transformers.diff
cd ..
echo "..............................................done apply transformers"

echo "..............................................start code_convert"
MindSpeed_Core_MS_PATH=$PWD
echo ${MindSpeed_Core_MS_PATH}

python3 tools/transfer.py \
--megatron_path ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/ \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/


echo "..............................................done code_convert"

