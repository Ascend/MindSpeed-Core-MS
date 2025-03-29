#!/bin/bash

#MindSpeed-LLM
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi
cd MindSpeed-LLM
git checkout 36bd5742b51c84ea762dc57f8943b0ee5301ee74
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
git checkout 0b832e42
cd ..
echo "...............................................done MindSpeed"

#MindSpeed-RL
rm -rf MindSpeed-RL/
git clone https://gitee.com/ascend/MindSpeed-RL.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-RL"
    exit 1
fi
cd MindSpeed-RL
git checkout 8eddc32877f686798f5cda6b1b34fc72a6beec10
cd ..
echo "...............................................done MindSpeed-RL"

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

#msadapter
rm -rf msadapter
git clone https://gitee.com/mindspore/msadapter.git
cd msadapter
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
cd ..
echo "..............................................done msadapter"

#vllm
rm -rf vllm
git clone https://gitee.com/mirrors/vllm.git
cd vllm
git checkout ed6e9075d31e32c8548b480a47d1ffb77da1f54c
if [ $? -ne 0 ]; then
    echo "Error: git clone vllm"
    exit 1
fi
cd ..
echo "..............................................done vllm"


#vllm-ascend
rm -rf vllm-ascend
git clone https://gitee.com/mirrors/vllm-ascend.git
cd vllm-ascend
git checkout 701a2870469d8849a50378f9450dc3e851c8af20
if [ $? -ne 0 ]; then
    echo "Error: git clone vllm-ascend"
    exit 1
fi
cd ..
echo "..............................................done vllm-ascend"

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
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/ \
--mindspeed_rl_path ${MindSpeed_Core_MS_PATH}/MindSpeed-RL/ \
--vllm_path ${MindSpeed_Core_MS_PATH}/vllm/ \
--vllm_ascend_path ${MindSpeed_Core_MS_PATH}/vllm-ascend/ \
--is_rl

echo "..............................................done code_convert"

