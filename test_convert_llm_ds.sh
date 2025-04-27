#!/bin/bash
#MindSpeed-LLM
git clone https://gitee.com/ascend/MindSpeed-LLM.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi
cd MindSpeed-LLM
git checkout 9679b658307480be6315ec82a086cc5a77c1e361 # date:0408
cd ..
echo "------------------------------------done MindSpeed-LLM"

#MindSpeed
git clone https://gitee.com/ascend/MindSpeed.git
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed"
    exit 1
fi
cd MindSpeed
git checkout b73c63aedc91bc84c1466bed6767bda1e13e9ed9 # date:0407
cd ..
echo "...............................................done MindSpeed"

#Megatron-LM
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
git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
echo "..............................................done msadapter"

#transformers
git clone https://gitee.com/mirrors/huggingface_transformers.git -b v4.47.0
if [ $? -ne 0 ]; then
    echo "Error: git clone transformers"
    exit 1
fi
mv huggingface_transformers transformers
echo "..............................................done transformers"

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
