#!/bin/bash

echo "===========================================start apply==========================================="
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM/
git checkout 1.0.0
git apply ../msadaptor/mindspeed_llm.diff
cd ..
echo "..............................................done apply mindspeed_llm"

rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 969686ff
git apply ../msadaptor/acclerate_mindspeed_llm.diff
cd ..
echo "...............................................done apply acclerate_mindspeed_llm"

rm -rf Megatron-LM/
git clone https://gitee.com/mirrors/Megatron-LM.git
cd Megatron-LM/
git checkout core_r0.6.0
git apply ../msadaptor/megatron.diff
cd ..
echo "..............................................done apply megatron"

rm -rf transformers/
git clone https://gitee.com/mirrors/huggingface_transformers.git
mv huggingface_transformers transformers
cd transformers
git checkout v4.47.0
cd src/transformers
git apply ../../msadaptor/transformers.diff
cd ../..
echo "..............................................done apply transformers"

rm -rf peft/
git clone https://gitee.com/mirrors/peft.git
cd peft
git checkout v0.7.1
cd src/peft
git apply ../../msadaptor/peft_lora.diff
cd ../..
echo "..............................................done apply peft"

pip install -r requirements.txt
echo "..............................................done install requirements"

echo "=======================================finish apply======================================="

MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadaptor:${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/MindSpeed/:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/peft/src/:$PYTHONPATH
echo "..............................................done set PYTHONPATH"
echo $PYTHONPATH

echo "=======================================finish env set======================================="