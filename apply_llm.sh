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
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
git checkout core_r0.6.0
git apply ../msadaptor/megatron.diff
cd ..
echo "..............................................done apply megatron"

rm -rf transformers/
pip install --no-deps transformers==4.47.0
mkdir src
TF_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r ${TF_PATH}/transformers ./src/
pip uninstall -y transformers
cd src/transformers
git apply ../../msadaptor/transformers.diff
cd ../..
mkdir transformers
mv src/ transformers/
echo "..............................................done apply transformers"

rm -rf peft/
pip install --no-deps peft==0.7.1
mkdir src
cp -r ${TF_PATH}/peft ./src/
pip uninstall -y peft
cd src/peft
git apply ../../msadaptor/peft_lora.diff
cd ../..
mkdir peft
mv src/ peft/
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