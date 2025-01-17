#!/bin/bash

echo "===========================================start apply==========================================="
rm -r MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM/
git checkout 1.0.0
git apply ../msadaptor/mindspeed_llm.diff
cd ..
echo "..............................................done apply mindspeed_llm"

rm -r MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 969686ff
git apply ../msadaptor/acclerate_mindspeed_llm.diff
cd ..
echo "...............................................done apply acclerate_mindspeed_llm"

rm -r Megatron-LM/
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
git checkout core_r0.6.0
git apply ../msadaptor/megatron.diff
cd ..
echo "..............................................done apply megatron"

rm -r transformers/
pip install transformers==4.47.0
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

pip install -r requirements.txt
echo "..............................................done install requirements"

echo "=======================================finish apply======================================="
