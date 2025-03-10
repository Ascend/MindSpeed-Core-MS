#!/bin/bash

echo "===========================================start apply==========================================="
rm -rf MindSpeed-MM/
git clone https://gitee.com/ascend/MindSpeed-MM.git
cd MindSpeed-MM/
git checkout 9526e82399d9db8a18cf3f2aa7089f0b928e57b0
git apply ../msadaptor/mindspeed_mm.diff
cd ..
echo "..............................................done apply mindspeed_mm"

rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout ab39de78be23e88e2c8b0d25edf6135940990c02
git apply ../msadaptor/acclerate_mindspeed_mm.diff
cd ..
echo "...............................................done apply acclerate_mindspeed_mm"

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
git apply ../../../msadaptor/transformers.diff
cd ../../..
echo "..............................................done apply transformers"

pip install -r requirements.txt
echo "..............................................done install requirements"

echo "=======================================finish apply======================================="

