#!/bin/bash

echo "===========================================start apply==========================================="
git clone https://gitee.com/ascend/MindSpeed-MM.git
cd MindSpeed-MM/
git checkout 9526e82399d9db8a18cf3f2aa7089f0b928e57b0
git apply ../msadaptor/mindspeed_mm.diff
cd ..
echo "..............................................done apply mindspeed_mm"

git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout ab39de78be23e88e2c8b0d25edf6135940990c02
git apply ../msadaptor/acclerate_mindspeed_mm.diff
cd ..
echo "...............................................done apply acclerate_mindspeed_mm"

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
git checkout core_r0.6.0
git apply ../msadaptor/megatron.diff
cd ..
echo "..............................................done apply megatron"

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
echo "=======================================finish apply======================================="

