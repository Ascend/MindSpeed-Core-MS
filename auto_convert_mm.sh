#!/bin/bash

#MindSpeed-MM
rm -rf MindSpeed-MM/
git clone https://gitee.com/ascend/MindSpeed-MM.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-MM"
    exit 1
fi
rm -rf MindSpeed-MM/tests
cd MindSpeed-MM/
pip install -e .
cd ..
echo "------------------------------------done MindSpeed-MM"

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
rm -rf MSAdapter
git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MSAdapter"
    exit 1
fi
cd MSAdapter
rm -rf tests
cd ..
echo "..............................................done MSAdapter"

#transformers
rm -rf transformers/
git clone https://gitee.com/mirrors/huggingface_transformers.git
if [ $? -ne 0 ]; then
    echo "Error: git clone transformers"
    exit 1
fi
mv huggingface_transformers transformers
cd transformers
git checkout fa56dcc2a
git apply ../tools/rules/transformers_v4.49.0.diff
rm -rf tests
cd ..
echo "..............................................done apply transformers"

#safetensors
rm -rf safetensors_dir
mkdir safetensors_dir
pip install --no-deps safetensors==0.5.1
if [ $? -ne 0 ]; then
    echo "Error: pip install safetensors fail"
else
    ST_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
    cp -r ${ST_PATH}/safetensors ./safetensors_dir
    cd safetensors_dir/safetensors
    git init
    git apply ../../tools/rules/safetensors.diff
    cd ../../
    export PYTHONPATH=$(pwd)/safetensors_dir:$PYTHONPATH
    echo "..............................................done apply safetensors"
fi

MindSpeed_Core_MS_PATH=$(pwd)
echo ${MindSpeed_Core_MS_PATH}
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
echo $PYTHONPATH
