#!/bin/bash

#MindSpeed-LLM
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi
rm -rf MindSpeed-LLM/tests
echo "------------------------------------done MindSpeed-LLM"

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
git clone https://gitee.com/mirrors/huggingface_transformers.git -b v4.47.0
if [ $? -ne 0 ]; then
    echo "Error: git clone msadaptor"
    exit 1
fi
mv huggingface_transformers transformers
cd transformers
git apply ../tools/rules/transformers.diff
rm -rf tests
cd ..
echo "..............................................done apply transformers"

#accelerate
rm -rf accelerate/
git clone https://gitee.com/modelee/accelerate.git -b v1.6.0
if [ $? -ne 0 ]; then
    echo "Error: git clone accelerate"
    exit 1
fi
cd accelerate
git apply ../tools/rules/accelerate.diff
rm -rf tests
cd ..
echo "..............................................done apply accelerate"

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

#huggingface_hub
rm -rf huggingface_hub_dir
mkdir huggingface_hub_dir
pip install --no-deps huggingface_hub==0.32.3
if [ $? -ne 0 ]; then
    echo "Error: pip install huggingface_hub fail"
else
    ST_PATH=$(python -c "import site; print(site.getsitepackages()[0])")
    cp -r ${ST_PATH}/huggingface_hub ./huggingface_hub_dir
    cd huggingface_hub_dir/huggingface_hub
    git init
    git apply ../../tools/rules/huggingface_hub.diff
    cd ../../
    export PYTHONPATH=$(pwd)/huggingface_hub_dir:$PYTHONPATH
    echo "..............................................done apply huggingface_hub"
fi

echo "..............................................start code_convert"
MindSpeed_Core_MS_PATH=$(pwd)
echo ${MindSpeed_Core_MS_PATH}

python3 tools/transfer.py \
--megatron_path ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/ \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/ \

export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MSAdapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:${MindSpeed_Core_MS_PATH}/accelerate/src/:$PYTHONPATH
echo $PYTHONPATH
echo "..............................................done code_convert"

