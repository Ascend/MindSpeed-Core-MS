#!/bin/bash

#MindSpeed-LLM
rm -rf MindSpeed-LLM/
git clone https://gitee.com/ascend/MindSpeed-LLM.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed-LLM"
    exit 1
fi
cd MindSpeed-LLM
git checkout 71c5af4d72078d826fd93fec6980004f0de51132
rm -rf tests
cd ..
echo "------------------------------------done MindSpeed-LLM"

#MindSpeed
rm -rf MindSpeed/
git clone https://gitee.com/ascend/MindSpeed.git -b core_r0.8.0
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed"
    exit 1
fi
cd MindSpeed
git checkout 31aaf3d4ca86234b15f4a5d3af20bd6df06e7d45
rm -rf tests_extend
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
if [[ "$1" == "is_rl_gongka" ]]; then
    echo "...............................................MindSpeed-RL GongKa"
    git checkout 0707949f152599862f0a28cb155681599659dc00
    PYTHON_ARG='--is_rl_gongka'
else
    echo "...............................................MindSpeed-RL"
    git checkout 559db0856891e5f8504a0b21d4b26969a82241df
    PYTHON_ARG='--is_rl'
fi
rm -rf tests
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
rm -rf tests
cd ..
echo "..............................................done Megatron-LM"

#msadapter
rm -rf msadapter
git clone https://gitee.com/mindspore/msadapter.git
cd msadapter
rm -rf tests
cd ..
if [ $? -ne 0 ]; then
    echo "Error: git clone msadapter"
    exit 1
fi
echo "..............................................done msadapter"

#vllm
rm -rf vllm
git clone https://gitee.com/mirrors/vllm.git
cd vllm
git checkout v0.7.3
rm -rf tests
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
git checkout 0713836e95fe993feefe334945b5b273e4add1f1
rm -rf tests
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
    echo "Error: git clone huggingface_transformers"
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
git clone https://github.com/huggingface/accelerate.git -b v1.6.0
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
rm -rf huggingface_hub
git clone https://github.com/huggingface/huggingface_hub.git -b v0.29.2
if [ $? -ne 0 ]; then
    echo "Error: git clone huggingface_hub"
    exit 1
fi
cd huggingface_hub
git apply ../tools/rules_rl/huggingface_hub.diff
rm -rf tests
cd ..
echo "..............................................done apply huggingface_hub"

echo "..............................................start code_convert"
MindSpeed_Core_MS_PATH=$PWD
echo ${MindSpeed_Core_MS_PATH}

python3 tools/transfer.py $PYTHON_ARG \
--megatron_path ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/ \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/ \
--mindspeed_rl_path ${MindSpeed_Core_MS_PATH}/MindSpeed-RL/ \
--vllm_path ${MindSpeed_Core_MS_PATH}/vllm/ \
--vllm_ascend_path ${MindSpeed_Core_MS_PATH}/vllm-ascend/

export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:${MindSpeed_Core_MS_PATH}/vllm/:${MindSpeed_Core_MS_PATH}/vllm-ascend/:${MindSpeed_Core_MS_PATH}/accelerate/src/:${MindSpeed_Core_MS_PATH}/safetensors_dir/:${MindSpeed_Core_MS_PATH}/huggingface_hub/src/:${MindSpeed_Core_MS_PATH}/MindSpeed-RL/:$PYTHONPATH
echo $PYTHONPATH
echo "..............................................done code_convert"

pip uninstall -y bitsandbytes-npu-beta
