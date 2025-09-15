#!/bin/bash
# The first argument must be 'llm' or 'mm', the second argument 'msa_latest' is optional.
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {llm|mm} [msa_latest]"
    exit 1
fi

if [ "$1" != "llm" ] && [ "$1" != "mm" ]; then
    echo "First argument must be 'llm' or 'mm'"
    exit 1
fi

if [ "$2" != "" ] && [ "$2" != "msa_latest" ]; then
    echo "Second argument must be '' or 'msa_latest'"
    exit 1
fi

MindSpeed_Core_MS_PATH=$(pwd)

if [ "$1" == "llm" ]; then
    echo "Cloning MindSpeed-LLM..."
    rm -rf MindSpeed-LLM/
    git clone https://gitee.com/ascend/MindSpeed-LLM.git -b master
    if [ $? -ne 0 ]; then
        echo "Error: git clone MindSpeed-LLM"
        exit 1
    fi
    MODEL_ENV=${MindSpeed_Core_MS_PATH}/MindSpeed-LLM
    echo "------------------------------------done MindSpeed-LLM"
elif [ "$1" == "mm" ]; then
    echo "Cloning MindSpeed-MM..."
    rm -rf MindSpeed-MM/
    git clone https://gitee.com/ascend/MindSpeed-MM.git -b master
    if [ $? -ne 0 ]; then
        echo "Error: git clone MindSpeed-MM"
        exit 1
    fi
    cd MindSpeed-MM/
    cp -f examples/mindspore/checkpoint/pyproject.toml ./
    pip install -e .
    cd ..
    MODEL_ENV=${MindSpeed_Core_MS_PATH}/MindSpeed-MM
    echo "------------------------------------done MindSpeed-MM"
fi

if [ "$2" == "msa_latest" ]; then
    echo "Cloning latest msadapter..."
    rm -rf msadapter
    git clone https://gitee.com/mindspore/msadapter.git -b master
    if [ $? -ne 0 ]; then
        echo "Error: git clone msadapter"
        exit 1
    fi
    MSA_ENV=${MindSpeed_Core_MS_PATH}/msadapter:${MindSpeed_Core_MS_PATH}/msadapter/msa_thirdparty
    echo "..............................................done msadapter"
else
    echo "Cloning default MSAdapter..."
    rm -rf MSAdapter
    git clone https://openi.pcl.ac.cn/OpenI/MSAdapter.git -b master
    if [ $? -ne 0 ]; then
        echo "Error: git clone MSAdapter"
        exit 1
    fi
    MSA_ENV=${MindSpeed_Core_MS_PATH}/MSAdapter:${MindSpeed_Core_MS_PATH}/MSAdapter/msa_thirdparty
    echo "..............................................done MSAdapter"
fi

echo "Cloning MindSpeed..."
rm -rf MindSpeed/
git clone https://gitcode.com/Ascend/MindSpeed.git -b master
if [ $? -ne 0 ]; then
    echo "Error: git clone MindSpeed"
    exit 1
fi
echo "...............................................done MindSpeed"

echo "Cloning Megatron-LM..."
rm -rf Megatron-LM/
git clone https://gitee.com/mirrors/Megatron-LM.git -b core_v0.12.1
if [ $? -ne 0 ]; then
    echo "Error: git clone Megatron-LM"
    exit 1
fi
echo "..............................................done Megatron-LM"

echo ${MindSpeed_Core_MS_PATH}
export PYTHONPATH=${MODEL_ENV}:${MSA_ENV}:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/Megatron-LM:$PYTHONPATH
echo $PYTHONPATH

pip uninstall -y bitsandbytes-npu-beta