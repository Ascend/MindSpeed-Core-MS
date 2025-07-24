#!/bin/bash
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/safetensors_dir:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/huggingface_hub_dir:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/msa_thirdparty/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/:${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/MindSpeed/:${MindSpeed_Core_MS_PATH}/transformers/src/:${MindSpeed_Core_MS_PATH}/accelerate/src/:$PYTHONPATH
echo "..............................................done set PYTHONPATH"
echo $PYTHONPATH
