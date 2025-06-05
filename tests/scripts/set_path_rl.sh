#!/bin/bash
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/RL/Megatron-LM/:${MindSpeed_Core_MS_PATH}/RL/MindSpeed/:${MindSpeed_Core_MS_PATH}/RL/MindSpeed-LLM/:${MindSpeed_Core_MS_PATH}/RL/MindSpeed-RL/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/RL/msadapter/mindtorch/:${MindSpeed_Core_MS_PATH}/RL/transformers/src:${MindSpeed_Core_MS_PATH}/RL/vllm/:${MindSpeed_Core_MS_PATH}/RL/vllm-ascend/:${MindSpeed_Core_MS_PATH}/accelerate/src/:${MindSpeed_Core_MS_PATH}/safetensors_dir/:${MindSpeed_Core_MS_PATH}/huggingface_hub/src/:$PYTHONPATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH
echo "..............................................done set PYTHONPATH"
echo $PYTHONPATH