#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
MindSpeed_Core_MS_PATH=$(dirname $script_dir)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch/:$PYTHONPATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/:${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/MindSpeed/:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
echo "..............................................done set PYTHONPATH"
echo $PYTHONPATH
