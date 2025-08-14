#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
cd ${MindSpeed_Core_MS_PATH}
sed -i '/#safetensors/,/#huggingface_hub/ s/^/#/' test_convert_llm.sh
sed -i '/#huggingface_hub/,/echo "..............................................start code_convert"/ s/^/#/' test_convert_llm.sh
sed -i '/echo "..............................................start code_convert"/ s/^#//' test_convert_llm.sh
bash test_convert_llm.sh
echo "..............................................done set LLM_env"
