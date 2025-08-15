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
modifygrammar() {
    fname=$1
    echo "Modifying PY310 grammar to adapt PY39..."
    # replace num_query_groups: int | None = None
    sed -i 's/    num_query_groups: int \| None = None/    num_query_groups = None/' "$fname"
    # replace ffn_hidden_size: float | None = None
    sed -i 's/    ffn_hidden_size: float \| None = None/    ffn_hidden_size = None/' "$fname"
    echo "PY310 grammar have been updated to adapt PY39 in $fname"
}
modifygrammar ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/core/transformer/heterogeneous/heterogeneous_config.py
echo "..............................................done set LLM_env"
