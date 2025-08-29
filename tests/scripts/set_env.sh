#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
cd ${MindSpeed_Core_MS_PATH}
script_file="test_convert_llm.sh"

sed -i '/rm -rf MindSpeed-LLM\/tests/i \
cd MindSpeed-LLM\n\
git checkout 7127be594109d796799edb5c1906e68a926fb536\n\
cd ..' "$script_file"

sed -i '/rm -rf MindSpeed\/tests_extend/i \
cd MindSpeed\n\
git checkout 47bf2a6b3bc0c908899e252e7b488367ab9eecfb\n\
cd ..' "$script_file"

bash test_convert_llm.sh
modifygrammar() {
    fname=$1
    echo "Modifying PY310 grammar to adapt PY39..."
    # replace num_query_groups: int | None = None
    sed -i 's/^[[:space:]]*num_query_groups:.*= None/    num_query_groups = None/' "$fname"
    # replace ffn_hidden_size: float | None = None
    sed -i 's/^[[:space:]]*ffn_hidden_size:.*= None/    ffn_hidden_size = None/' "$fname"
    echo "PY310 grammar have been updated to adapt PY39 in $fname"
}
modifygrammar ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/core/transformer/heterogeneous/heterogeneous_config.py
echo "..............................................done set LLM_env"
