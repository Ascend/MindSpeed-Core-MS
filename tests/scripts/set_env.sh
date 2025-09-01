#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
cd ${MindSpeed_Core_MS_PATH}
sed -i '/MindSpeed_Core_MS_PATH=$(pwd)/i rm -rf MindSpeed-LLM\/tests\nrm -rf MindSpeed\/tests_extend\nrm -rf Megatron-LM\/tests\nrm -rf MSAdapter\/tests' auto_convert_llm.sh
sed -i '/MindSpeed_Core_MS_PATH=$(pwd)/i rm -rf MindSpeed-LLM\/tests\nrm -rf MindSpeed\/tests_extend\nrm -rf Megatron-LM\/tests\nrm -rf msadapter\/tests' test_convert_llm.sh
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
