#!/bin/bash
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
parent_dir=$(dirname "$script_dir")
MindSpeed_Core_MS_PATH=$(dirname "$parent_dir")
cd ${MindSpeed_Core_MS_PATH}
sed -i '/echo ${MindSpeed_Core_MS_PATH}/i rm -rf MindSpeed-LLM\/tests\nrm -rf MindSpeed\/tests_extend\nrm -rf Megatron-LM\/tests\nrm -rf msadapter\/tests' auto_convert.sh llm
bash auto_convert.sh llm msa_latest
modifygrammar1() {
    fname=$1
    echo "Modifying PY310 grammar to adapt PY39..."
    # replace num_query_groups: int | None = None
    sed -i 's/^[[:space:]]*num_query_groups:.*= None/    num_query_groups = None/' "$fname"
    # replace ffn_hidden_size: float | None = None
    sed -i 's/^[[:space:]]*ffn_hidden_size:.*= None/    ffn_hidden_size = None/' "$fname"
    echo "PY310 grammar have been updated to adapt PY39 in $fname"
}
modifygrammar2() {
    fname=$1
    echo "Modifying PY310 grammar to adapt PY39..."
    # replace from typing import Type with from typing import Union, Type
    sed -i 's/^from typing import Type/from typing import Union, Type/' "$fname"
    # replace Type['BaseGMMFunction'] | None with Union[Type['BaseGMMFunction'], None]
    sed -i "s/-> Type\['BaseGMMFunction'\] | None:/-> Union[Type['BaseGMMFunction'], None]:/" "$fname"
    echo "PY310 grammar have been updated to adapt PY39 in $fname"
}
modifygrammar1 ${MindSpeed_Core_MS_PATH}/Megatron-LM/megatron/core/transformer/heterogeneous/heterogeneous_config.py
modifygrammar2 ${MindSpeed_Core_MS_PATH}/MindSpeed/mindspeed/core/transformer/moe/grouped_matmul_util.py
echo "..............................................done set LLM_env"
