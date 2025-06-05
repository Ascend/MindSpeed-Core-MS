#!/usr/bin/env bash

MindSpeed_Core_MS_PATH=$(pwd)
echo ${MindSpeed_Core_MS_PATH}

python ./tools/convert/merge_patch.py \
--mindspeed_path ${MindSpeed_Core_MS_PATH}/MindSpeed/ \
--mindspeed_llm_path ${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/