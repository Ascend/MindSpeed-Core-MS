#!/bin/bash
# set -e

ROOT_DIR=$1
PATCH_JSON_PATH=$2
MERGE_PY="tools/convert/patch_merge/modules/merge.py"

export PYTHONPATH=$ROOT_DIR/"MindSpeed-LLM":$PYTHONPATH
export PYTHONPATH=$ROOT_DIR/"MindSpeed":$PYTHONPATH
export PYTHONPATH=$ROOT_DIR/"Megatron-LM":$PYTHONPATH
export PYTHONPATH=$ROOT_DIR/"transformers/src":$PYTHONPATH
export PYTHONPATH=$ROOT_DIR/"MSAdapter/mindtorch/":$PYTHONPATH

echo "Start merge ${ROOT_DIR} with ${PATCH_JSON_PATH}"
python $MERGE_PY \
    --root-dir $ROOT_DIR \
    --json-file $PATCH_JSON_PATH \
    | tee $ROOT_DIR/merge.log

echo "Check log ${ROOT_DIR}/merge.log"