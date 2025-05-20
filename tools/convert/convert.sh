#!/bin/bash
cp -r MindSpeed/mindspeed MindSpeed-LLM/
cp -r Megatron-LM/megatron MindSpeed-LLM/
cp -r transformers/src/transformers MindSpeed-LLM/

cp -r msadapter/mindtorch/apex MindSpeed-LLM/
cp -r msadapter/mindtorch/torchair MindSpeed-LLM/
cp -r msadapter/mindtorch/torchvision MindSpeed-LLM/
cp -r msadapter/mindtorch/transformer_engine MindSpeed-LLM/
cp -r msadapter/mindtorch/torch MindSpeed-LLM/msadapter
cp -r msadapter/mindtorch/torch_npu MindSpeed-LLM/msadapter_npu

# handle third party 
third_party_pkg='tensordict'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg MindSpeed-LLM/
third_party_pkg='safetensors'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg MindSpeed-LLM/
third_party_pkg='einops'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg MindSpeed-LLM/
third_party_pkg='peft'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg MindSpeed-LLM/
third_party_pkg='accelerate'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg MindSpeed-LLM/

cd MindSpeed-LLM/einops
patch -p1 < ../../tools/rules/einops.diff
cd -

# handle file with special code
F_WITH_BAD_CODE=MindSpeed-LLM/mindspeed/core/auto_parallel/auto_parallel_memory.py
grep -avP '[^\x00-\x7F]' $F_WITH_BAD_CODE > tmp.py && mv tmp.py $F_WITH_BAD_CODE
F_WITH_BAD_CODE=MindSpeed-LLM/mindspeed/core/auto_parallel/mm_search/optimizer.py
grep -avP '[^\x00-\x7F]' $F_WITH_BAD_CODE > tmp.py && mv tmp.py $F_WITH_BAD_CODE

python tools/convert/convert.py --path_to_change MindSpeed-LLM/
