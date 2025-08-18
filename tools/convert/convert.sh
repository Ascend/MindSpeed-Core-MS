#!/bin/bash
dir=$1
echo "start convert $dir"
cp -r MindSpeed/mindspeed ${dir}
cp -r Megatron-LM/megatron ${dir}
cp -r transformers/src/transformers ${dir}

cp -r msadapter/msa_thirdparty/apex ${dir}
cp -r msadapter/msa_thirdparty/torchair ${dir}
cp -r msadapter/msa_thirdparty/gpytorch ${dir}
cp -r msadapter/msa_thirdparty/transformer_engine ${dir}
cp -r msadapter/msadapter/ ${dir}/msadapter
cp -r msadapter/msa_thirdparty/torch_npu ${dir}/msadapter_npu
cp -r msadapter/csrc ./

# handle third party 
third_party_pkg='tensordict'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg ${dir}
third_party_pkg='einops'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg ${dir}
third_party_pkg='peft'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg ${dir}
third_party_pkg='accelerate'
install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
cp -r $install_path/$third_party_pkg ${dir}

if [[ "${dir%/}" == "MindSpeed-LLM" ]]; then
    third_party_pkg='safetensors'
    install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
    cp -r $install_path/$third_party_pkg ${dir}
fi

if [[ "${dir%/}" == "MindSpeed-MM" ]]; then
    cp -r safetensors_dir/safetensors ${dir}
    third_party_pkg='diffusers'
    install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
    cp -r $install_path/$third_party_pkg ${dir}
    third_party_pkg='timm'
    install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
    cp -r $install_path/$third_party_pkg ${dir}
    third_party_pkg='qwen_vl_utils'
    install_path=$(python -m pip show $third_party_pkg | grep 'Location:' | awk '{print $2}')
    cp -r $install_path/$third_party_pkg ${dir}
fi

cd ${dir}
python ../tools/convert/convert.py
echo "finish convert $dir"
