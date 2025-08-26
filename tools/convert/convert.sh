#!/bin/bash
dir="${1%/}"

if [[ -z "$dir" ]]; then
    echo "Error: Missing required argument."
    exit 1
fi

if [[ ! -d "$dir" ]]; then
    echo "Error: Directory '$dir' does not exist."
    exit 1
fi

safe_copy() {
    local src="$1"
    local dst="$2"

    if [[ ! -d "$src" ]]; then
        echo "Error: Source directory not found: $src"
        return 1
    fi

    if [[ -e "$dst" ]]; then
        rm -rf "$dst"
    fi

    cp -r "$src" "$dst"
    echo "Copied: $src -> $dst"
}

copy_third_party() {
    local pkg_name=$1
    local install_path
    install_path=$(python -m pip show "$pkg_name" 2>/dev/null | grep 'Location:' | awk '{print $2}')

    if [[ -z "$install_path" ]]; then
        echo "Warning: Package '$pkg_name' not installed or not found via pip."
        return 1
    fi

    local src="$install_path/$pkg_name"
    local dst="$dir/$pkg_name"

    if [[ ! -d "$src" ]]; then
        echo "Warning: Package files not found at: $src"
        return 1
    fi

    if [[ -e "$dst" ]]; then
        rm -rf "$dst"
    fi

    cp -r "$src" "$dst"
    echo "Copied third-party package: $pkg_name"
}

echo "start copy process"
safe_copy "MindSpeed/mindspeed"           "$dir/mindspeed"
safe_copy "Megatron-LM/megatron"          "$dir/megatron"
safe_copy "msadapter/msa_thirdparty/apex" "$dir/apex"
safe_copy "msadapter/msa_thirdparty/torchair" "$dir/torchair"
safe_copy "msadapter/msa_thirdparty/gpytorch" "$dir/gpytorch"
safe_copy "msadapter/msadapter"           "$dir/msadapter"
safe_copy "msadapter/msa_thirdparty/torch_npu" "$dir/msadapter_npu"
safe_copy "msadapter/csrc" "./csrc"

copy_third_party 'tensordict'
copy_third_party 'einops'
copy_third_party 'peft'
copy_third_party 'accelerate'
copy_third_party 'transformers'

if [[ "$dir" == "MindSpeed-LLM" ]]; then
    copy_third_party 'safetensors'
fi

if [[ "$dir" == "MindSpeed-MM" ]]; then
    copy_third_party 'diffusers'
    copy_third_party 'timm'
    copy_third_party 'qwen_vl_utils'
fi

cd ${dir}
echo "start convert $dir"
python ../tools/convert/convert.py
echo "finish convert $dir"
