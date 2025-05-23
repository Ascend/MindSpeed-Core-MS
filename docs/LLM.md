# 拉取MindSpeed-Core-MS最新代码

```shell
git clone https://gitee.com/ascend/MindSpeed-Core-MS.git -b dev
cd MindSpeed-Core-MS/
```

# 拉取MindSpeed-LLM并应用patch

```shell
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM/
git checkout 1.0.0
git apply ../msadaptor/mindspeed_llm.diff
cd ..
```

# 拉取MindSpeed-Core并应用patch

```shell
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 969686ff
git apply ../msadaptor/acclerate_mindspeed_llm.diff
git diff
cd ..
```

# 拉取megatron并应用patch

```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
git checkout core_r0.6.0
git apply ../msadaptor/megatron.diff
```

# 拉取transformers并应用patch

```shell
git clone https://github.com/huggingface/transformers.git -b v4.47.0
cd transformers
git apply ../msadaptor/transformers.diff
```

# 设置环境变量

```shell
MindSpeed_Core_MS_PATH=PATH
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadaptor:${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/MindSpeed/:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
cd MindSpeed-LLM/
```

# 正常修改shell脚本并运行