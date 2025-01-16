# 拉取MindSpeed-Core-MS最新代码

```shell
git clone https://gitee.com/ascend/MindSpeed-Core-MS.git -b dev
cd MindSpeed-Core-MS/
```

# 拉取MindSpeed-MM并应用patch

```shell
git clone https://gitee.com/ascend/MindSpeed-MM.git
cd MindSpeed-MM/
git checkout 9526e82399d9db8a18cf3f2aa7089f0b928e57b0
git apply ../msadaptor/mindspeed_mm.diff
cd ..
```

# 拉取MindSpeed-Core并应用patch

```shell
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout ab39de78be23e88e2c8b0d25edf6135940990c02
git apply ../msadaptor/acclerate_mindspeed_mm.diff
git diff
cd ..
```

# 拉取megatron并应用patch

```shell
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM/
git checkout core_r0.6.0
git apply ../msadaptor/megatron.diff
cd ..
```

# 拉取transformers并应用patch

```shell
git clone https://github.com/huggingface/transformers.git -b v4.47.0
cd transformers
git apply ../msadaptor/transformers.diff
cd ..
```

# 设置环境变量

```shell
MindSpeed-Core-MS_PATH=PATH
export PYTHONPATH=$MindSpeed-Core-MS_PATH/msadaptor:$MindSpeed-Core-MS_PATH/Megatron-LM/:$MindSpeed-Core-MS_PATH/MindSpeed/:$MindSpeed-Core-MS_PATH/transformers/src/:$PYTHONPATH
cd MindSpeed-MM/
```

# 正常修改shell脚本并运行