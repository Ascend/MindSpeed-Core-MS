# 简介

MindSpeed-Core-MS 是基于MindSpore+昇腾设备的大模型加速库。

# LLM使用

1. 安装MindSpore https://www.mindspore.cn/install/
2. `source apply_llm.sh`，脚本里会自动克隆MindSpeed-LLM，MindSpeed，Megatron-LM, transformers和peft仓，根据msadaptor中的diff完成patch，同时设置PYTHONPATH
3. 执行模型训练脚本

# MM使用

1. 安装MindSpore https://www.mindspore.cn/install/
2. `source apply_mm.sh`，脚本里会自动克隆MindSpeed-MM，MindSpeed，Megatron-LM和transformers仓，根据msadaptor中M的diff完成patch，同时设置PYTHONPATH
3. 执行模型训练脚本

# 转换工具

1. 执行`test_convert_llm.sh`脚本
2. 添加代码仓到`PYTHONPATH`中

```shell
MindSpeed_Core_MS_PATH=$PWD
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
```