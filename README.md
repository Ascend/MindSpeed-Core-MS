# 简介

MindSpeed-Core-MS 是基于MindSpore+昇腾设备的大模型加速库。

# LLM使用

1. 安装MindSpore https://www.mindspore.cn/install/
2. `source auto_convert_llm.sh`，脚本里会自动克隆MindSpeed-LLM，MindSpeed，Megatron-LM, transformers和peft仓，根据tools中的转换工具完成patch，同时设置PYTHONPATH
3. 执行模型训练脚本

# MM使用

1. 安装MindSpore https://www.mindspore.cn/install/
2. `source auto_convert_mm.sh`，脚本里会自动克隆MindSpeed-MM，MindSpeed，Megatron-LM和transformers仓，根据tools中的转换工具完成patch，同时设置PYTHONPATH
3. 执行模型训练脚本
