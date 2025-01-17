# 简介

MindSpeed-Core-MS 是基于MindSpore+昇腾设备的大模型加速库。

# 使用

1. 安装MindSpore https://www.mindspore.cn/install/
2. `bash apply_llm.sh`，脚本里会自动克隆MindSpeed-LLM，MindSpeed，Megatron-LM和transformers仓，根据msadaptor中的diff完成patch
3. `source set_path.sh` 设置PYTHONPATH
4. 执行模型训练脚本
