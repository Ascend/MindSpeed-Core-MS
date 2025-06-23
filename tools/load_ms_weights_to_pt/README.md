# MS权重加载工具

适用场景：基于MindSpeed/Megatron生态的Pytorch模型，直接加载MindSpore兼容方案保存的.pt格式权重文件。

## 依赖文件

该工具的实现由两个核心文件组成：

1. checkpointing: 对MindSpeed-LLM中底层的torch.load()打patch, 当加载mindspore兼容方案保存的.pt格式权重时，会自动调用load_ms_weights，用户脚本侧无感。
2. serialization：定义权重加载的核心逻辑及关键接口：load_ms_weights。

## 使用说明

如需加载权重文件，需要在pretrain.py启动命令行中添加--load ${ms_weights_path}参数

### 方式一：兼容方式使用

通过在MindSpeed-LLM对torch.load接口打patch，调用torch.load接口时，如果加载mindspore兼容方案保存的.pt格式权重，会调用到load_ms_weights接口。  
MindSpeed-LLM版本要求：使用master分支。  

**步骤一**：执行patch，覆盖torch.load()  
在当前目录执行如下命令:

```shell
python transfer.py --mindspeed_llm_path ${Your_MindSpeed_Core_MS_PATH}/MindSpeed-LLM
```

命令执行后会在MindSpeed-Core-MS/MindSpeed-LLM/mindspeed_llm/tasks/megatron_adaptor.py文件中打patch，覆盖MindSpeed-LLM底层的torch.load()。  
**步骤二**：打开--no-load-rng开关，并按MindSpeed/Megatron使用方式执行训练

### 方式二：用户自定义使用

**步骤一**：将tools/load_ms_weights_to_pt目录下的serialization文件拷贝到MindSpeed-Core-MS/MindSpeed-LLM/mindspeed_llm/mindspore/training目录下。  
**步骤二**：根据用户自己的python引用层级，添加from serialization import load_ms_weights进行使用。
示例代码：  

```python
from serialization import load_ms_weights  # 导入load_ms_weights

# 权重文件路径
weights_path = args.load_weight_path  # 权重文件路径

# 使用load_ms_weights加载模型权重
try:
    weights = load_ms_weights(weights_path)  # 加载权重
    print("模型权重加载成功！")
except Exception as e:
    print(f"加载模型权重时出现错误: {e}")

# 之后可以将加载的权重应用到模型上
```
