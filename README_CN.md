# 欢迎来到MindSpeed-Core-MS

## 一、介绍

MindSpeed-Core-MS 是基于MindSpore+昇腾设备的大模型加速库，涵盖丰富的并行特性（数据并行、张量并行、流水线并行、序列并行等场景），
还支持使用分布式优化器、动态学习率等场景，期望帮助开发者快速、便捷基于动态图并行框架的大模型预训练。关于MindSpeed-Core-MS相关使用教程以及API文档，请参阅[MindSpeed-Core-MS文档](https://gitee.com/ascend/MindSpeed-Core-MS/tree/r0.1.0/docs)。

如果您对MindSpeed-Core-MS有任何建议，请通过issue与我们联系，我们将及时处理。

> 本文档主要涉及MindSpeed-Core-MS核心代码等相关内容，若需要运行PanGu_ms/38BV3等模型，请参考[动态图并行-大模型预训练开发指南](https://gitee.com/ascend/MindSpeed-Core-MS/blob/r0.1.0/docs/source_zh_cn/usage/pretrain_gpt.md)，以及盘古仓[配置教程](https://codehub-y.huawei.com/MindSpore-enterprise/Production/38Bv3/files?ref=refactor&filePath=pangu_sophon_pytorch-master%2FPanGu_ms%2FREADME.md&isFile=true)准备相关数据集、封装层转换脚本`launch.py` 及启动训练脚本`run_ms_launch.sh`。

## 二、安装

MindSpeed-Core-MS目前支持**源码使用**。

```shell
git clone -b r0.1.0 https://gitee.com/ascend/MindSpeed-Core-MS.git
```

注意安装包的名称为`mindspeed_ms`，导入所需接口时可以使用`from mindspeed_ms import xxx`。

```python
from mindspeed_ms import xxx
```

## 三、使用指南

MindSpeed-Core-MS支持Transformer类预训练模型，以下为模型分布式启动方式的说明与示例。

MindSpeed-Core-MS推荐使用分布式方式拉起模型训练，目前以`msrun`分布式启动作为模型的主要启动方式，`msrun`特性说明可以参考[msrun启动](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html)。
该特性的主要输入参数说明如下：

  | **参数**           | **单机是否必选** | **多机是否必选** |     **默认值**      | **说明**           |
  |------------------|:----------:|:----------:|:----------------:|------------------|
  | worker_num       |  &check;   |  &check;   |        8         | 所有节点中使用计算卡的总数    |
  | local_worker_num |     -      |  &check;   |        8         | 当前节点中使用计算卡的数量    |
  | master_addr      |     -      |  &check;   |    127.0.0.1     | 指定分布式启动主节点的ip    |
  | master_port      |     -      |  &check;   |       8118       | 指定分布式启动绑定的端口号    |
  | node_rank        |     -      |  &check;   |        0         | 指定当前节点的rank id   |
  | log_dir          |     -      |  &check;   | output/msrun_log | 日志输出路径，若不存在则递归创建 |
  | join             |     -      |  &check;   |      False       | 是否等待所有分布式进程退出    |
  | cluster_time_out |     -      |  &check;   |       7200       | 分布式启动的等待时间，单位为秒  |

> 注：如果需要指定`device_id`启动，可以设置环境变量`ASCEND_RT_VISIBLE_DEVICES`，如要配置使用2、3卡则输入`export ASCEND_RT_VISIBLE_DEVICES=2,3`。

以下为以msrun方式启动多卡训练的示例，可将以下启动命令保存至`run.sh`后，使用`bash run.sh`运行。

- `msrun`使用示例

  ```bash
  msrun --worker_num=8 \
        --local_worker_num=8 \
        --master_port=8132 \
        --log_dir=log \
        --join=True \
        --cluster_time_out=300 \
        run_language_model.py --config-path /path/to/config/test_language_model.yaml
  ```

上述msrun启动命令中的`run_language_model.py`代码及`test_language_model.yaml` 可以参考[配置示例](https://gitee.com/ascend/MindSpeed-Core-MS/tree/r0.1.0/tests/st/test_distri_core/test_language_model)
