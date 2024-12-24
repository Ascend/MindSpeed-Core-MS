mindspeed_ms.legacy.model.gpt_model.GPTModel
============================================

.. py:class:: mindspeed_ms.legacy.model.gpt_model.GPTModel(config, num_tokentypes=0, parallel_output=True, pre_process=True, post_process=True, **kwargs)

    生成式预训练Transformer（GPT）的实现，是一个仅有解码器的Transformer模型。

    参数：
        - **config** (TransformerConfig) - Transformer模型配置，包括初始化函数和并行参数配置等。
        - **num_tokentypes** (int，可选) - 如果大于0，则使用tokentype嵌入。默认值：0.
        - **parallel_output** (bool，可选) - 指定是否返回各张量并行权重上的并行输出。默认值： ``True`` 。
        - **pre_process** (bool，可选) - 使用流水线并行时，标记它是否为第一阶段。默认值： ``True`` 。
        - **post_process** (bool，可选) - 使用流水线并行时，标记它是否为最后的阶段。默认值： ``True`` 。
        - **kwargs** (dict) - 其他输入。

    输入：
        - **tokens** (tuple[Tensor]) - 输入索引。形状为 :math:`(B, S)` 。
        - **position_ids** (tuple[Tensor]) - 位置偏移量。形状为 :math:`(B, S)` 。
        - **attention_mask** (tuple[Tensor]) - 注意力掩码。形状为 :math:`(B, S)` 。
        - **loss_mask** (tuple[Tensor]) - 损失掩码。形状为 :math:`(B, S)` 。
        - **retriever_input_ids** (tuple[Tensor]，可选) - 检索器输入标记索引。默认值： ``None`` 。
        - **retriever_position_ids** (tuple[Tensor]，可选) - 检索器输入位置索引。默认值： ``None`` 。
        - **labels** (tuple[Tensor]，可选) - 样本的基准真实值。形状为 :math:`(N, )` 。默认值： ``None`` 。
        - **tokentype_ids** (tuple[Tensor]，可选) - 给模型输入的标记类型索引列表。形状为 :math:`(B, S)` 。默认值： ``None`` 。
        - **inference_params** (tuple[Tensor]，可选) - 推理参数，用于在推理过程中指定特定设置，如最大生成长度、最大批处理大小等。默认值： ``None`` 。

    输出：
        - 返回GPT模型的loss或hidden states。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
