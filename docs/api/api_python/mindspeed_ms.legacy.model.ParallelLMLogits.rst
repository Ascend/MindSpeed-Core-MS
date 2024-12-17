mindspeed_ms.legacy.model.ParallelLMLogits
==========================================

.. py:class:: mindspeed_ms.legacy.model.ParallelLMLogits(config, bias=False, compute_dtype=None)

    计算vocab中每一个token的logits。

    参数：
        - **config** (dict) - Transformer模型配置。详情请参考TransformerConfig类。
        - **bias** (bool，可选) - 指定模型是否使用偏置向量。默认值: ``False`` 。
        - **compute_dtype** (dtype.Number，可选) - 计算类型。默认值: ``None`` 。

    输入：
        - **input_** (Tensor) - 隐藏状态的张量。
        - **word_embedding_table** (Parameter) - 从嵌入层通过的权重矩阵。
        - **parallel_output** (bool，可选) - 指定是否返回各张量并行权重上的并行输出。默认值: ``True`` 。
        - **bias** (Tensor，可选) - 可训练的偏置参数。默认值: ``None`` 。

    输出：
        - **logits_parallel** (Tensor) - 如果在 ParallelLMLogits 中设置 parallel_output 为 ``True`` ，则每个张量并行等级上的输出将是一个并行的logits张量，否则，输出将是一个收集所有并行输出的logits张量。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
