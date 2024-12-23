mindspeed_ms.legacy.model.eos_mask.EosMask
==========================================

.. py:class:: mindspeed_ms.legacy.model.eos_mask.EosMask(batch_size, seq_len, eod_token_id, reset_position_ids)

    生成特定token对应的注意力掩码。

    参数：
        - **batch_size** (int) - 批大小。
        - **seq_len** (int) - 序列长度。
        - **eod_token_id** (int) - 文档结束标记的标识符。
        - **reset_position_ids** (bool) - 如果为 ``True`` ，位置索引将被重置。

    输入：
        - **input_ids** (Tensor) - 输入索引。形状为 :math:`(B, S)` 的张量。

    输出：
        - **position_ids** (Tensor) - 位置索引。形状为 :math:`(B, S)` 的张量。
        - **mask** (Tensor) - 掩码。形状为 :math:`(B, S, S)` 的张量。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
