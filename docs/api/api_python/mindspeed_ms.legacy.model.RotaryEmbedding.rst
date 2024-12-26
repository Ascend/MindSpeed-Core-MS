mindspeed_ms.legacy.model.RotaryEmbedding
=========================================

.. py:class:: mindspeed_ms.legacy.model.RotaryEmbedding(kv_channels, rotary_percent=1.0, rotary_interleaved=False, seq_len_interpolation_factor=None, rotary_base=10000)

    用于语言模型的旋转位置嵌入。

    参数：
        - **kv_channels** (int) - 多头注意力中的投影权重维度。从transformer的配置中获取。
        - **rotary_percent** (float, 可选) - 旋转位置编码中旋转维度的使用比例。默认值： ``1.0`` 。
        - **rotary_interleaved** (bool, 可选) - 是否以交错方式将旋转位置编码应用于输入维度。默认值： ``False`` 。目前暂不支持设置为 ``True`` 。
        - **seq_len_interpolation_factor** (float, 可选) - 对更长序列进行线性插值的比例。如果设置非None，则该值必须是大于1.0的浮点数。默认值： ``None`` 。
        - **rotary_base** (int, 可选) - 旋转位置嵌入编码的基期。默认值： ``10000`` 。

    输入：
        - **max_seq_len** (int) - 输入的最大序列长度。
        - **offset** (int) - 位置编码偏移量。

    输出：
        - **emb** (Tensor) - 应用旋转位置编码后的嵌入向量。

    异常：
        - **NotImplementedError** - 当 `rotary_interleaved` 为 ``True`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
