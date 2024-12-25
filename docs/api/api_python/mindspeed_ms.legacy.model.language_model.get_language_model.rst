mindspeed_ms.legacy.model.language_model.get_language_model
===========================================================

.. py:function:: mindspeed_ms.legacy.model.language_model.get_language_model(config, num_tokentypes, add_pooler, encoder_attn_mask_type, add_encoder=True, add_decoder=False, decoder_attn_mask_type=None, pre_process=True, post_process=True)

    使用此函数来获取语言模型。

    参数：
        - **config** (TransformerConfig) - Transformer模型配置，包括初始化函数和并行参数配置等。
        - **num_tokentypes** (int) - 如果大于0，则使用tokentypes嵌入。
        - **add_pooler** (bool) - 如果为 ``True`` ，使用池化层。
        - **encoder_attn_mask_type** (int) - 编码器注意力掩码类型。
        - **add_encoder** (bool，可选) - 如果为 ``True`` ，使用编码器。默认值：``True`` 。
        - **add_decoder** (bool，可选) - 如果为 ``True`` ，使用解码器。默认值：``False`` 。
        - **decoder_attn_mask_type** (int，可选) - 解码器注意力掩码类型。默认值：``AttnMaskType.causal`` 。
        - **pre_process** (bool，可选) - 使用流水线并行时，标记它是否为第一阶段。默认值：``True`` 。
        - **post_process** (bool，可选) - 使用流水线并行时，标记它是否为最后的阶段。默认值：``True`` 。

    返回：
        - **language_model** (TransformerLanguageModel) - Transformer模型。
        - **language_model_key** (str) - 模型的键。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
