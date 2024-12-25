mindspeed_ms.legacy.model.moe.moe_layer.MoELayer
================================================

.. py:class:: mindspeed_ms.legacy.model.moe.moe_layer.MoELayer(config: TransformerConfig, submodules=None, layer_number: int=None)

    专家层。

    参数：
        - **config** (TransformerConfig) - Transformer模型的配置，详情请参考TransformerConfig类。
        - **submodules** (MLPSubmodules，可选) - 保留参数，目前没有使用。默认值： ``None`` 。
        - **layer_number** (int，可选) - 保留参数，目前没有使用。默认值： ``None`` 。

    输入：
        - **hidden_states** (Tensor) - 局部专家的隐藏层输入。

    输出：
        两个张量的元组。

        - **output** (Tensor) - 局部专家的输出。
        - **mlp_bias** (Tensor) - 目前没有使用。

    异常：
        - **ValueError** - 如果 `ep_world_size` 小于等于 ``0`` 。
        - **ValueError** - 如果 `num_experts` 不能被 `ep_world_size` 整除。
        - **ValueError** - 如果 `local_expert_indices` 的元素数量大于等于 `num_experts` 。
        - **ValueError** - 如果 `moe_config.moe_token_dispatcher_type` 不为 `alltoall` 。
        - **ValueError** - 如果 `self.training` 为 ``True`` 且 `get_tensor_model_parallel_world_size()` 大于 ``1`` 且 `self.sp` 不为 ``True`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
