mindspeed_ms.core.config.MoEConfig
=========================================

.. py:class:: mindspeed_ms.core.config.MoEConfig(num_experts = 1, moe_grouped_gemm = False, moe_router_topk = 2, moe_router_load_balancing_type = "none", moe_token_dispatcher_type = 'alltoall', use_self_defined_alltoall = False, moe_expert_capacity_factor = None, moe_pad_expert_input_to_capacity = False, moe_token_drop_policy = "probs", moe_aux_loss_coeff = 0.0, moe_z_loss_coeff = None, moe_input_jitter_eps = None, **kwargs)

    MoE配置类。

    参数：
        - **num_experts** (int，可选) - 专家的数量。默认值： ``1`` 。
        - **moe_grouped_gemm** (bool，可选) - 使能分组gemm。默认值： ``False`` 。
        - **moe_router_topk** (int，可选) - TopK路由数。默认值： ``2`` 。
        - **moe_router_load_balancing_type** (str，可选) - Moe路由负载均衡算法的类型。可选项：[ ``"aux_loss"`` , ``"none"`` ]。默认值： ``"none"`` 。
        - **moe_token_dispatcher_type** (str，可选) - Moe令牌调度算法的类型。可选项：[ ``'alltoall'`` ]。默认值： ``'alltoall'`` 。
        - **use_self_defined_alltoall** (bool，可选) - 使用自定义的alltoall操作符。默认值： ``False`` 。
        - **moe_expert_capacity_factor** (float，可选) - 每个专家的容量因子。默认值： ``None`` 。
        - **moe_pad_expert_input_to_capacity** (bool，可选) - 是否填充每个专家的输入以匹配专家容量长度。默认值： ``False`` 。
        - **moe_token_drop_policy** (str，可选) - 令牌弃置策略。默认值： ``"probs"`` 。
        - **moe_aux_loss_coeff** (float，可选) - 辅助损耗的缩放系数。默认值： ``0.0`` 。
        - **moe_z_loss_coeff** (float，可选) - Z损失的缩放系数。默认值： ``None`` 。
        - **moe_input_jitter_eps** (float，可选) - 通过应用具有指定epsilon值的抖动为输入张量添加噪声。默认值： ``None`` 。
        - **kwargs** (dict) - 额外的关键字配置参数。
