mindspeed_ms.core.config.ModelParallelConfig
============================================

.. py:class:: mindspeed_ms.core.config.ModelParallelConfig(tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, context_parallel_size: int = 1, context_parallel_algo: str = "ulysses_cp_algo", ulysses_degree_in_cp: int = None, expert_model_parallel_size: int = 1, virtual_pipeline_model_parallel_size: int = None, sequence_parallel: bool = False, recv_dtype: str = "float32", zero_level: str = None, standalone_embedding_stage: bool = False, overlap_grad_reduce: bool = False, gradient_accumulation_fusion: bool = False, overlap_p2p_comm: bool = True, use_cpu_initialization: bool = False, deterministic_mode: bool = False, num_layer_list: list = None, recompute_config: dict = None, recompute: str = None, select_recompute: str = None, select_comm_recompute: str = None, variable_seq_lengths: bool = False, **kwargs)

    模型并行配置类。

    参数：
        - **tensor_model_parallel_size** (int，可选) - 张量平行的维数。默认值： ``1`` 。
        - **pipeline_model_parallel_size** (int，可选) - 使用流水线并行时的阶段数。默认值： ``1`` 。
        - **context_parallel_size** (int，可选) - 上下文并行的维度。默认值： ``1`` 。
        - **context_parallel_algo** (str，可选) - 上下文并行算法。默认值： ``"ulysses_cp_algo"`` 。可选项：[ ``"ulysses_cp_algo"`` ， ``"megatron_cp_algo"`` ， ``"hybrid_cp_algo"`` ]。
        - **ulysses_degree_in_cp** (int，可选) - 当 `--context-parallel-algo` 设置为 ``hybrid_cp_algo``  且环注意力并行度设置为 ``cp//ulyess`` 时，定义ulyess并行度的程度。
        - **expert_model_parallel_size** (int，可选) - 专家并行的维度。默认值： ``1`` 。
        - **virtual_pipeline_model_parallel_size** (int，可选) - 使用流水线并行时的虚拟阶段数。默认值： ``None`` 。
        - **sequence_parallel** (bool，可选) - 启用序列并行。默认值： ``False`` 。
        - **recv_dtype** (str，可选) - 使用管道并行时p2p通信的通信数据类型。默认值： ``"float32"`` 。
        - **zero_level** (str，可选) - ZeRO优化器的级别，如果为 ``None`` ，则不会使用ZeRO优化器。默认值： ``None`` 。
        - **standalone_embedding_stage** (bool，可选) - 启用独立嵌入。默认值： ``False`` 。
        - **overlap_grad_reduce** (bool，可选) - 启用重叠梯度下降。默认值： ``False`` 。
        - **gradient_accumulation_fusion** (bool，可选) - 在线性向后执行期间启用梯度累加。默认值： ``False`` 。
        - **overlap_p2p_comm** (bool，可选) - 在流水线交错模式下启用重叠p2p通信。默认值： ``True`` 。
        - **use_cpu_initialization** (bool，可选) - 使用CPU初始化。默认值： ``False`` 。
        - **deterministic_mode** (bool，可选) - 确定性计算模式。默认值： ``False`` 。
        - **num_layer_list** (list，可选) - 自定义流水线并行模型层划分。默认值： ``None`` 。
        - **recompute_config** (dict，可选) - 重计算配置。默认值： ``None`` 。
        - **recompute** (str，可选) - 启用重计算。默认值： ``None`` 。
        - **select_recompute** (str，可选) - 启用选择重计算。默认值： ``None`` 。
        - **select_comm_recompute** (str，可选) - 启用选择通信重计算。默认值： ``None`` 。
        - **variable_seq_lengths** (bool，可选) - 启用可变序列长度。默认值： ``False`` 。
        - **kwargs** (dict) - 额外的关键字配置参数。
