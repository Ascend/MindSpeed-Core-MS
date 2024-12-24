mindspeed_ms.core.parallel_state.initialize_model_parallel
==========================================================

.. py:function:: mindspeed_ms.core.parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None, pipeline_model_parallel_split_rank=None, context_parallel_size=1, expert_model_parallel_size=1, order="tp-cp-ep-dp-pp", communicator_config_path=None, **kwargs)

    初始化模型数据并行组。

    参数：
        - **tensor_model_parallel_size** (int，可选) - 将单个张量分割到多少个设备（NPU，GPU等设备）上。默认值： ``1`` 。
        - **pipeline_model_parallel_size** (int，可选) - 将Transformer层分割到多少个张量并行设备组（NPU，GPU等设备）中。默认值： ``1`` 。
        - **virtual_pipeline_model_parallel_size** (int，可选) - 每个流水线组将拥有多少个阶段，必要时进行交错。如果为 ``None`` ，则不进行交错。默认值： ``None`` 。
        - **pipeline_model_parallel_split_rank** (int，可选) - 对于具有编码器和解码器的模型，在流水线中切换编码器和解码器的秩（即解码器的第一个秩）（秩，英文为rank，可以简单理解为在分布式训练中设备或者处理单元的编号、顺序之类的标识）。这允许用户独立设置编码器和解码器的流水线并行大小。默认值： ``None`` 。
        - **context_parallel_size** (int，可选) - 将网络输入序列长度分割到多少个张量并行设备组中。计算注意力模块需要完整的序列长度的token，因此上下文并行组中的设备需要相互通信以交换其他序列块的信息。每个设备及其在其他张量并行组中的对应物组成一个上下文并行组。默认值： ``1`` 。
        - **expert_model_parallel_size** (int，可选) - 在MoE模型中，将专家分割到多少个设备上。默认值： ``1`` 。
        - **order** (str，可选) - 每个并行策略遵循的顺序。默认值： ``"tp-cp-ep-dp-pp"`` 。
        - **communicator_config_path** (str，可选) - HCCL通信器配置的yaml文件路径。目前未使用。默认值： ``None`` 。
        - **kwargs** (dict) - 额外的关键字配置参数。

    异常：
        - **RuntimeError** - `mindspore.communication._comm_helper` 没有被初始化。
        - **RuntimeError** - `world_size` 不能被 `tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size` 整除。
        - **RuntimeError** - `data_parallel_size` 不能被 `expert_model_parallel_size` 整除。
        - **RuntimeError** - `expert_model_parallel_size > 1` 且 `context_parallel_size > 1`。
        - **RuntimeError** - `virtual_pipeline_model_parallel_size` 不是 ``None`` 且 `pipeline_model_parallel_size < 2`。
        - **RuntimeError** - `order` 是 ``None`` 。
        - **RuntimeError** - `order` 中有重复元素。
        - **RuntimeError** - `ep` 在 `order` 中，`ep-dp` 不在 `order` 中 且 `dp-ep` 也不在 `order` 中。
        - **RuntimeError** - `_GLOBAL_STREAM` 已经被初始化。
