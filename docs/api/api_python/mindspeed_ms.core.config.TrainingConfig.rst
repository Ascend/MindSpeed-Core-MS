mindspeed_ms.core.config.TrainingConfig
=========================================

.. py:class:: mindspeed_ms.core.config.TrainingConfig(parallel_config: ModelParallelConfig, dataset_config: DatasetConfig = DatasetConfig(), lora_config: LoraConfig = LoraConfig(), seed: int = None, output_dir: str = "./output", training_iters: int = 0, epochs: int = None, log_interval: int = None, eval_interval: int = None, save_interval: int = None, best_metric_comparison: str = None, eval_metric: str = None, grad_clip_kwargs: dict = None, loss_scale: Union[float, int] = None, loss_scale_value: Union[float, int] = None, loss_scale_factor: int = None, loss_scale_window: int = None, loss_reduction: str = "mean", calculate_per_token_loss: bool = False, wrap_with_ddp: bool = False, accumulate_allreduce_grads_in_fp32: bool = False, overlap_grad_reduce: bool = False, delay_grad_reduce: bool = False, use_distributed_optimizer: bool = False, bucket_size: Optional[int] = None, check_for_nan_in_grad: bool = False, fp16: bool = False, bf16: bool = False, resume_training: bool = False, crc_check: bool = False, load_checkpoint: str = "", enable_compile_cache: bool = False, compile_cache_path: str = None, ckpt_format: str = "ckpt", prefix: str = "network", keep_checkpoint_max: int = 5, no_load_optim: bool = False, no_load_rng: bool = True, new_dataset: bool = False, enable_mem_align: bool = False, profile: bool = False, profile_save_path: str = None, profile_step_start: int = 1, profile_step_end: int = 5, profile_level: str = "level0", profile_with_stack: bool = False, profile_memory: bool = False, profile_framework: str = "all", profile_communication: bool = False, profile_parallel_strategy: bool = False, profile_aicore_metrics: int = 0, profile_l2_cache: bool = False, profile_hbm_ddr: bool = False, profile_pcie: bool = False, profile_data_process: bool = False, profile_data_simplification: bool = False, profile_op_time: bool = True, profile_offline_analyse: bool = False, profile_dynamic_profiler_config_path: str = "", **kwargs)

    训练配置类。

    参数：
        - **parallel_config** (ModelParallelConfig) - 并行配置。
        - **dataset_config** (DatasetConfig) - 数据配置。默认值： ``DatasetConfig()`` 。
        - **lora_config** (LoraConfig) - Lora配置。默认值：``LoraConfig()`` 。
        - **seed** (int，可选) - 初始化使用的随机种子。默认值： ``None`` 。
        - **output_dir** (str，可选) - 用来存放ckpt和日志的目录。默认值： ``"./output"`` 。
        - **training_iters** (int，可选) - 训练使用的迭代次数。默认值： ``0`` 。
        - **epochs** (int，可选) - 训练使用的迭代周期次数。默认值： ``None`` 。
        - **log_interval** (int，可选) - 训练日志输出间隔。默认值： ``None`` 。
        - **eval_interval** (int，可选) - 训练评估间隔。默认值： ``None`` 。
        - **save_interval** (int，可选) - 训练存储间隔。默认值： ``None`` 。
        - **best_metric_comparison** (str，可选) - 比较最佳指标的方法。默认值： ``None`` 。
        - **eval_metric** (str，可选) - 评估指标的名称。默认值： ``None`` 。
        - **grad_clip_kwargs** (dict，可选) - 梯度裁剪参数。默认值： ``None`` 。
        - **loss_scale** (Union[float, int]，可选) - 损失缩放的初始值。如果设置，将使用静态损失缩放器（static loss scaler）。默认值： ``None`` 。
        - **loss_scale_value** (Union[float, int]，可选) - 动态损失缩放的初始值。默认值： ``None`` 。
        - **loss_scale_factor** (int，可选) - 动态损失缩放因子。默认值： ``None`` 。
        - **loss_scale_window** (int，可选) - 动态损失缩放窗口大小。默认值： ``None`` 。
        - **loss_reduction** (str，可选) - 损失归约（Loss Reduction）方法。默认值： ``"mean"`` 。
        - **calculate_per_token_loss** (bool，可选) - 根据令牌数量应用梯度和损失计算。默认值： ``False`` 。
        - **wrap_with_ddp** (bool，可选) - 使用分布式数据并行封装模型。默认值： ``False`` 。
        - **accumulate_allreduce_grads_in_fp32** (bool，可选) - 在fp32开启时, 是否累积 `allreduce` 梯度。默认值： ``False`` 。
        - **overlap_grad_reduce** (bool，可选) - 使用分布式数据并行时启用梯度计算和同步通信重叠。默认值： ``False`` 。
        - **delay_grad_reduce** (bool，可选) - 如果设置为 ``True`` ，则在除第一个PP阶段外的所有阶段中延迟梯度归约。默认值： ``False`` 。
        - **use_distributed_optimizer** (bool，可选) - 使用分布式数据并行时使用分布式优化器。默认值： ``False`` 。
        - **bucket_size** (Optional[int]，可选) - 当 `overlap_grad_reduce` 为 ``True`` 时，用于将缓冲区划分为桶的桶大小。默认值： ``None`` 。
        - **check_for_nan_in_grad** (bool，可选) - 如果设置 ``True`` ，则同步后会检查缓冲区中的梯度是否是有限的（非NaN的）。默认值： ``False`` 。
        - **fp16** (bool，可选) - 是否使用 ``fp16`` 类型。默认值： ``False`` 。
        - **bf16** (bool，可选) - 是否使用 ``bf16`` 类型。默认值： ``False`` 。
        - **resume_training** (bool，可选) - 是否开启续训。默认值： ``False`` 。
        - **crc_check** (bool，可选) - 保存/加载ckpt时进行CRC检查，启用此选项可能会导致训练性能降低。默认值： ``False`` 。
        - **load_checkpoint** (str，可选) - 加载ckpt的路径。默认值： ``""`` 。
        - **enable_compile_cache** (bool，可选) - 保存编译缓存，启用此选项可能会导致训练性能降低。默认值： ``False`` 。
        - **compile_cache_path** (str，可选) - 保存编译缓存的路径。默认值： ``None`` 。
        - **ckpt_format** (str，可选) - ckpt保存的格式。默认值： ``"ckpt"`` 。
        - **prefix** (str，可选) - ckpt保存的前缀。默认值： ``"network"`` 。
        - **keep_checkpoint_max** (int，可选) - 存储ckpt的最大数量。默认值： ``5`` 。
        - **no_load_optim** (bool，可选) - 恢复训练时，是否加载优化器状态。默认值： ``False`` 。
        - **no_load_rng** (bool，可选) - 恢复训练时，是否加载RNG状态。默认值： ``True`` 。
        - **new_dataset** (bool，可选) - 恢复训练时，是否使用新数据集。默认值： ``False`` 。
        - **enable_mem_align** (bool，可选) - 是否启用内存对齐。默认值： ``False`` 。
        - **profile** (bool，可选) - 是否打开分析。默认值： ``False`` 。
        - **profile_save_path** (str，可选) - 保存分析文件的路径。默认值： ``None`` 。
        - **profile_step_start** (int，可选) - 分析开始的步骤。默认值： ``1`` 。
        - **profile_step_end** (int，可选) - 分析结束的步骤。默认值： ``5`` 。
        - **profile_level** (str，可选) - 分析级别。默认值： ``"level0"`` 。
        - **profile_with_stack** (bool，可选) - 使用堆栈信息进行分析。默认值： ``False`` 。
        - **profile_memory** (bool，可选) - 使用内存信息进行分析。默认值： ``False`` 。
        - **profile_framework** (str，可选) - 使用框架信息进行分析。默认值： ``"all"`` 。
        - **profile_communication** (bool，可选) - 使用通信信息进行分析。默认值： ``False`` 。
        - **profile_parallel_strategy** (bool，可选) - 使用并行策略信息进行分析。默认值： ``False`` 。
        - **profile_aicore_metrics** (int，可选) - 使用aicore度量信息进行分析。默认值： ``0`` 。
        - **profile_l2_cache** (bool，可选) - 使用二级缓存信息进行分析。默认值： ``False`` 。
        - **profile_hbm_ddr** (bool，可选) - 使用hbm ddr信息进行分析。默认值： ``False`` 。
        - **profile_pcie** (bool，可选) - 使用pcie信息进行分析。默认值： ``False`` 。
        - **profile_data_process** (bool，可选) - 使用数据进程信息进行分析。默认值： ``False`` 。
        - **profile_data_simplification** (bool，可选) - 使用数据简化进行分析。默认值： ``False`` 。
        - **profile_op_time** (bool，可选) - 使用操作时间信息进行分析。默认值： ``True`` 。
        - **profile_offline_analyse** (bool，可选) - 使用离线分析。默认值： ``False`` 。
        - **profile_dynamic_profiler_config_path** (str，可选) - 动态分析器的配置路径。默认值： ``""`` 。
        - **kwargs** (dict) - 额外的关键字配置参数。

    异常：
        - **ValueError** - `fp16` 是 ``True`` 并且 `bf16` 也是 ``True`` 。
