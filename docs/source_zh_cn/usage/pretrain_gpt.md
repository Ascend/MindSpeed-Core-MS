# 动态图并行-GPT模型预训练开发指南

## 概述

本教程演示如何使用MindSpeed-Core-MS动态图并行框架训练GPT模型，此框架支持张量并行、流水线并行、序列并行等并行场景，还支持使用分布式优化器、动态学习率等场景，帮助开发者快速、便捷地构建和训练基于动态图并行框架的GPT预训练模型。

## 操作实践

下面基于Ascend平台，进行GPT模型训练。

### 样例代码参考

目录结构可以参考如下[配置](https://codehub-y.huawei.com/MindSpore-enterprise/Production/38Bv3/files?ref=refactor&filePath=pangu_sophon_pytorch-master%2FPanGu_ms)：

```text
└─ gpt
    ├─ pretrain_gpt.py
    ├─ pretrain_gpt.sh
    └─ config
        └─ 38BV3
            ├─ model_38BV3_seq_4K.yaml
            ├─ model_38BV3_seq_4K_device_B2_accelerate.yaml
            └─ model_38BV3_seq_4K_train.yaml
```

其中，`pretrain_gpt.py`是环境配置、模型对象创建及训练的脚本。`pretrain_gpt.sh`是启动执行脚本。`config/38Bv3`文件夹里是配置项。

### 模型结构

GPT以`Transformer`模型为主要架构，网络结构主要围绕`Transformer`的基本构建块构建。本框架目前提供了一个构建GPT模型的高阶接口[GPTModel](https://gitee.com/ascend/MindSpeed-Core-MS/blob/r0.1.0/mindspeed_ms/legacy/model/gpt_model.py#L36)，供开发者参考，便于快速搭建网络。下面以此接口为例讲一下模型结构。

在模型中，初始化五个参数，`config`是模型配置项（在yaml文件的`model_config`中），`num_tokentypes`指定embedding的类型，`parallel_output`用来确认是否输出每一个并行Tensor的输出，`pre_process`和`post_process`分别指定是否为第一阶段和最后一阶段。

注意：数据集返回值要与模型定义的前向过程所需要的参数相对应。

```python
# 引入必需的模块
import enum
from mindspore import ops
from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.legacy.model.language_model import get_language_model
from mindspeed_ms.legacy.model.transformer import ParallelLMLogits
from mindspeed_ms.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from mindspeed_ms.training.loss_func import LossWithMask
from mindspeed_ms.legacy.model.eos_mask import EosMask
from mindspeed_ms.core.parallel_state import get_context_parallel_rank, get_context_parallel_world_size

# 定义attention mask的类型
class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
    no_mask = 3
    padding_causal = 4

# GPT模型定义
class GPTModel(Module):
    """
    GPT model
    """
    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 **kwargs):

        super().__init__(config)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.batch_size = config.dataset_config.batch_size
        self.cp_size = get_context_parallel_world_size()
        self.cp_rank = get_context_parallel_rank()

        # 获取eod mask数据
        self.eod = kwargs['eod'] if 'eod' in kwargs else None
        self.reset_position_ids = kwargs['reset_position_ids'] if 'reset_position_ids' in kwargs else False
        if self.eod:
            self.eod_mask = EosMask(self.batch_size, config.seq_length, self.eod, self.reset_position_ids)

        # 获取LLM模型
        self.language_model, _ = get_language_model(config=config,
                                                    encoder_attn_mask_type=AttnMaskType.causal,
                                                    num_tokentypes=num_tokentypes,
                                                    pre_process=self.pre_process,
                                                    post_process=self.post_process,
                                                    add_pooler=False)
        # 后处理部分代码，包括head层和loss计算
        if self.post_process:
            self.parallel_lm_logits = ParallelLMLogits(config=config,
                                                       bias=False,
                                                       compute_dtype=config.compute_dtype)
            self.loss = LossWithMask(VocabParallelCrossEntropy())
        # 流水线并行中embedding层和head层共享权重的功能
        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    # 流水线并行中需要调用来设置模型的输入
    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    # GPT模型执行
    def construct(self, tokens, position_ids, attention_mask, loss_mask,
                  retriever_input_ids=None,
                  retriever_position_ids=None,
                  retriever_attn_mask=None,
                  labels=None, tokentype_ids=None, inference_params=None):
        """GPT model construct"""

        if (position_ids is None or attention_mask is None) and self.eod:
            position_ids, attention_mask = self.eod_mask(tokens)

        if self.cp_size > 1:
            seq_dim = 1
            tokens = ops.chunk(tokens, self.cp_size, seq_dim)[self.cp_rank]
            position_ids = ops.chunk(position_ids, self.cp_size, seq_dim)[self.cp_rank]
            loss_mask = ops.chunk(loss_mask, self.cp_size, seq_dim)[self.cp_rank]
            if labels is not None:
                labels = ops.chunk(labels, self.cp_size, seq_dim)[self.cp_rank]

        lm_output = self.language_model(tokens,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        retriever_input_ids=retriever_input_ids,
                                        retriever_position_ids=retriever_position_ids,
                                        retriever_attn_mask=retriever_attn_mask,
                                        inference_params=inference_params)

        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            return post_language_model_processing(self.parallel_lm_logits,
                                                  self.loss,
                                                  lm_output,
                                                  labels,
                                                  logit_weights,
                                                  self.parallel_output,
                                                  self.fp16_lm_cross_entropy,
                                                  loss_mask)

        return lm_output
```

当`post_process`为`True`时，需要对语言模型的输出`lm_output`进行后处理，输出损失和预测结果。将如下代码添加到上述代码块中。

```python
import mindspore.common.dtype as mstype

def post_language_model_processing(parallel_lm_logits, loss_fn, lm_output, labels, logit_weights,
                                   parallel_output, fp16_lm_cross_entropy, loss_mask):
    """ gpt model post process forward """
    # head层计算
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)

    if labels is None:
        # [s b h] -> [b s h]
        return output.swapaxes(0, 1).contiguous()

    # [s b] -> [b s]
    output = output.swapaxes(0, 1).contiguous()

    # loss计算
    if fp16_lm_cross_entropy:
        if output.dtype != mstype.float16:
            raise ValueError(f"When fp16_lm_cross_entropy=True, output should be float16, but got {output.dtype}")
        loss = loss_fn(output, labels, loss_mask)
    else:
        loss = loss_fn(output.float(), labels, loss_mask)

    return loss
```

### 动态图并行训练配置

动态图并行的配置项通过yaml文件来读取，与Megatron的使用方法一致。框架内部提供与Megatron参数的映射，所以可以从Megatron的配置平滑切换到mindspeed_ms。下面列出了目前已支持的参数映射。

模型配置（model_config）的映射内容包括：

| Megatron                            | MindSpeed-Core-MS                                | 功能说明                                        |
|-------------------------------------|--------------------------------------------------|---------------------------------------------|
| padded_vocab_size                   | model_config.vocab_size                          | 词汇表大小，无缺省                                   |
| hidden_size                         | model_config.hidden_size                         | 隐藏层维度的大小，无缺省                                |
| seq_length                          | model_config.seq_length                          | 序列长度，无缺省                                    |
| rotary_base                         | model_config.rotary_base                         | 旋转位置嵌入的基本周期，仅当使用旋转位置嵌入时有效，默认值：10000         |
| num_layers                          | model_config.num_layers                          | Transformer层数，无缺省                           |
| position_embedding_type             | model_config.position_embedding_type             | 位置编码类型，默认值：absolute                         |
| init_method_std                     | model_config.init_method_std                     | 零均值正态分布的标准差，默认值：0.01                        |
| normalization                       | model_config.normalization                       | 归一化方法，默认值：LayerNorm                         |
| norm_epsilon                        | model_config.norm_epsilon                        | LayerNorm操作的Epsilon值，默认值：1e-5               |
| group_query_attention               | model_config.group_query_attention               | 是否开启分组查询注意力（GQA），默认：False                   |
| num_attention_heads                 | model_config.num_attention_heads                 | 注意力头的数量，无缺省                                 |
| num_query_groups                    | model_config.num_query_groups                    | 分组查询注意力的查询组组数，默认值：32                        |
| attention_dropout                   | model_config.attention_dropout_rate              | 注意力机制中的丢弃率，默认值：0.0                          |
| ffn_hidden_size                     | model_config.ffn_hidden_size                     | 前馈网络的隐藏层大小，无缺省                              |
| hidden_dropout                      | model_config.hidden_dropout_rate                 | 隐藏层的丢弃率，默认值：0.0                             |
| attention_softmax_in_fp32           | model_config.attention_softmax_in_fp32           | 注意力机制中softmax操作是否使用FP32精度，默认值：True          |
| use_flash_attn                      | model_config.use_flash_attention                 | 是否使用Flash Attention，默认值：False               |
| untie_embeddings_and_output_weights | model_config.untie_embeddings_and_output_weights | 用于确定是否解开嵌入层和输出层的权重绑定，默认值：False              |
| transformer_impl                    | model_config.transformer_impl                    | 用于指定Transformer的具体实现方式，目前尚未实现其他方式，默认值：local |
| recompute_granularity               | model_config.recompute_granularity               | 重计算的粒度，默认值：None                             |
| recompute_method                    | model_config.recompute_method                    | 重计算的方法，默认值：None                             |
| recompute_num_layers                | model_config.recompute_num_layers                | 重计算的层数，默认值：None                             |
| fp16_lm_cross_entropy               | model_config.fp16_lm_cross_entropy               | 模型交叉熵计算是否使用FP16精度，默认值：False                 |
| fp32_residual_connection            | model_config.fp32_residual_connection            | 是否使用 FP32 精度来处理残差连接，默认值：False               |
| add_qkv_bias                        | model_config.qkv_has_bias                        | 是否给查询（Q）、键（K）、值（V）向量添加偏置项，默认值：True          |
| add_dense_bias                      | model_config.out_proj_has_bias                   | 是否给输出投影层添加偏置项，默认值：True                      |
| add_bias_linear                     | model_config.add_bias_linear                     | 是否给所有线性层添加偏置，默认值：False                      |
| use_sandwich_norm                   | model_config.use_sandwich_norm                   | 是否使用Sandwich归一化方法，默认值：False                 |
| pre_tockens                         | model_config.fa_config.pre_tokens                | FA特有参数，用于稀疏计算的参数，表示向前计数多少个token             |
| next_tockens                        | model_config.fa_config.next_tokens               | FA特有参数，稀疏计算的参数，表示向后计数多少个token               |
| shape_order                         | model_config.fa_config.input_layout              | FA特有参数，用于指定输入数据的形状顺序等布局信息，无缺省               |
| attn_post_norm_scale                | model_config.attn_post_norm_scale                | 注意力机制后归一化的缩放因子，默认值：1.0                      |
| ffn_post_norm_scale                 | model_config.ffn_post_norm_scale                 | 前馈网络后归一化的缩放因子，默认值：1.0                       |
| params_dtype                        | model_config.params_dtype                        | 参数的数据类型，默认值：float32                         |
| compute_dtype                       | model_config.compute_dtype                       | 计算过程中使用的数据类型，默认值：float16                    |
| embedding_dtype                     | model_config.embedding_init_dtype                | 嵌入层初始化的数据类型，默认值：float32                     |
| use_fused_swiglu                    | model_config.apply_swiglu_fusion                 | 是否使用融合的Swiglu操作，默认值：False                   |
| apply_rope_fusion                   | model_config.apply_rope_fusion                   | 是否应用旋转位置嵌入（RoPE）融合技术，默认值：False              |
| hidden_act                          | model_config.hidden_act                          | 隐藏层的激活函数类型，默认值：gelu                         |

训练配置（training_config）的映射内容包括：

| Megatron                             | MindSpeed-Core-MS                                    | 功能说明                                                    |
|--------------------------------------|------------------------------------------------------|---------------------------------------------------------|
| seed                                 | training_config.seed                                 | 随机种子，默认值：None                                           |
| log_interval                         | training_config.log_interval                         | 日志记录间隔，指定每多少迭代后记录一次训练日志，默认值：None                        |
| train_iters                          | training_config.training_iters                       | 训练迭代次数，默认值：0                                            |
| save_interval                        | training_config.save_interval                        | 模型保存间隔，默认值：None                                         |
| eval_interval                        | training_config.eval_interval                        | 模型验证间隔，默认值：None                                         |
| accumulate_allreduce_grads_in_fp32   | training_config.accumulate_allreduce_grads_in_fp32   | 是否在 FP32 精度下累积和进行梯度归约操作，默认值：False                       |
| clip_grad                            | optimizer_config.clip_grad                           | 梯度裁剪相关配置，以字典形式传入裁剪参数，默认值：0.0                            |
| bf16                                 | training_config.bf16                                 | 是否启用BF16数据格式进行计算，默认值：False                              |
| fp16                                 | training_config.fp16                                 | 是否启用FP16数据格式进行计算，默认值：False                              |
| loss_scale                           | training_config.loss_scale                           | 初始的损失缩放比例值，默认值：None                                     |
| initial_loss_scale                   | training_config.loss_scale_value                     | 动态损失缩放的初始值，默认值：None                                     |
| loss_scale_window                    | training_config.loss_scale_window                    | 动态损失缩放的窗口大小，默认值：None                                    |
| hysteresis                           | training_config.loss_scale_factor                    | 动态损失缩放的因子，默认值：None                                      |
| use_distributed_optimizer            | training_config.use_distributed_optimizer            | 是否使用分布式优化器，默认值：False                                    |
| resume_training                      | training_config.resume_training                      | 是否恢复之前中断的训练，默认值：False                                   |
| resume_crc_check                     | training_config.crc_check                            | 保存/加载ckpy时是否进行循环冗余校验（CRC）检查，默认值：False                   |
| load                                 | training_config.load_checkpoint                      | 加载ckpy的路径，默认值：''                                        |
| save                                 | training_config.output_dir                           | 保存ckpt、日志等的输出目录，默认值：'./output'                          |
| ckpt_prefix                          | training_config.prefix                               | ckpt保存的前缀，默认值：'network'                                 |
| ckpt_format                          | training_config.ckpt_format                          | ckpt保存格式，默认值：'ckpt'                                     |
| keep_checkpoint_max                  | training_config.keep_checkpoint_max                  | 最多保留的ckpt数量，默认值：5                                       |
| wrap_with_ddp                        | training_config.wrap_with_ddp                        | 是否使用分布式数据并行（DDP）包装模型，默认值：False                          |
| bucket_size                          | training_config.bucket_size                          | 用于在`overlap_grad_reduce=True`时将缓冲区划分成桶的桶大小，默认值：None     |
| enable_mem_align                     | training_config.enable_mem_align                     | 是否开启内存对齐，默认值：False                                      |
| overlap_grad_reduce                  | training_config.overlap_grad_reduce                  | 是否启用梯度计算与同步通信重叠（使用分布式数据并行时），默认值：False                   |
| delay_grad_reduce                    | training_config.delay_grad_reduce                    | 是否延迟除第一个管道并行（PP）阶段外的梯度归约，默认值：False                      |
| no_load_optim                        | training_config.no_load_optim                        | 恢复训练时是否加载优化器状态，默认值：False                                |
| no_load_rng                          | training_config.no_load_rng                          | 恢复训练时是否加载随机数生成器（RNG）状态，默认值：True                         |
| new_dataset                          | training_config.new_dataset                          | 恢复训练时是否使用新数据集，默认值：False                                 |
| profile                              | training_config.profile                              | 是否开启性能分析功能，默认值：False                                    |
| profile_save_path                    | training_config.profile_save_path                    | 性能分析文件的保存路径，默认值：'./{output_dir}/profile'                |
| profile_step_start                   | training_config.profile_step_start                   | 性能分析开始的步骤，默认值：1                                         |
| profile_step_end                     | training_config.profile_step_end                     | 性能分析结束的步骤，默认值：5                                         |
| profile_level                        | training_config.profile_level                        | 性能分析的详细程度级别，可选"level0", "level1", "level2"，默认值："level0" |
| profile_with_stack                   | training_config.profile_with_stack                   | 是否在性能分析时包含调用栈信息，默认值：False                               |
| profile_memory                       | training_config.profile_memory                       | 是否分析内存相关情况，默认值：False                                    |
| profile_framework                    | training_config.profile_framework                    | 是否分析深度学习框架相关性能情况，默认值：False                              |
| profile_communication                | training_config.profile_communication                | 是否分析通信相关性能情况，比如分布式训练中的节点间通信，默认值：False                   |
| profile_parallel_strategy            | training_config.profile_parallel_strategy            | 是否分析并行策略相关性能情况，比如数据并行、模型并行等策略，默认值：False                 |
| profile_aicore_metrics               | training_config.profile_aicore_metrics               | 是否分析AI核心相关性能指标，默认值：0                                    |
| profile_l2_cache                     | training_config.profile_l2_cache                     | 是否分析二级缓存相关性能情况，默认值：False                                |
| profile_hbm_ddr                      | training_config.profile_hbm_ddr                      | 是否分析高带宽内存（HBM）和双倍数据速率（DDR）相关性能情况，默认值：False              |
| profile_pcie                         | training_config.profile_pcie                         | 是否分析PCIe（周边元件扩展接口）相关性能情况，默认值：False                      |
| profile_data_process                 | training_config.profile_data_process                 | 是否分析数据处理相关性能情况，比如数据加载、预处理等环节，默认值：False                  |
| profile_data_simplification          | training_config.profile_data_simplification          | 是否分析数据简化相关性能情况，默认值：False                                |
| profile_op_time                      | training_config.profile_op_time                      | 是否分析操作执行时间相关性能情况，默认值：True                               |
| profile_offline_analyse              | training_config.profile_offline_analyse              | 是否进行离线性能分析，默认值：False                                    |
| profile_dynamic_profiler_config_path | training_config.profile_dynamic_profiler_config_path | 动态性能分析器配置文件的路径，默认值：""                                   |

数据集配置（dataset_config）的映射内容包括：

| Megatron             | MindSpeed-Core-MS                   | 功能说明                     |
|----------------------|-------------------------------------|--------------------------|
| micro_batch_size     | dataset_config.batch_size           | 训练和评估时的批次大小，默认值：1        |
| micro_batch_num      | dataset_config.micro_batch_num      | 微批次大小，默认值：1              |
| reset_attention_mask | dataset_config.reset_attention_mask | 用于控制是否重置注意力掩码，当前未使用      |
| reset_position_ids   | dataset_config.reset_position_ids   | 用于控制是否重置位置标识，当前未使用       |
| eod_mask_loss        | dataset_config.eod_mask_loss        | eod掩码损失，当前未使用            |
| pad_token_id         | dataset_config.pad_token_id         | 填充标记索引，当前未使用             |
| eos_token_id         | dataset_config.eos_token_id         | 结束符的token id，默认值：0       |
| drop_remainder       | dataset_config.drop_remainder       | 是否丢弃不能被批次整除的数据集，默认值：True |

优化器配置（optimizer_config）的映射内容包括：

| Megatron                           | MindSpeed-Core-MS                                   | 功能说明                          |
|------------------------------------|-----------------------------------------------------|-------------------------------|
| optimizer                          | optimizer_config.optimizer_type                     | 指定优化器类型，默认值："AdamWeightDecay" |
| adam_beta1                         | optimizer_config.betas                              | Adam优化器中控制一阶矩估计指数衰减率的参数       |
| adam_beta2                         | optimizer_config.betas                              | Adam优化器中控制二阶矩估计指数衰减率的参数       |
| adam_eps                           | optimizer_config.eps                                | Adam优化器中用于数值稳定性的小常数           |
| lr_decay_style                     | optimizer_config.lr_decay_style                     | 学习率衰减方式，默认值：'constant'        |
| lr                                 | optimizer_config.learning_rate                      | 初始学习率，决定参数更新步长，默认值：1e-3       |
| min_lr                             | optimizer_config.min_lr                             | 学习率衰减的最小值，默认值：0.0             |
| lr_warmup_iters                    | optimizer_config.lr_warmup_iters                    | 训练开始时学习率逐步增加的迭代次数，默认值：0       |
| lr_decay_iters                     | optimizer_config.lr_decay_iters                     | 学习率开始衰减的迭代次数                  |
| use_checkpoint_opt_param_scheduler | optimizer_config.use_checkpoint_opt_param_scheduler | 是否使用检查点优化参数调度器，默认值：True       |
| override_opt_param_scheduler       | optimizer_config.override_opt_param_scheduler       | 是否覆盖已有参数调度器设置，默认值：False       |
| weight_decay                       | optimizer_config.weight_decay                       | 对模型权重施加正则化避免过拟合，默认值：0.0       |
| overlap_param_gather               | optimizer_config.overlap_param_gather               | 是否启用特定计算与参数收集通信重叠功能，默认值：False |
| start_weight_decay                 | optimizer_config.start_weight_decay                 | 权重衰减开始时的系数，默认值：0.0            |
| end_weight_decay                   | optimizer_config.end_weight_decay                   | 权重衰减结束时的系数，默认值：0.0            |
| weight_decay_incr_style            | optimizer_config.weight_decay_incr_style            | 权重衰减系数增加方式，默认值：'constant'     |

并行配置（parallel_config）的映射内容包括：

| Megatron                              | MindSpeed-Core-MS                                     | 功能说明                              |
|---------------------------------------|-------------------------------------------------------|-----------------------------------|
| tensor_model_parallel_size            | parallel_config.tensor_model_parallel_size            | 张量模型并行大小，默认值：1                    |
| context_parallel_size                 | parallel_config.context_parallel_size                 | 上下文并行（CP）大小，默认值：1                 |
| context_parallel_algo                 | parallel_config.context_parallel_algo                 | 上下文并行（CP）算法，默认值："ulysses_cp_algo" |
| ulysses_degree_in_cp                  | parallel_config.ulysses_degree_in_cp                  | 开启CP时，ulysses并行度，默认值：None         |
| expert_model_parallel_size            | parallel_config.expert_model_parallel_size            | 专家模型并行大小，默认值：1                    |
| virtual_pipeline_model_parallel_size  | parallel_config.virtual_pipeline_model_parallel_size  | 虚拟流水线并行（VPP）的大小，默认值：None          |
| num_layers_per_virtual_pipeline_stage | parallel_config.num_layers_per_virtual_pipeline_stage | 每个VPP的层数，无缺省                      |
| sequence_parallel                     | parallel_config.sequence_parallel                     | 是否启用序列并行，默认值：False                |
| pipeline_model_parallel_size          | parallel_config.pipeline_model_parallel_size          | 流水线并行（PP）大小，默认值：1                 |
| num_layer_list                        | parallel_config.num_layer_list                        | 用于用户自定义PP切分的形状，无缺省                |
| recompute                             | parallel_config.recompute                             | （按层）完全重计算，用于降低训练时内存占用，默认值：None    |
| select_recompute                      | parallel_config.select_recompute                      | （按算子）选择重计算，默认值：None               |
| select_comm_recompute                 | parallel_config.select_comm_recompute                 | （按算子）选择通信重计算，默认值：None             |

接下来简单介绍一下大模型训练需要的基本配置：

#### 配置训练参数（training_config）

```yaml
train_config:
  ms_training_config:
    resume_training: False  # 是否启用断点续训
    wrap_with_ddp: True     # 是否使用分布式数据分桶功能
    bucket_size: 200000000  # 分桶大小
  recompute_config:
    select_comm_recompute:  # "[[4,4,4,4],[4,4,4,4],[4,4,4,4],[4,4,4,2]]"  # 依赖并行配置里的layer_list，不开虚拟流水线时注释掉
```

#### 配置并行模式（accelerate_config）

```yaml
accelerate_config:
  parallelization:
    tensor_parallel: 8                                          # 模型并行大小
    pipeline:                                                   # 流水线并行配置
      num_stages: 4
      layer_list: # "[[4,4,4,4],[4,4,4,4],[4,4,4,4],[4,4,4,2]]" # 用户指定模型分层形状，不指定时自动切分，一般无需指定。对应的pp为[16,16,16,14]
      micro_batch_size: 1
      noop_layers:
      num_layers_per_vpp: # 4                                   # 虚拟流水线并行每个流水线的层数，不开VPP时注释掉
      virtual_pipeline_model_parallel_size: # 4                 # 虚拟流水线并行大小，不开VPP时注释掉。value = num_layers / pp / num_layers_per_vpp
    use_sequence_parallel: True                                 # 是否开启序列并行
    use_distributed_optimizer: True                             # 是否启用分布式优化器
  precision:                                                    # 网络训练的精度设置
    param_init_dtype: float32
    compute_dtype: bfloat16
    softmax_compute_dtype: float32
    logits_compute_dtype: float32
    grad_accumulation_dtype: float32
    residual-connection: bfloat16
    fused_operators: flash_attention, rope, rmsnorm, swiglu
    nofused_operators: masked_softmax, gradient_accumulation
```

#### 配置模型参数（model_config）

```yaml
model_config:
  llm:
    tokenizer:
      padded_size: 165664
      divided_by: 16
      use_fast: True
    embedding:
      hidden_size: 6144
      init_std: 0.5
    position:
      type: 'rope'
      max_size: 4096
      rope_base: 10000.0
    blocks:
      num_layers: 62
      init_std: 0.00884
      sandwich_norm: True
      layernorm:
        norm_type: 'RMSNorm'
        epsilon: 1e-5
      attention:
        attention_type: 'GQA'
        head_num: 48
        kv_group_size: 8
        qkv_bias: False
        projection_bias: False
        dropout_rate: 0.0
        post_norm_scale: 0.0447
      ffn:
        hidden_size: 27648
        activation: 'swiglu'
        dropout_rate: 0.0
        mlp_bias: False
        post_norm_scale: 0.0825
    head:
      shared_embedding: False
```

### 预备训练

在pretrain_gpt.py里对传入的yaml配置文件进行解析，初始化并行环境和数据。

```python
# 初始化megatron参数
initialize_megatron(extra_args_provider=_add_pangu_args)
args = get_args()
# 根据megatron参数映射到mindspeed_ms参数
all_config = init_configs_from_args(args)
training_config = all_config.training_config
# 设置环境变量，指定Ascend后端和动态图模式
ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE)
# 设置并行配置，初始化组网
set_parallel_context(all_config.parallel_config)
# 设置随机种子
set_seed(all_config.training_config.seed)
# 构建数据集
train_ds, _, _ = train_valid_test_datasets_provider(get_train_valid_test_num_samples())
train_dataloader = build_pretraining_data_loader(train_ds, 0)
```

### 创建网络对象

从模型库获取GPT模型，根据配置文件创建网络模型对象。

```python
from mindspeed_ms.legacy.model.gpt_model import GPTModel

def model_provider_func(pre_process=True, post_process=True):
    """get gpt model"""
    network_with_loss = GPTModel(
        all_config.model_config,
        pre_process=pre_process,
        post_process=post_process
    )

    return network_with_loss
```

### 权重加载

动态图并行中训练包括两种模式，第一种是从头训练，参数可以选择初始化参数或者加载权重；第二种是断点续训，参数通过加载已训练参数来初始化，已训练参数包括两种：第一种是自身模型训练后的参数，可以直接加载；第二种是将Megatron模型训练后的参数，进行转换后加载，可参考[权重转换指南](https://gitee.com/ascend/MindSpeed-Core-MS/blob/r0.1.0/docs/source_zh_cn/docs/source_zh_cn/usage/convert_tool_guide.md)。

### 执行训练

构建完网络，并加载好权重后，可以开始执行训练。`pretrain`接口相关说明可以参考[接口说明文档](https://gitee.com/ascend/MindSpeed-Core-MS/blob/r0.1.0/docs/api/api_python/mindspeed_ms.training.training.pretrain.rst)。

```python
from mindspeed_ms.training.training import pretrain

pretrain(
    train_valid_test_datasets_provider=None,
    model_provider_func=model_provider_func,
    model_type=None,
    forward_step_func=None,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    all_config=all_config,
    train_data_loader=train_dataloader,
    get_batch_func=get_batch,
    args_defaults={},
)
```

### 运行训练脚本

```bash
bash pretrain_gpt.sh
```

训练脚本`pretrain_gpt.sh`解析如下：

#### 设置环境变量

`HCCL_BUFFSIZE=200`设置两个NPU之间共享数据的缓存区大小为200M；`HCCL_EXEC_TIMEOUT=600`设置设备间执行时同步的等待时间为10分钟。`ASCEND_RT_VISIBLE_DEVICES`指定了可见的设备编号，这里设置为8卡。

```bash
export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=600
export ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
```

#### 以msrun模式执行训练脚本

```bash
msrun --worker_num 8 --local_worker_num=8 --master_port=8848 --log_dir=msrun_log --join=True --cluster_time_out=300 pretrain_gpt.py
```

#### 运行结果

接下来通过命令调用对应的脚本。

```bash
bash pretrain_gpt.sh
```

关于Loss部分结果保存在`msrun_log/worker_*.log`中，示例如下：

```text
...
INFO - Epoch: 0, Step: 13, Loss: 12.0177145004272461, Time: 15742.81 ms
INFO - Epoch: 0, Step: 14, Loss: 12.0177106857299805, Time: 15729.62 ms
INFO - Epoch: 0, Step: 15, Loss: 12.0177059173583984, Time: 15787.36 ms
INFO - Epoch: 0, Step: 16, Loss: 12.0177021026611328, Time: 15775.01 ms
INFO - Epoch: 0, Step: 17, Loss: 12.0176992416381836, Time: 15806.00 ms
INFO - Epoch: 0, Step: 18, Loss: 12.0176963806152344, Time: 15825.05 ms
...
```
