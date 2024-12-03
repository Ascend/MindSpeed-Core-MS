# 动态图并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/usage/pretrain_gpt.md)

## 概述

本教程演示如何使用MindFormers动态图并行框架训练GPT模型，此框架支持张量并行、流水线并行、序列并行等并行场景，还有支持使用分布式优化器动态学习率等场景，帮助开发者快速、便捷地构建和训练基于动态图并行框架的GPT预训练模型。

## 操作实践

下面基于Ascend平台，进行GPT模型训练。

### 样例代码参考

目录结构如下：

```text
└─ gpt
    ├─ pretrain_gpt.py
    ├─ pretrain_gpt.sh
    └─ pretrain_gpt_38BV3.yaml
    ...
```

其中，`pretrain_gpt.py`是环境配置、模型对象创建及训练的脚本。`pretrain_gpt.sh`是启动执行脚本。`pretrain_gpt_38BV3.yaml`是配置项。

### 模型结构

GPT以`Transformer`模型为主要架构，网络结构主要围绕`Transformer`的基本构建块构建。

在模型中，初始化五个参数，`config`是模型配置项（在yaml文件的`model_config`中），`num_tokentypes`指定embedding的类型，`parallel_output`用来确认是否输出每一个并行Tensor的输出，`pre_process`和`post_process`分别指定是否为第一阶段和最后一阶段。

注意：数据集返回值要与模型定义的前向过程所需要的参数相对应。

```python
# 引入必需的模块
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.transformer.language_model import get_language_model
from mindformers.experimental.parallel_core.pynative.transformer import ParallelLMLogits
from mindformers.experimental.parallel_core.pynative.training.loss_func import VocabParallelCrossEntropy
from mindspeed_ms.training.loss_func import LossWithMask
from mindspeed_ms.legacy.model.eos_mask import EosMask

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

        # 获取eod mask数据
        self.eod = kwargs['eod'] if 'eod' in kwargs else None
        self.reset_position_ids = kwargs['reset_position_ids'] if 'reset_position_ids' in kwargs else False
        if self.eod:
            self.eod_mask = EosMask(self.batch_size, config.seq_length, self.eod, self.reset_position_ids)

        self.set_model_key()

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

    def set_model_key(self):
        """ set model key for differentiate PipelineCell process """
        self.model_key = "gpt_model"

    # GPT模型执行
    def construct(self, tokens, position_ids, attention_mask, loss_mask,
                  retriever_input_ids=None,
                  retriever_position_ids=None,
                  retriever_attn_mask=None,
                  labels=None, tokentype_ids=None, inference_params=None):
        """GPT model construct"""

        if (position_ids is None or attention_mask is None) and self.eod:
            position_ids, attention_mask = self.eod_mask(tokens)

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

当`post_process`为`True`时，需要对语言模型的输出`lm_output`进行后处理，输出损失和预测结果。

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

动态图并行的配置项通过yaml文件来读取，与Megatron的使用方法一致。框架内部提供与Megatron参数的映射，所以可以从Megatron的配置平滑切换到mindspeed_ms。

模型配置（model_config）的映射内容包括：

| Megatron | MindSpeed-Core-MS |
| -------- | -------- |
| padded_vocab_size | model_config.vocab_size |
| hidden_size | model_config.hidden_size |
| seq_length | model_config.seq_length |
| rotary_base | model_config.rotary_base |
| num_layers | model_config.num_layers |
| position_embedding_type | model_config.position_embedding_type |
| use_rotary_position_embeddings | model_config.use_rotary_embedding |
| init_method_std | model_config.init_method_std |
| normalization | model_config.normalization |
| norm_epsilon | model_config.norm_epsilon |
| group_query_attention | model_config.group_query_attention |
| num_attention_heads | model_config.num_attention_heads |
| num_query_groups | model_config.num_query_groups |
| attention_dropout | model_config.attention_dropout_rate |
| ffn_hidden_size | model_config.ffn_hidden_size |
| hidden_dropout | model_config.hidden_dropout_rate |
| attention_softmax_in_fp32 | model_config.attention_softmax_in_fp32 |
| use_flash_attn | model_config.use_flash_attention |
| untie_embeddings_and_output_weights | model_config.untie_embeddings_and_output_weights |
| transformer_impl | model_config.transformer_impl |
| recompute_granularity | model_config.recompute_granularity |
| recompute_method | model_config.recompute_method |
| recompute_num_layers | model_config.recompute_num_layers |
| fp16_lm_cross_entropy | model_config.fp16_lm_cross_entropy |
| fp32_residual_connection | model_config.fp32_residual_connection |
| add_qkv_bias | model_config.qkv_has_bias |
| add_dense_bias | model_config.out_proj_has_bias |
| add_bias_linear | model_config.add_bias_linear |
| use_sandwich_norm | model_config.use_sandwich_norm |
| pre_tockens | model_config.fa_config.pre_tokens |
| next_tockens | model_config.fa_config.next_tokens |
| shape_order | model_config.fa_config.input_layout |
| attn_post_norm_scale | model_config.attn_post_norm_scale |
| ffn_post_norm_scale | model_config.ffn_post_norm_scale |
| params_dtype | model_config.params_dtype |
| compute_dtype | model_config.compute_dtype |
| embedding_dtype | model_config.embedding_init_dtype |
| use_fused_swiglu | model_config.apply_swiglu_fusion |
| apply_rope_fusion | model_config.apply_rope_fusion |
| hidden_act | model_config.hidden_act |

训练配置（training_config）的映射内容包括：

| Megatron | MindSpeed-Core-MS |
| -------- | -------- |
| seed | training_config.seed |
| log_interval | training_config.log_interval |
| train_iters | training_config.training_iters |
| save_interval | training_config.save_interval |
| eval_interval | training_config.eval_interval |
| accumulate_allreduce_grads_in_fp32 | training_config.accumulate_allreduce_grads_in_fp32 |
| clip_grad | optimizer_config.clip_grad |
| bf16 | training_config.bf16 |
| fp16 | training_config.fp16 |
| loss_scale | training_config.loss_scale |
| initial_loss_scale | training_config.loss_scale_value |
| loss_scale_window | training_config.loss_scale_window |
| hysteresis | training_config.loss_scale_factor |
| use_distributed_optimizer | training_config.use_distributed_optimizer |
| resume_training | training_config.resume_training |
| resume_crc_check | training_config.crc_check |
| load | training_config.load_checkpoint |
| save | training_config.output_dir |
| ckpt_prefix | training_config.prefix |
| ckpt_format | training_config.ckpt_format |
| keep_checkpoint_max | training_config.keep_checkpoint_max |
| wrap_with_ddp | training_config.wrap_with_ddp |
| bucket_size | training_config.bucket_size |
| enable_mem_align | training_config.enable_mem_align |
| overlap_grad_reduce | training_config.overlap_grad_reduce |
| delay_grad_reduce | training_config.delay_grad_reduce |
| no_load_optim | training_config.no_load_optim |
| no_load_rng | training_config.no_load_rng |
| new_dataset | training_config.new_dataset |
| profile | training_config.profile |
| profile_save_path | training_config.profile_save_path |
| profile_step_start | training_config.profile_step_start |
| profile_step_end | training_config.profile_step_end |
| profile_level | training_config.profile_level |
| profile_with_stack | training_config.profile_with_stack |
| profile_memory | training_config.profile_memory |
| profile_framework | training_config.profile_framework |
| profile_communication | training_config.profile_communication |
| profile_parallel_strategy | training_config.profile_parallel_strategy |
| profile_aicore_metrics | training_config.profile_aicore_metrics |
| profile_l2_cache | training_config.profile_l2_cache |
| profile_hbm_ddr | training_config.profile_hbm_ddr |
| profile_pcie | training_config.profile_pcie |
| profile_data_process | training_config.profile_data_process |
| profile_data_simplification | training_config.profile_data_simplification |
| profile_op_time | training_config.profile_op_time |
| profile_offline_analyse | training_config.profile_offline_analyse |
| profile_dynamic_profiler_config_path | training_config.profile_dynamic_profiler_config_path |

数据集配置（dataset_config）的映射内容包括：

| Megatron | MindSpeed-Core-MS |
| -------- | -------- |
| reset_attention_mask | dataset_config.reset_attention_mask |
| reset_position_ids | dataset_config.reset_position_ids |
| eod_mask_loss | dataset_config.eod_mask_loss |
| pad_token_id | dataset_config.pad_token_id |
| eos_token_id | dataset_config.eos_token_id |
| drop_remainder | dataset_config.drop_remainder |
| micro_batch_size | dataset_config.batch_size |
| micro_batch_num | dataset_config.micro_batch_num |

优化器配置（optimizer_config）的映射内容包括：

| Megatron | MindSpeed-Core-MS |
| -------- | -------- |
| optimizer | optimizer_config.optimizer_type |
| adam_beta1 | optimizer_config.betas |
| adam_beta2 | optimizer_config.betas |
| adam_eps | optimizer_config.eps |
| lr_decay_style | optimizer_config.lr_decay_style |
| lr | optimizer_config.learning_rate |
| min_lr | optimizer_config.min_lr |
| lr_warmup_iters | optimizer_config.lr_warmup_iters |
| lr_decay_iters | optimizer_config.lr_decay_iters |
| use_checkpoint_opt_param_scheduler | optimizer_config.use_checkpoint_opt_param_scheduler |
| override_opt_param_scheduler | optimizer_config.override_opt_param_scheduler |
| weight_decay | optimizer_config.weight_decay |
| overlap_param_gather | optimizer_config.overlap_param_gather |
| start_weight_decay | optimizer_config.start_weight_decay |
| end_weight_decay | optimizer_config.end_weight_decay |
| weight_decay_incr_style | optimizer_config.weight_decay_incr_style |

并行配置（parallel_config）的映射内容包括：

| Megatron | MindSpeed-Core-MS |
| -------- | -------- |
| tensor_model_parallel_size | parallel_config.tensor_model_parallel_size |
| context_parallel_size | parallel_config.context_parallel_size |
| context_parallel_algo | parallel_config.context_parallel_algo |
| ulysses_degree_in_cp | parallel_config.ulysses_degree_in_cp |
| expert_model_parallel_size | parallel_config.expert_model_parallel_size |
| virtual_pipeline_model_parallel_size | parallel_config.virtual_pipeline_model_parallel_size |
| num_layers_per_virtual_pipeline_stage | parallel_config.num_layers_per_virtual_pipeline_stage |
| sequence_parallel | parallel_config.sequence_parallel |
| pipeline_model_parallel_size | parallel_config.pipeline_model_parallel_size |
| num_layer_list | parallel_config.num_layer_list |
| recompute | parallel_config.recompute |
| select_recompute | parallel_config.select_recompute |
| select_comm_recompute | parallel_config.select_comm_recompute |

接下来简单介绍一下大模型训练需要的基本配置：

#### 配置训练参数（training_config）

```yaml
train_config:
  ms_training_config:
    resume_training: False          # 是否启用断点续训
    wrap_with_ddp: True             # 是否使用分布式数据分桶功能
    bucket_size: 40000000           # 分桶大小
```

#### 配置并行模式（accelerate_config）

```yaml
accelerate_config:
  parallelization:
    tensor_parallel: 8                                          # 模型并行大小
    pipeline:                                                   # 流水线并行配置
      num_stages: 4
      layer_list: "[[4,4,4,4],[4,4,4,4],[4,4,4,4],[4,4,4,2]]"
      micro_batch_size: 1
      noop_layers:
      num_layers_per_vpp: 4                                     # 虚拟流水线并行配置
      virtual_pipeline_model_parallel_size: 4
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

构建完网络，并加载好权重后，可以开始执行训练。

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

`HCCL_BUFFSIZE=200`设置两个NPU之间共享数据的缓存区大小为200M；`HCCL_EXEC_TIMEOUT=600`设置设备间执行时同步的等待时间为10分钟。`ASCEND_RT_VISIBLE_DEVICES`指定了可见的设备编号，这里设置为设备`0`号卡。

```bash
export HCCL_BUFFSIZE=200
export HCCL_EXEC_TIMEOUT=600
export ASCEND_RT_VISIBLE_DEVICES='0'
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
