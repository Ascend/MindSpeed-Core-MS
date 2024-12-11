# 1. 合并 MindSpore 分布式优化器状态

本文档为 `combine_ckpt_dp_zero.py` 脚本使用文档。

`combine_ckpt_dp_zero.py` 脚本提供使能分布式优化器 `DistributedOptimizer` 时将分布式优化器保存下来的权重在数据并行通信域内进行合并的功能。

## 1.1. 分布式优化器开启时 ckpt 如何保存

当使用 MindSpore 框架默认配置训练时，会开启分布式优化器，该操作会将模型权重和分布式优化器状态展平存储在 Buffer 中，并根据 `bucket_size`​ 的大小进行分桶(Bucketing)。在保存 ckpt 时，分布式优化器持有的模型参数（fp32副本）及优化器状态会以优化器并行切分后桶的形式进行保存。例如，仅有1个 Buffer，该 Buffer 包含2个桶的场景，原始保存下来的 ckpt 文件 state_dict 将包含以下字段：

```python
state_dict.keys()
# ['buffer_0_bucket_0', 'exp_avg.buffer_0_bucket_0', 'exp_avg_sq.buffer_0_bucket_0', 'buffer_0_bucket_1', 'exp_avg.buffer_0_bucket_1', 'exp_avg_sq.buffer_0_bucket_1'， 'state_step', ...]
```

在相同的并行策略下，该 ckpt 能够被直接加载到相同的模型中进行续训。

当集群规模变化或需要改变训练的并行策略时，需要使用 `combine_ckpt_dp_zero.py`​ 脚本将上述形式的 ckpt 转换为以模型参数名称为 key，以模型参数真实 shape（非展平形式）为 value 的 ckpt 形式。

## 1.2. 合并分布式优化器

在使能分布式优化器进行训练时，在配置文件指定的输出目录 `output_dir`​ 下会生成 `opt_shard_info`​ 文件夹，并为每个 rank 创建子文件夹。在 `dp_rank = 0`​ 的设备对应的文件夹下，会生成 `*.json`​ 文件，该文件中记录了当前数据并行通信组内各参数在对应 Buffer 中的位置信息及原始 Shape 及数据类型等信息。以 `data_parallel_size = 2`​、`tensor_model_parallel_size = 2`​ 为例，输出目录结构将如下所示：

```bash
output_dir/
    ├── opt_shard_info
    │   ├── rank_0
    │   │    └── dist_opt_shard_info_rank_0-0_0.json #（仅有dp_rank==0的rank文件夹下存在该json文件）
    │   ├── rank_1
    │   │    └── dist_opt_shard_info_rank_1-0_0.json #（仅有dp_rank==0的rank文件夹下存在该json文件）
    │   ├── rank_2
    │   └── rank_3
    ├── rank_0
    │   ├── network_rank_0-0_0.ckpt
    │   └── network_rank_0-0_1.ckpt
    ├── rank_1
    │   ├── network_rank_1-0_0.ckpt
    │   └── network_rank_1-0_1.ckpt
    ├── rank_2
    │   ├── network_rank_1-0_0.ckpt
    │   └── network_rank_1-0_1.ckpt
    ├── rank_3
    │   ├── network_rank_1-0_0.ckpt
    │   └── network_rank_1-0_1.ckpt
    └── strategy
```

利用 `dist_opt_shard_info_*.json`​ 文件中的信息就能实现 DP 域内的参数聚合。`combine_ckpt_dp_zero.py`​ 接受以下入参：

- strategy-dir：分布式优化器分桶信息 `opt_shard_info`​ 文件夹路径。默认值：`output/opt_shard_info`​。
- checkpoint-dir：权重文件存储路径，各卡权重保存在该路径下 `rank_*`​ 文件夹下。默认值：`output`​。
- output-dir：合并后权重文件的保存路径，各卡权重保存在该路径下 `rank_*`​ 文件夹下。默认值：`output`​。
- src-format：原始权重文件格式，支持 [`ckpt`​, `safetensors`​]。默认值：`ckpt`​。
- dst-format: 合并权重文件格式，支持 [`ckpt`​, `safetensors`​]。默认值：`ckpt`​。
- copy-to-all-dp-ranks：配置时，将合并后的权重拷贝到该dp域内所有rank的对应文件夹下。默认不配置，仅在dp_rank == 0的文件夹下生成合并后的权重文件。
- max-proccess-limit：最大多进程并行进程数。默认值：`8`​。
- rank-section：指定处理的rank_id范围，默认值：`[None,None]`​。例，`--rank-section 2 6`​将处理全局rank_id为2、3、4及5的文件夹下的权重。该配置一般用于需要在多个节点上分别启动进程进行权重合并的场景。

默认使用各rank权重文件夹下最后一个权重进行聚合。执行完成后，将在各rank的权重保存路径下生成以 `*_dp_merged.ckpt`​ 为后缀的合并后权重。

**注**：`max_process_limit`​的设置需要考虑 host 内存，进程数过多可能导致 host 侧内存不足而报错。

### 1.2.1. 合并分布式优化器使用示例

```bash
export PYTHONPATH=/path/to/MindSpeed-Core-MS:$PYTHONPATH
python mindspeed_ms/tools/converter/combine_ckpt_dp_zero.py \
    --strategy-dir /path/to/ms_ckpt/opt_shard_info/ \
    --checkpoint-dir /path/to/ms_ckpt/ \
    --output-dir /path/to/ms_ckpt_merge \
    --max-proccess-limit 8 \
    > combine_ckpt_dp_zero.log 2>&1 &
    # --copy-to-all-dp-ranks \
tail -f combine_ckpt_dp_zero.log
```

以 dp_size = 2, pp_size = 2 为例，转换后的目录结构（未使能 `--copy-to-all-dp-ranks`），仅保存 dp_rank = 0 的 ckpt：

```bash
ms_layer1_dp2tp2pp1_hidden_128_ffn_512_merge
├── rank_0
│   └── network_rank_0-0_2_dp_merged.ckpt
└── rank_1
    └── network_rank_1-0_2_dp_merged.ckpt
```

以 dp_size = 2, pp_size = 2 为例，转换后的目录结构（使能 `--copy-to-all-dp-ranks`），保存所有 dp_rank 的 ckpt：

```bash
ms_layer1_dp2tp2pp1_hidden_128_ffn_512_merge_all_dp/
├── rank_0
│   └── network_rank_0-0_2_dp_merged.ckpt
├── rank_1
│   └── network_rank_1-0_2_dp_merged.ckpt
├── rank_2
│   └── network_rank_2-0_2_dp_merged.ckpt
└── rank_3
    └── network_rank_3-0_2_dp_merged.ckpt
```

以 dp_size = 2, pp_size = 2 为例，转换后的 ckpt 包含键值，key 变得有实际意义，包含模型权重和优化器参数：

```python
>>> param_dict1 = ms.load_checkpoint("ms_layer1_dp2tp2pp1_hidden_128_ffn_512_merge/rank_0/network_rank_0-0_2_dp_merged.ckpt")
>>> for k, v in param_dict1.items():
...     print(f"{k:70s}, type is {type(v)}")
...
language_model.output_layer.weight                                    , type is <class 'abc.Parameter'>
exp_avg.language_model.output_layer.weight                            , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.output_layer.weight                         , type is <class 'abc.Parameter'>
language_model.encoder.final_norm.weight                              , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.final_norm.weight                      , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.final_norm.weight                   , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.mlp.projection.weight                 , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.mlp.projection.weight         , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.mlp.projection.weight      , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.mlp.mapping.weight                    , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.mlp.mapping.weight            , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.mlp.mapping.weight         , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.ffn_post_norm.weight                  , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.ffn_post_norm.weight          , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.ffn_post_norm.weight       , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.post_attention_norm.weight            , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.post_attention_norm.weight    , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.post_attention_norm.weight , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.attention.out_proj.weight             , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.attention.out_proj.weight     , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.attention.out_proj.weight  , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.attention.qkv_proj.weight             , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.attention.qkv_proj.weight     , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.attention.qkv_proj.weight  , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.attn_post_norm.weight                 , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.attn_post_norm.weight         , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.attn_post_norm.weight      , type is <class 'abc.Parameter'>
language_model.encoder.layers.0.input_norm.weight                     , type is <class 'abc.Parameter'>
exp_avg.language_model.encoder.layers.0.input_norm.weight             , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.encoder.layers.0.input_norm.weight          , type is <class 'abc.Parameter'>
language_model.embedding.word_embeddings.weight                       , type is <class 'abc.Parameter'>
exp_avg.language_model.embedding.word_embeddings.weight               , type is <class 'abc.Parameter'>
exp_avg_sq.language_model.embedding.word_embeddings.weight            , type is <class 'abc.Parameter'>
......
```

## 1.3. 使用合并后的ckpt进行续训及权重转换

当模型并行切分不变（`tensor_model_parallel_size`​、`pipeline_model_parallel_size`​ 及 `virtual_pipeline_model_parallel_size`​ 不变）的场景下，合并后的权重能够直接用于数据并行大小（`data_parallel_size`​）发生变化的场景。

当模型并行切分发生变化时，需要使用 MindSpore 框架 [mindspore.transform_checkpoints](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.transform_checkpoints.html?highlight=transform_checkpoints#mindspore.transform_checkpoints) ​接口进行模型权重的转换后再进行加载。
