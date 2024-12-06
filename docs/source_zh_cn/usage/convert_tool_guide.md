# 1. 转换指南

本指南包含两个部分：

1. 将 MindSpore Checkpoint 转换为 Megatron Checkpoint；
2. 将 Megatron Checkpoint 转换为 MindSpore Checkpoint；
3. 合并 MindSpore 分布式优化器状态。

可以实现 Megatron Checkpoint ⇆ MindSpore Checkpoint 自由转换，并且转换时支持两种转换模式：

1. 仅转换模型权重；
2. 转换模型和优化器状态。

**本指南仅适用于 Megatron 生成的 Checkpoint 和 MindSpeed-Core-MS Checkpoint 互相转换。**

# 2. Megatron Checkpoint 转为 MindSpore Checkpoint

## 2.1. 获取 param_map

在使用 Megatron 框架时，框架会按照 bucket 形式保存分布式优化器状态，此时会丢失参数的排布信息。因此需要在 Megatron 框架中插入代码，在保存分布式优化器时将该信息一同保存，也就是 param_map。此信息为 Megatron 优化器状态中，桶具体以何种排布进行数据构造。

### 2.1.1. 具体获取步骤

1. 将 patch 文件复制到 Megatron 框架根目录

```bash
[ma-user Megatron-LM]$cp /path/to/MindSpeed-Core-MS/mindspeed_ms/tools/converter/patch/param_map.patch .
```

注：该 patch 适用于 Megatron-LM 框架 core_r0.6.0 分支，commit cac60ce4，Thu Apr 18 15:07:52，其他分支正在适配中，您也可以自行适配。

2. 应用 patch

```bash
git apply param_map.patch
```

3. 正常执行 Megatron 训练脚本，并至少执行 1 次正反向训练后，保存训练 Checkpoint。`param_map` 文件会被保存在 `--save` 目录下的 `param_map` 文件夹中。

在 Megatron Checkpoint 转 MindSpore Checkpoint 以及 MindSpore Checkpoint 转 Megatron Checkpoint 时均依赖此信息。

## 2.2. pt2ckpt 使用示例

脚本：mindspeed_ms/tools/converter/convert_pt2ckpt.py

​`python mindspeed_ms/tools/converter/convert_pt2ckpt.py -h`​

```bash
usage: convert_pt2ckpt.py [-h] --megatron-path MEGATRON_PATH --ms-path MS_PATH [--param-map-path PARAM_MAP_PATH] [--pp-size PP_SIZE] [--tp-size TP_SIZE] [--dp-size DP_SIZE]
                          [--cp-size CP_SIZE] [--parallel-init-order PARALLEL_INIT_ORDER] [--vpp-size VPP_SIZE] [--stage-list STAGE_LIST [STAGE_LIST ...]] [--num-layers NUM_LAYERS]
                          [--generate-all-dp-cp] [--convert-param-only] [--file-prefix FILE_PREFIX] [--format {ckpt,safetensors}]
                          [--dist-opt-content DIST_OPT_CONTENT [DIST_OPT_CONTENT ...]] [--debug] [--multiprocess-off] [--process-limit PROCESS_LIMIT] [--process_timeout PROCESS_TIMEOUT]

Pt-to-ckpt conversion Arguments

optional arguments:
  -h, --help            show this help message and exit

File Path/Location:
  --megatron-path MEGATRON_PATH
                        Path to megatron pt files.
  --ms-path MS_PATH     Path for saving Mindspore Ckpt.
  --param-map-path PARAM_MAP_PATH
                        Path to param_map files.

distributed:
  --pp-size PP_SIZE     Degree of pipeline model parallelism.
  --tp-size TP_SIZE     Degree of tensor model parallelism.
  --dp-size DP_SIZE     Degree of data model parallelism.
  --cp-size CP_SIZE     Degree of context model parallelism.
  --parallel-init-order PARALLEL_INIT_ORDER
                        expected parallel initialized order, only support tp-cp-ep-dp-pp now
  --vpp-size VPP_SIZE   euqals to virtual_pipeline_model_parallel_size, vpp_size * num_layers_per_virtual_pipeline_stage == num_layers // pp_size
  --stage-list STAGE_LIST [STAGE_LIST ...]
                        Number of layers per pipeline stage
  --num-layers NUM_LAYERS
                        Number of layers in Megatron models

Advanced feature:
  --generate-all-dp-cp  Whether generate all dp ckpt, or just dp-cp-0.
  --convert-param-only  Convert `model_optim_rng.pt`, which only contains bf16 model parameter without optimizer; If you want to convert model with optimizer、optimizer scheduler and other
                        args, please do NOT use this argument.
  --file-prefix FILE_PREFIX
                        Mindspore checkpoint filename
  --format {ckpt,safetensors}
                        Mindspore checkpoint file format
  --dist-opt-content DIST_OPT_CONTENT [DIST_OPT_CONTENT ...]
                        Torch distributed file content. i.e exp_avg in Adam optimizer
  --debug               print debug info
  --multiprocess-off    Turn off multiprocess.
  --process-limit PROCESS_LIMIT
                        Max num of processes.
  --process_timeout PROCESS_TIMEOUT
                        Timeout for each process.
```

Megatron Checkpoint 转为 MindSpore Checkpoint 支持两种模式，对应转换 Megatron 框架存储的两种 Checkpoint 类型

- **仅转换模型权重**：对应 `model_optim_rng.pt` 文件，该文件中仅包含以 `bfloat16` 类型保存的模型权重，不含优化器状态。适用场景：
    - 权重用于推理
    - 预训练权重用于微调
- **转换模型和优化器状态**：`distrib_optim.pt` 文件，该文件中包含以 `float32` 类型保存的模型权重和分布式优化器状态。适用场景：
    - Megatron Checkpoint 保存后，加载到 MindSpore 上继续训练

您可以按需取用。

### 2.2.1. 仅转换模型权重

添加 `--convert-param-only` 参数，这种模式会读取 Megatron Checkpoint 目录底下的 `model_optim_rng.pt` 文件，该文件中仅包含以 `bfloat16` 类型保存的模型权重，不含优化器状态。因此，转换后的文件仅含模型权重，不含优化器状态。该转换模式下无需 `param_map` 信息。Megatron 的部分训练配置 args 将会被保存到 `--ms-path` 指定的目录下 `pt_meta` 文件夹中，该 args 在 MindSpore Checkpoint 转回 Megatron Checkpoint 时，需要回写入 Megatron Checkpoint 中。

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python mindspeed_ms/tools/converter/convert_pt2ckpt.py \
       --megatron-path /path/to/torch_baseline_ckpts/iter_0001000/ \
       --param-map-path /path/to/torch_baseline_ckpts/param_map \
       --ms-path /path/to/ms_baseline_ckpts/ \
       --num-layers 64 \
       --pp-size 4 \
       --dp-size 1 \
       --vpp-size 4 \
       --tp-size 8 \
       --process-limit 8 \
       --convert-param-only \
       --debug \
       > convert_ckpt.log 2>&1 &
```

### 2.2.2. 转换模型和优化器状态

这种模式会读取 Megatron Checkpoint 目录底下的 `distrib_optim.pt` 文件，该文件中包含以 `float32` 类型保存的模型权重和分布式优化器状态。转换后的文件含模型权重和优化器状态。该转换模式需要传入 `param_map` 路径，用于读取 `param_map` 信息。Megatron 的部分训练配置 args 和 `param_map` 路径下 `param_map*.json` 文件将会被保存到 `--ms-path` 指定的目录下 `pt_meta` 文件夹中。args 在 MindSpore Checkpoint 转回 Megatron Checkpoint 时，需要回写入 Megatron Checkpoint 中。

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python mindspeed_ms/tools/converter/convert_pt2ckpt.py \
       --megatron-path /path/to/torch_baseline_ckpts/iter_0001000/ \
       --param-map-path /path/to/torch_baseline_ckpts/param_map \
       --ms-path /path/to/ms_baseline_ckpts/ \
       --num-layers 64 \
       --pp-size 4 \
       --dp-size 1 \
       --vpp-size 4 \
       --tp-size 8 \
       --process-limit 8 \
       --debug \
       > convert_ckpt.log 2>&1 &
```

- 可以在生成的日志中搜索 `successfully`​, 源文件有几个文件夹，就应该有几个 `successfully`​，可以依此判断转换是否成功。
- 脚本默认以多进程模式执行，测试或 debug 问题建议添加 `--debug`​、 `--multiprocess-off`​

## 2.3. 默认目录结构

以 tp_size = 2, pp_size = 2 为例

### 2.3.1. Megatron Checkpoint 开启分布式优化器

```bash
pt_path
├── iter_0000001
│   ├── mp_rank_00_000                        # mp_rank_{tp_rank}_{pp_rank}
│   │   ├── distrib_optim.pt
│   │   └── model_optim_rng.pt
│   ├── mp_rank_00_001
│   │   ├── distrib_optim.pt
│   │   └── model_optim_rng.pt
│   ├── mp_rank_01_000
│   │   ├── distrib_optim.pt
│   │   └── model_optim_rng.pt
│   └── mp_rank_01_001
│       ├── distrib_optim.pt
│       └── model_optim_rng.pt
├── latest_checkpointed_iteration.txt
└── param_map                                 # args & param_map
    ├── param_map_buffer0_dp0tp0pp0vpp0.json  # param_map
    ├── param_map_buffer0_dp0tp0pp1vpp0.json
    ├── param_map_buffer0_dp0tp1pp0vpp0.json
    └── param_map_buffer0_dp0tp1pp1vpp0.json
```

- 目录 `mp_rank_00_000` 含义为 `mp_rank_{tp_rank}_{pp_rank}`

### 2.3.2. MindSpore Checkpoint

```bash
ms_path
├── rank_0
│   └── network.ckpt
├── rank_1
│   └── network.ckpt
├── rank_2
│   └── network.ckpt
├── rank_3
│   └── network.ckpt
└── pt_meta                                   # args & param_map
    ├── args_tp00_pp000.pt                    # Megatron args
    ├── args_tp00_pp001.pt
    ├── args_tp01_pp000.pt
    ├── args_tp01_pp001.pt
    ├── param_map_buffer0_dp0tp0pp0vpp0.json  # param_map
    ├── param_map_buffer0_dp0tp0pp1vpp0.json
    ├── param_map_buffer0_dp0tp1pp0vpp0.json
    └── param_map_buffer0_dp0tp1pp1vpp0.json
```

- 目录 `rank_x` 含义为 `global_rank`；
- 转换脚本默认执行在零冗余模式，即仅保存 `dp_rank = 0` 权重。在这种模式下转换出的 ckpt 可能会出现 `rank_x` 不连续的情况；
- 假如转换后，`dp_size` 发生变动，需要重新导出新的 `dp_size` 或 `cp_size` 的 ckpt，不同 `dp_size` 或 `cp_size` 之间的 ckpt 不适用。或者您可以利用以下公式手动对已有的 ckpt 进行更名使用。

    ```python
    global_rank = pp_rank * dp_size * cp_size * tp_size + tp_rank
    ```

    分别将新、旧的 `dp_size` 、 `cp_size` 填入公式中，即可得到新旧的 `global_rank` 映射关系。
- 对于多机、分布式存储环境，零冗余的所有 ckpt 应当被复制到每一个机器中才能被正确读取；
- 对于多机、集中式存储环境，零冗余的所有 ckpt 被复制到集中式存储中即可被所有机器读取。

# 3. MindSpore Checkpoint 转为 Megatron Checkpoint

## 3.1. 前置条件

1. param_map*.json：Megatron 训练时保存的 param_map。如何获取请参照 [获取 param_map](#21-获取-param_map)；
2. args_xx_xxx.pt：Megatron Checkpoint 转为 MindSpore Checkpoint 时，会将 Megatron 部分参数保存在 args_xx_xxx.pt 中，转换时需要将此参数回写到 Megatron Checkpoint 中，如何获取请参照 [Megatron Checkpoint 转为 MindSpore Checkpoint](#2-megatron-checkpoint-转为-mindspore-checkpoint)。
3. 前两者需保存在同一路径下，路径下存有 json 及 pt 文件，即 `--param-map-path` 指定的目录 `pt_meta` 目录树如下：

```bash
pt_meta
├── args_tp00_pp000.pt
├── args_tp00_pp001.pt
├── args_tp01_pp000.pt
├── args_tp01_pp001.pt
├── param_map_buffer0_dp0tp0pp0vpp0.json
├── param_map_buffer0_dp0tp0pp1vpp0.json
├── param_map_buffer0_dp0tp1pp0vpp0.json
└── param_map_buffer0_dp0tp1pp1vpp0.json
```

当使用 `pt2ckpt.py` 脚本将 Megatron Checkpoint 转为 MindSpore Checkpoint，且转换时给定了 `--param-map-path` 参数，输出目录 `ckpt_path/pt_meta` 即满足此要求，将 `--param-map-path` 指定为 `ckpt_path/pt_meta` 即可；

1. 使用 MindSpore 训练保存的权重，需要先使用 combine_ckpt_dp_zero.py 脚本进行参数合并（dp_size = 1 也需要执行此项操作），之后使用合并后的权重做 MindSpore 到 Megatron 的转换，combine_ckpt_dp_zero.py 脚本使用请参考 [合并 MindSpore 分布式优化器状态](#5-合并-mindspore-分布式优化器状态)。

## 3.2. ckpt2pt 使用示例

脚本：mindspeed_ms/tools/converter/convert_ckpt2pt.py

参数说明

```text
usage: convert_ckpt2pt.py [-h] --num-layers NUM_LAYERS --dp-size DP_SIZE --cp-size CP_SIZE --tp-size TP_SIZE --pp-size PP_SIZE --vpp-size VPP_SIZE
                          [--src-model-format SRC_MODEL_FORMAT] [--noop NOOP [NOOP ...]] --ms-path MS_PATH --param-map-path PARAM_MAP_PATH --megatron-path MEGATRON_PATH
                          [--convert-param-only] [--process-limit PROCESS_LIMIT] [--multiprocess-off] [--process_timeout PROCESS_TIMEOUT]

ckpt-to-pt conversion Arguments

optional arguments:
  -h, --help            show this help message and exit

distributed:
  --num-layers NUM_LAYERS
                        Number of layers in models
  --dp-size DP_SIZE     Degree of data model parallelism.
  --cp-size CP_SIZE     Degree of context model parallelism.
  --tp-size TP_SIZE     Degree of tensor model parallelism.
  --pp-size PP_SIZE     Degree of pipeline model parallelism.
  --vpp-size VPP_SIZE   Number of virtual pipeline per pipeline stage
  --src-model-format SRC_MODEL_FORMAT
                        Path to param_map files.
  --noop NOOP [NOOP ...]
                        Number of virtual pipeline per pipeline stage

File Path/Location:
  --ms-path MS_PATH     Path to MindSpore checkpoint files.
  --param-map-path PARAM_MAP_PATH
                        Path to param_map files.
  --megatron-path MEGATRON_PATH
                        Path to save Megatron checkpoint.
  --convert-param-only  Convert only the model parameter without optimizer params;
  --process-limit PROCESS_LIMIT
                        Max num of processes.
  --multiprocess-off    Turn off multiprocess.
  --process_timeout PROCESS_TIMEOUT
                        Timeout for each process.
```

MindSpore Checkpoint 转为 Megatron Checkpoint 支持两种模式，对应转换成 Megatron 框架存储的两种 Checkpoint 类型

- **仅转换模型权重**：转换成 `model_optim_rng.pt` 文件，该文件中仅包含以 `bfloat16` 类型保存的模型权重，不含优化器状态。适用场景：
    - 权重用于推理
    - 预训练权重用于微调
- **转换模型和优化器状态**：转换成 `distrib_optim.pt` 文件，该文件中包含以 `float32` 类型保存的模型权重和分布式优化器状态。适用场景：
    - MindSpore Checkpoint 保存后，加载到 Megatron 上继续训练

您可以按需取用。

### 3.2.1. 仅转换模型权重

转换时添加 `--convert-param-only` 参数，这种模式仅生成包含 `bfloat16` 类型的模型权重文件 `model_optim_rng.pt`。

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python mindspeed_ms/tools/converter/convert_ckpt2pt.py \
    --num-layers 8 \
    --dp-size 1 \
    --cp-size 1 \
    --tp-size 2 \
    --pp-size 4 \
    --vpp-size 2 \
    --noop 62 63 \
    --ms-path /home/ma-user/work/ckpt2pt/ms_ckpt \
    --convert-param-only \
    --param-map-path /home/ma-user/work/work/ckpt2pt/pt_meta/ \
    --megatron-path /home/ma-user/work/ckpt2pt/megatron_pt
```

### 3.2.2. 转换模型权重和分布式优化器状态

这种模式生成包含 `bfloat16` 类型的模型权重文件 `model_optim_rng.pt`，以及以 `float32` 桶形式存储的模型权重和分布式优化器状态文件 `distrib_optim.pt`。

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python mindspeed_ms/tools/converter/convert_ckpt2pt.py \
    --num-layers 8 \
    --dp-size 1 \
    --cp-size 1 \
    --tp-size 2 \
    --pp-size 4 \
    --vpp-size 2 \
    --noop 62 63 \
    --ms-path /home/ma-user/work/ckpt2pt/ms_ckpt \
    --param-map-path /home/ma-user/work/work/ckpt2pt/pt_meta/ \
    --megatron-path /home/ma-user/work/ckpt2pt/megatron_pt
```

- 脚本默认以多进程模式执行，测试或 debug 问题建议添加 `--multiprocess-off`​

# 4. 转换正确性验证

## 4.1. 转换模型权重和分布式优化器状态

对于转换模型权重和分布式优化器状态的 Checkpoint场景，验证转换后的正确性，可以将 Megatron Checkpoint `A` 转为 MindSpore Checkpoint `B`，再将 MindSpore Checkpoint `B` 转换为 Megatron Checkpoint `C`。此时，`B → C` 过程无需合并分布式优化器状态。

接下来，使用以下脚本：

```bash
python mindspeed_ms/tools/converter/md5sum.py origin/saved_cp2tp2pp2_0_0.pt refresh/saved_cp2tp2pp2_1_0.pt

```

输出：

```bash
>20:52:22  : -------------Comparing MD5Sum-------------
All files have the same md5sum value: 0e3fa0233ca695a9b8061c3716a8bc46
```

md5 一致，转换正确。

# 5. 合并 MindSpore 分布式优化器状态

## 5.1. 分布式优化器开启时 ckpt 如何保存

当使用 MindSpore 框架默认配置训练时，会开启分布式优化器，该操作会将模型权重和分布式优化器状态展平存储在 Buffer 中，并根据 `bucket_size`​ 的大小进行分桶(Bucketing)。在保存 ckpt 时，分布式优化器持有的模型参数（fp32副本）及优化器状态会以优化器并行切分后桶的形式进行保存。例如，仅有1个 Buffer，该 Buffer 包含2个桶的场景，原始保存下来的 ckpt 文件 state_dict 将包含以下字段：

```python
state_dict.keys()
# ['buffer_0_bucket_0', 'exp_avg.buffer_0_bucket_0', 'exp_avg_sq.buffer_0_bucket_0', 'buffer_0_bucket_1', 'exp_avg.buffer_0_bucket_1', 'exp_avg_sq.buffer_0_bucket_1'， 'state_step', ...]
```

在相同的并行策略下，该 ckpt 能够被直接加载到相同的模型中进行续训。

当集群规模变化或需要改变训练的并行策略时，需要使用 `combine_ckpt_dp_zero.py`​ 脚本将上述形式的 ckpt 转换为以模型参数名称为 key，以模型参数真实 shape（非展平形式）为 value 的 ckpt 形式。

## 5.2. 合并分布式优化器

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

## 5.3. 合并分布式优化器使用示例

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

## 5.4. 使用合并后的ckpt进行续训及权重转换

当模型并行切分不变（`tensor_model_parallel_size`​、`pipeline_model_parallel_size`​ 及 `virtual_pipeline_model_parallel_size`​ 不变）的场景下，合并后的权重能够直接用于数据并行大小（`data_parallel_size`​）发生变化的场景。

当模型并行切分发生变化时，需要使用 MindSpore 框架 [mindspore.transform_checkpoints](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.transform_checkpoints.html?highlight=transform_checkpoints#mindspore.transform_checkpoints) ​接口进行模型权重的转换后再进行加载。
