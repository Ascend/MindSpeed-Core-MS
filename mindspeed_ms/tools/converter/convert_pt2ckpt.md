# 1. Megatron Checkpoint 转为 MindSpore Checkpoint

## 1.1. 获取 param_map

在使用 Megatron 框架时，框架会按照 bucket 形式保存分布式优化器状态，此时会丢失参数的排布信息。因此需要在 Megatron 框架中插入代码，在保存分布式优化器时将该信息一同保存，也就是 param_map。此信息为 Megatron 优化器状态中，桶具体以何种排布进行数据构造。

### 1.1.1. 具体获取步骤

1. 将 patch 文件复制到 Megatron 框架根目录

```bash
[ma-user Megatron-LM]$cp MindSpeed-Core-MS/mindspeed_ms/tools/converter/patch/param_map.patch .
```

注：该 patch 适用于 Megatron-LM 框架 core_r0.6.0 分支，commit cac60ce4，Thu Apr 18 15:07:52，其他分支正在适配中，您也可以自行适配。

2. 应用 patch

```bash
git apply param_map.patch
```

3. 正常执行 Megatron 训练脚本，并至少执行 1 次正反向训练后，保存训练 Checkpoint。`param_map` 文件会被保存在 `--save` 目录下的 `param_map` 文件夹中。

在 Megatron Checkpoint 转 MindSpore Checkpoint 以及 MindSpore Checkpoint 转 Megatron Checkpoint 时均依赖此信息。

## 1.2. 使用示例

脚本：convert_pt2ckpt.py

​`python convert_pt2ckpt.py -h`​

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

### 1.2.1. 仅转换模型权重

添加 `--convert-param-only` 参数，同时不指定`--param-map-path`。这种模式会读取 Megatron Checkpoint 目录底下的 `model_optim_rng.pt` 文件，该文件中仅包含以 `bfloat16` 类型保存的模型权重，不含优化器状态。因此，转换后的文件仅含模型权重，不含优化器状态。该转换模式下无需 `param_map` 信息。Megatron 的部分训练配置 args 将会被保存到 `--ms-path` 指定的目录下 `pt_meta` 文件夹中，该 args 在 MindSpore Checkpoint 转回 Megatron Checkpoint 时，需要回写入 Megatron Checkpoint 中。

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python mindspeed_ms/tools/converter/convert_pt2ckpt.py \
       --megatron-path /path/to/torch_baseline_ckpts/iter_0001000/ \
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

### 1.2.2. 转换模型和优化器状态

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

## 1.3. 默认目录结构

以 tp_size = 2, pp_size = 2 为例

### 1.3.1. Megatron Checkpoint 开启分布式优化器

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

### 1.3.2. MindSpore Checkpoint

```bash
ms_path
├── iter_0000001
│   ├── mp_rank_00_000                        # mp_rank_{tp_rank}_{pp_rank}
│   │   ├── distrib_optim.ckpt
│   │   └── model_optim_rng.ckpt
│   ├── mp_rank_00_001
│   │   ├── distrib_optim.ckpt
│   │   └── model_optim_rng.ckpt
│   ├── mp_rank_01_000
│   │   ├── distrib_optim.ckpt
│   │   └── model_optim_rng.ckpt
│   └── mp_rank_01_001
│       ├── distrib_optim.ckpt
│       └── model_optim_rng.ckpt
├── latest_checkpointed_iteration.txt
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

- 目录结构与 Megatron 保持一致；
- 转换脚本默认执行在零冗余模式，即仅保存 `dp_rank = 0` 权重。在这种模式下转换出的 ckpt 可能会出现 `mp_rank_x` 不连续的情况；
- 对于多机、分布式存储环境，零冗余的所有 ckpt 应当被复制到每一个机器中才能被正确读取；
- 对于多机、集中式存储环境，零冗余的所有 ckpt 被复制到集中式存储中即可被所有机器读取。