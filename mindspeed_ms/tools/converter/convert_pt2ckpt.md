# Megatron 权重转为 MindSpore 权重

## 1 获取 param_map

在使用 Megatron 框架时，框架会按照 bucket 形式保存分布式优化器权重，此时会丢失参数的排布信息。因此需要在 Megatron 框架中插入代码，
在保存分布式优化器时将该信息一同保存，也就是 param_map。此信息为 Megatron 优化器权重中，桶具体以何种排布进行数据构造。

### 1.1 具体获取步骤

1. 将 patch 文件复制到框架根目录

```bash
[ma-user Megatron-LM]$cp MindSpeed-Core-MS/mindspeed_ms/tools/converter/patch/param_map.patch .
```

注：该 patch 适用于 Megatron-LM 框架 core_r0.6.0 分支，commit cac60ce4，Thu Apr 18 15:07:52，其他分支正在适配中，
您也可以自行适配。

2. 应用 patch

```bash
git apply param_map.patch
```

3. 正常执行 Megatron 训练脚本，并至少执行 1 次正反向训练后，保存训练权重。

在 Megatron 权重转 MindSpore 权重以及 MindSpore 权重转 Megatron 权重时均依赖此信息；为确保 param_map 信息正确，
请在每次执行 megatron 脚本前，删除 `--save` 目录。

## 2 使用示例

脚本：convert_pt2ckpt.py

### 2.1 参数说明

`python convert_pt2ckpt.py -h`

```log
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

2.2 使用示例

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python convert_pt2ckpt.py \
       --megatron-path /data/ckpt/torch_baseline_ckpts/iter_0001000/ \
       --param-map-path /home/ma-user/work/ckpt_convertion/param_map \
       --ms-path /cache/ms_save_sandbox_2/ \
       --num-layers 64 \
       --pp-size 4 \
       --dp-size 1 \
       --vpp-size 4 \
       --tp-size 8 \
       --process-limit 8 \
       --debug \
       > convert_ckpt.log 2>&1 &
```

- 可以在生成的日志中搜索 `successfully`, 源文件有几个文件夹，就应该有几个 `successfully`，可以依此判断转换是否成功。

- 脚本默认以多进程模式执行，测试或 debug 问题建议添加 `--debug`、 `--multiprocess-off`

## 3 默认目录结构

以 tp2 pp2 为例

1. 分布式优化器pt

```text
pt_path
├── iter_0000001
│   ├── mp_rank_00_000
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
└── param_map
    ├── args_tp00_pp000.pt
    ├── args_tp00_pp001.pt
    ├── args_tp01_pp000.pt
    ├── args_tp01_pp001.pt
    ├── param_map_buffer0_dp0tp0pp0vpp0.json
    ├── param_map_buffer0_dp0tp0pp1vpp0.json
    ├── param_map_buffer0_dp0tp1pp0vpp0.json
    └── param_map_buffer0_dp0tp1pp1vpp0.json
```

2. 转换文件保存后

```text
ms_path
├── rank_0
│   └── network.ckpt
├── rank_1
│   └── network.ckpt
├── rank_2
│   └── network.ckpt
└── rank_3
    └── network.ckpt
```
