# 1. MindSpore Checkpoint 转为 Megatron Checkpoint

## 1.1. 前置条件

1. param_map*.json：Megatron 训练时保存的 param_map。如何获取请参照 `convert_pt2ckpt.md/获取 param_map`；
2. args_xx_xxx.pt：Megatron Checkpoint 转为 MindSpore Checkpoint 时，会将 Megatron 部分参数保存在 args_xx_xxx.pt 中，转换时需要将此参数回写到 Megatron Checkpoint 中，如何获取请参照 `convert_pt2ckpt.md`；
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

4. 使用 MindSpore 训练保存的权重，需要先使用 combine_ckpt_dp_zero.py 脚本进行参数合并（dp_size = 1 也需要执行此项操作），之后使用合并后的权重做 MindSpore 到 Megatron 的转换，combine_ckpt_dp_zero.py 脚本使用请参考 `combine_ckpt_dp_zero.md`。

## 1.2. ckpt2pt 使用示例

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

### 1.2.1. 仅转换模型权重

转换时添加 `--convert-param-only` 参数，这种模式无需param-map-path中的json文件，仅生成包含 `bfloat16` 类型的模型权重文件 `model_optim_rng.pt`。

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

### 1.2.2. 转换模型权重和分布式优化器状态

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
