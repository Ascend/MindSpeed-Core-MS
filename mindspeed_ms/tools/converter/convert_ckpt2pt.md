# Mindspore 权重转为 Megatron 权重

## 前置条件

1. 在使用 Megatron 脚本进行训练时，会保存 param_map 信息，此信息为 Megatron 优化器权重中，桶具体以何种排布进行数据构造，在 Megatron 转 Mindspore 权重以及 Mindspore 权重转Megatron权重时均依赖此信息；同时依赖Megatron训练时保存的param_map、以及 Megatron 权重转为 Mindspore 权重时保存的 args_xx_xxx.pt 的权重参数文件，此两者需保存在同一路径下，路径下直接为 json 及 pt 文件。

2. 使用 Mindspore 训练保存的权重，其参数以桶形式在dp域内进行分布式保存，需要先使用 combine_ckpt_dp_zero.py 脚本进行参数合并（dp_size = 1 也需要执行此项操作），之后使用合并后的权重做 Mindspore 到 Megatron 的转换，combine_ckpt_dp_zero.py 脚本使用请参考同级目录下combine_ckpt_dp_zero.md 脚本；

## 使用脚本

convert_ckpt2pt.py

## 参数说明

```text
--num-layers:        int, 总层数
--dp-size:           int, dp 数
--cp-size:           int, cp 数
--tp-size:           int, 模型并行数
--pp-size:           int, 流水线并行数
--vpp-size:          int, vpp 数，每个 pp stage 下有几个vpp
--src-model-format:  str, Mindspore 权重格式，可选 ckpt、safetensors，不传默认 ckpt
--noop:              int, 空层所在位置，当前仅支持在最后，不传默认没有空层
--ms-path:           str, Mindspore 权重路径，此路径下为 rank_xx 的权重文件夹
--param-map-dir:     str, param_map 路径
--megatron-path:     str, 需要保存的转换后的 Megatron 权重路径
--convert-param-only:arg, 设定是否只转换模型参数而忽略优化器参数
```

## 示例

```bash
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
python convert_ckpt2pt.py --num-layers 8 \
    --dp-size 1 \
    --cp-size 1 \
    --tp-size 2 \
    --pp-size 4 \
    --vpp-size 2 \
    --noop 62 63 \
    --ms-path /home/ma-user/work/ckpt2pt/ms_ckpt \
    --param-map-path /home/ma-user/work/work/ckpt2pt/param_map/buffers \
    --megatron-path /home/ma-user/work/ckpt2pt/megatron_pt
```
