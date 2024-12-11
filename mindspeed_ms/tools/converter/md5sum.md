# 比较转换前后pt文件的md5sum

比较pt->ckpt->pt转换前后pt文件的md5sum，可以快速验证转换脚本的基本功能。

预期结果：

1. `convert_param_only=True`，转换前后 `model_optim_rng.pt` 使用 `md5sum.py` 预期md5值完全一致
2. `convert_param_only=False`, 转换前后 `distrib_optim.pt` 使用 `md5sum`/`md5sum.py` 预期md5值完全一致；由于megatron 0.6 bug, 转换前后 `model_optim_rng.pt` 的模型权重不一致，无法比较md5值

## 1 背景

| pt文件 | distrib_optim.pt | model_optim_rng.pt |
| :-----:| :----: | :----: |
| 内容 | **已在cpu侧完成dp0聚合的**权重及优化器状态 | 模型权重，训练超参，版本号，随机数状态等 |
| key | `per_bucket_numel`, `per_bucket_numel_unpadded`, `{vpp}`等 | `args`, `model{vpp}`, `optimizer`, `rng_state`, `checkpoint_version`等 |
| 比较工具 | `md5sum`, `md5sum.py` | `md5sum.py`|

`distrib_optim.pt` 可采用 `md5sum`, `md5sum.py` 任一工具进行比较。
`model_optim_rng.pt` 保存的模型权重落盘时map_location仍为device侧，而转换脚本采用cpu侧转换的策略，md5sum无法直接比较。使用`md5sum.py`工具，会在cpu侧加载pt文件，经str序列化后，比较md5sum值并输出结果。

## 2 使用示例

与`md5sum`工具相同，`python md5sum` 接受n个pt文件并比较他们的md5sum，如下所示：

```bash
python md5sum.py origin/model_optim_rng.pt converted/model_optim_rng.pt
```

结果不一致时，输出如下：

```bash
# python md5sum.py origin/saved_cp2tp2pp2_0_0.pt refresh/saved_cp2tp2pp2_1_0.pt
>20:46:08  : -------------Comparing MD5Sum-------------
Different md5sum
origin/saved_cp2tp2pp2_0_0.pt md5sum: 0e3fa0233ca695a9b8061c3716a8bc46
refresh/saved_cp2tp2pp2_1_0.pt md5sum: 6ac91839e607966389ebb4b877186ae8
```

结果一致时，输出如下：

```bash
>20:52:22  : -------------Comparing MD5Sum-------------
All files have the same md5sum value: 0e3fa0233ca695a9b8061c3716a8bc46
```