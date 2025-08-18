# Patch融合

## 零、环境安装

安装依赖包

```python
pip install inflection==0.5.1
pip install libcst==1.8.2
```

## 一、使用方式

1）run_merge.sh 脚本接收两个参数：补充融合的 MindSpeed-Core-MS 文件夹路径 `ROOT_DIR` 和 检索出的 patch json文件路径 `PATCH_JSON_PATH`

- `ROOT_DIR`: **待融合的** MindSpeed-Core-MS 文件夹路径
- `PATCH_JSON_PATH`: patch 检索的 json 文件路径

```python
bash tools/convert/patch_merge/run_merge.sh <ROOT_DIR> <PATCH_JSON_PATH>
```

2）运行模型启动脚本前，修改 MindSpeed-LLM/mindspeed_llm/mindspore/utils.py

由于融合后，有些patch已经不再被register，因此需要将 `reset_patch` 修改为

```python
def reset_patch(original_func_name):
    '''clear the wrapper info in Patch object'''
    if original_func_name in MegatronAdaptation._patch_info_collection:
        target_patch = MegatronAdaptation._patch_info_collection[original_func_name]
        target_patch.wrappers = []
```

## 二、覆盖率检测

patch 融合后会在融合的分支处打印 patch 信息，运行模型脚本后，可以根据 log 和对应的 patch json 文件，统计融合后的 patch覆盖率

python 脚本需要两个输入

- `JSON_FILE`: patch 融合的 json 文件
- `LOG_FILE`: 模型脚本运行结束后的 log 文件（目前仅支持输入单个 worker 的log）

使用方式

```python
python tools/convert/patch_merge/modules/coverage.py --json-file <JSON_FILE> --log-file <LOG_FILE>
```

## 三、约束条件

1. 目前仅支持融合 megatron core r0.8.0，因此需要首先拉取 MindSpeed-Core-MS r0.3.0 分支，执行 `auto_convert_llm.sh` 后，再将路径作为 `run_merge.sh` 的 `ROOT_DIR`
2. patch融合中，为了获取 patch 对应的文件路径，使用了 importlib 进行模拟导入，因此需要**在 MindSpeed-Core-MS 运行环境下进行 patch融合**

## 四、Debug工具

在 `check_merge.sh` 中

1. 设置待融合的 MindSpeed-Core-MS 路径 `ROOT_DIR` 和 patch 融合工具路径 `CONVERT_TOOL_DIR` （均需要绝对路径）

2. 修改 `tools/convert/patch_merge/run_merge.sh` 中的 `MERGE_PY` 为绝对路径

3. 添加模型脚本名 `MODEL_SHELL`，使用在 MindSpeed-LLM 下的相对路径

4. 修改 `PATTERN` 为想要对齐的 loss 打印

```python
bash tools/convert/patch_merge/check_merge.sh
```

测试会首先在 `ROOT_DIR` 下创建工作空间，拷贝MindSpeed-LLM、MindSpeed、Megatron-LM 到新的文件夹，执行融合后，运行模型脚本，最后搜索 msrun_log 中是否有对应的 `PATTERN`
二分测试会自动二分查找出融合失败或精度不对齐的 patch

`check_merge.sh`中可以选择以下两种测试，可根据需要修改调用：

- 单用例测试：`test_single_case ${PATCH_JSON_PATH}`

- 二分测试：`binary_search ${PATCH_JSON_PATH}`
