# 代码转写工具

Megatron、MindSpeed、MindSpeed-LLM等仓库中基于PyTorch的原生代码的部分写法，与基于MindSpore框架的写法有部分差异。
这部分代码需要通过转写工具转写为可基于MindSpore框架运行的代码。此转写工具可将PyTorch代码批量转写为
[MSAdapter](https://openi.pcl.ac.cn/OpenI/MSAdapter.git) 兼容性代码，仍保留MSAdapter兼容仓库的PyTorch对应接口; 后续方案可直接转换为MindSpore原生代码（敬请期待）。

**概览**
该工具基于 [libcst](https://libcst.dev/) 实现静态语法树转换，
将原仓库中基于 `torch` 的python API 调用、注释和文档字符串（docstring）统一替换为指定的新名称（如 `msadapter` 或 `mindspore`），
并支持多进程批量处理与日志输出，方便代码高速一键转换。

## 目录结构

```text
tools
└── convert
   ├── convert.py
   ├── convert.sh # 转换脚本入口
   ├── mapping_resources
   │   ├── api_mapping.json # 已对齐的API
   │   └── special_case.py # 需要特殊处理的文件和处理方法
   └── modules
       ├── api_transformer.py # 核心转写模块
       ├── string_transformer.py # 辅助模块
       └── utils.py

```

## 安装与依赖

1. Python ≥3.9。
2. 安装依赖库：

   ```bash
   pip install libcst tqdm
   ```

## 命令行使用

```bash
cd MindSpeed-Core-MS
bash auto_convert_xxx.sh # 正常获取MindSpeed, MindSpeed-LLM 等代码仓并应用MindSpore需要的patch, 注意不需要设置PYTHONPATH
# 此时若设置PYTHONPATH, 应能正常拉起模型训练
bash tools/convert/convert.sh dir_name/ # 拷贝需要的三方库和依赖的代码至dir_name目录, 并对dir_name目录应用代码转写
```

* 需将`dir_name`设置为目标文件夹的名称，当前仅支持 `MindSpeed-LLM/` 与 `MindSpeed-MM/`。

* 默认在`dir_name`目录下生成日志文件`result.log`, 其中记录每个文件转写结果, 期望日志文件中不包含`False`。

* 此时能基于`mindspore/msadapter`拉起模型,不需要额外设置`PYTHONPATH`。

## 单独使用(只转换某个代码仓)

假设要将 Megatron-LM、MindSpeed-LLM、MindSpeed-MM 等仓库转写为 MindSpore 适配：

1. 执行完本仓库根目录下的 `auto_convert_xxx.sh` 脚本后，根目录下存在Megatron、MindSpeed等仓库：

2. 进入父目录，执行转写：

   ```bash
   python tools/convert/convert.py \
     --path_to_change ./Megatron-LM \
     --multiprocess 16 \
     && \
   python tools/convert/convert.py \
     --path_to_change ./MindSpeed \
     --multiprocess 16 \
     && \
   python tools/convert/convert.py \
     --path_to_change ./MindSpeed-LLM \  # 根据实际仓库修改
     --multiprocess 16
   ```
