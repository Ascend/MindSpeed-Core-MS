# 版本控制脚本

该目录包含两个实用脚本，用于同步主仓库及其子模块的提交记录。

## 脚本说明

### 1. `export_version.sh`

导出所有被跟踪仓库当前的提交 ID、提交时间和分支名称，生成一个统一的版本快照文件。

#### 用法

```bash
bash tools/version_control/export_version.sh [输出文件]
```

* 默认输出文件：`repo_commits.txt`
* 跟踪的仓库包括：

```text
MindSpeed-Core-MS  
MindSpeed-LLM  
MindSpeed  
Megatron-LM  
msadapter  
transformers
```

#### 示例输出（列对齐）

```text
#Repo                CommitID                                CommitTime               Branch
MindSpeed-Core-MS    2fbd....da80                            2025-07-25 14:00:00 +0800 main
...
```

---

### 2. `import_version.sh`

根据版本快照文件中的提交 ID，将所有仓库恢复到指定状态。

#### 用法

```bash
bash tools/version_control/import_version.sh [输入文件]
```

* 默认输入文件：`repo_commits.txt`
* 此操作将对所有仓库执行 **detached HEAD checkout**（游离 HEAD 检出）。
* 如果你有未提交的更改，请谨慎使用。
