# RL使用

## 转换脚本

test_convert_rl.sh 脚本自动执行以下设置过程：

1. 克隆所需仓库：
   - MindSpeed-LLM（特定提交：36bd5742b51c84ea762dc57f8943b0ee5301ee74）
   - MindSpeed（特定提交：0b832e42）
   - Megatron-LM（分支：core_r0.8.0）
   - msadapter（特定提交：f0ba1ec1c231135e668f502f13d76f1f1a37e6cb）
   - vllm（特定提交：ed6e9075d31e32c8548b480a47d1ffb77da1f54c）
   - vllm-ascend（特定提交：701a2870469d8849a50378f9450dc3e851c8af20）
   - transformers（分支：v4.47.0）

2. 应用必要的补丁：
   - 将 transformers.diff 应用于 Transformers 仓库

3. 使用 transfer.py 工具运行代码转换

## 使用说明

1. 克隆 MindSpeed-Core-MS 仓库：

   ```shell
   git clone https://gitee.com/ascend/MindSpeed-Core-MS.git
   cd MindSpeed-Core-MS/
   ```

2. 使转换脚本可执行并运行：

   ```shell
   chmod +x test_convert_rl.sh
   ./test_convert_rl.sh
   ```

启动任务前需要确保运行：

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0
source /usr/local/Ascend/ascend-toolkit/latest/env/ascend_env.sh
```