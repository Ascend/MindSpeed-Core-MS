<p align="center"> <img src="docs/logo.png" height="103px" width="700px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed-Core-MS/blob/dev/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed-Core-MS/blob/dev/LICENSE">
        <img alt="Badge" src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed-Core-MS">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

MindSpeed-Core-MS 是链接华为自研AI框架[MindSpore](https://www.mindspore.cn/install/)+华为[昇腾大模型加速解决方案MindSpeed](https://www.hiascend.cn/software/mindspeed)的重要组件，旨在提供华为全栈易用的端到端的自然语言模型以及多模态模型训练解决方案。MindSpeed-Core-MS内部提供一键式补丁工具，可帮助用户将模型使能加速库MindSpeed/MindSpeed-LLM/MindSpeed-MM以及三方库的AI框架依赖由PyTorch无缝切换为MindSpore，以此获得更极致的性能体验。

---

# 配套版本与支持模型

## 配套版本

MindSpeed-Core-MS的依赖配套如下表，安装步骤参考[基础安装指导](./docs/INSTALLATION.md)。

| 依赖软件         |                                                              |
| ---------------- | ------------------------------------------------------------ |
| 昇腾NPU驱动固件  | [在研版本](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) |
| 昇腾 CANN        | [在研版本](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| MindSpore        | [MindSpore 2.6.0](https://www.mindspore.cn/install/)         |
| Python           | >=3.9                                                        |
| Python三方库依赖 | requirements.txt                                             |
| 镜像链接         | [images](http://mirrors.cn-central-221.ovaijisuan.com/detail/129.html) |

注：Python 三方库依赖文件`requirements.txt`列举的是模型训练所需要的python三方库

## 支持模型
下方仅部分列举所支持模型。

<table>
  <a id="jump1"></a>
  <caption>模型部分列表</caption>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数量</th>
      <th>序列</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">DeepSeek V3</a></td>
      <td rowspan="4"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">671B</a></td>
      <td> 4K </td>
      <td>预训练</td>
      <td> 64x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td> 4K </td>
      <td>微调</td>
      <td> 64x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td> 4K </td>
      <td>Lora微调</td>
      <td> 8x8 </td>
      <td> BF16 </td>
    </tr>
    <tr>
      <td> 4K </td>
      <td> r1-zero </td>
      <td> 52x8 </td>
      <td> BF16 </td>
      <td> </td>
      <td> </td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main">Qwen25</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main">7B</a></td>
      <td> 4k </td>
      <td> r1-zero </td>
      <td> 2x8 </td>
      <td> BF16 </td>
      <td> </td>
      <td> </td>
    </tr>
  </tbody>
</table>

# 使用指南

按照[基础安装指导](./docs/INSTALLATION.md)完成相关基础依赖安装后，用户可试用自然语言模型对应的补丁工具拉取代码并完成相关转换。

## 仓库拉取

执行以下命令拉取MindSpeed-Core-MS代码仓

```shell
git clone -b feature-0.2 https://gitee.com/ascend/MindSpeed-Core-MS.git
```

## MindSpeed-LLM

### 一键转换

补丁工具集成了相关代码仓拉取、代码自动转换适配以及模型启动shell脚本自动适配功能，依赖以下配置：

- 所部署容器网络可用，python已安装
- git已完成配置，可以正常进行clone操作

详细介绍请见补丁工具说明。用户执行命令即可一键转换。

```shell
cd MindSpeed-Core-MS
#deepseek v3预训练、微调
source test_convert_llm.sh
```

### 设置环境

```shell
#deepseek v3预训练、微调
MindSpeed_Core_MS_PATH=$(pwd)
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/Megatron-LM:${MindSpeed_Core_MS_PATH}/MindSpeed:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM:${MindSpeed_Core_MS_PATH}/transformers/src/:$PYTHONPATH
```

### 模型训练

在进行一键转换安装后，用户即可进行模型训练，提供以下模型任务拉起流程作为参考。

- [**DEEPSEEK-V3预训练 & 微调**](./docs/deepseekv3.md)

---

## MindSpeed-RL

### 一键转换

补丁工具集成了相关代码仓拉取、代码自动转换适配以及模型启动shell脚本自动适配功能，依赖以下配置：

- 所部署容器网络可用，python已安装
- git已完成配置，可以正常进行clone操作

详细介绍请见补丁工具说明。用户执行命令即可一键转换。

```shell
cd MindSpeed-Core-MS
#deepseek v3-r1-zero、qwen25-7b-r1-zero
source test_convert_rl.sh
```

### 设置环境

```shell
#deepseek v3-r1-zero、qwen25-7b-r1-zero
MindSpeed_Core_MS_PATH=$(pwd)
Ascend_PATH=/usr/loacl/Ascend/ascend-toolkit/latest/
export PYTHONPATH=${MindSpeed_Core_MS_PATH}/msadapter/mindtorch:${MindSpeed_Core_MS_PATH}/MindSpeed-LLM/:${MindSpeed_Core_MS_PATH}/MindSpeed/: \
                     ${MindSpeed_Core_MS_PATH}/Megatron-LM/:${MindSpeed_Core_MS_PATH}/vllm/:${MindSpeed_Core_MS_PATH}/vllm-ascend/: \
                     ${MindSpeed_Core_MS_PATH}/transformers/src/: \
                     ${Ascend_PATH}/python/site-packages:${Ascend_PATH}/opp/built-in/op_impl/ai_core/tbe:
```

### 模型训练

在进行一键转换安装后，用户即可进行模型训练，提供以下模型任务拉起流程作为参考。

- [**DEEPSEEK V3-R1-ZERO/QWEN25-7B-R1-ZERO**](./docs/GRPO.md)

---

# 常见问题 FAQ

相关FAQ请参考链接：[FAQ](./docs/FAQ.md)

---

# 版本维护策略

MindSpeed-Core-MS版本有以下五个维护阶段：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划                | 1—3 个月 | 计划特性                                                                 |
| 开发                | 3 个月   | 开发特性                                                                 |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed-Core-MS版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                                           |

MindSpeed-Core-MS已发布版本维护策略：

| **MindSpeed-Core-MS版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**         | **EOL日期** |
|-----------------|-----------|--------|------------|-----------------------|-----------|
| feature-0.2 |  Demo  | 开发   |  |   |           |
|              |  Demo  | 开发   |  |   |           |

---

# 安全声明

[MindSpeed-Core-MS安全声明](./docs/SECURITYNOTE.md)
