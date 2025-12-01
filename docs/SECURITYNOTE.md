## 安全声明

### 依赖三方软件安全

推荐用户通过MindSpeed-Core-MS自动安装依赖三方软件（指定版本或默认最新版本），如因用户使用旧版本的三方软件而导致安全漏洞产生影响，MindSpeed-Core-MS不承担相关责任。

### 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户。

### 安全隐私声明

用户在使用个人数据时请遵从当地适用的法律法规。

### 系统安全加固

- 用户可在运行系统配置时开启 ASLR（级别2）以提高系统安全性，保护系统随机化开启。

可参考以下方式进行配置：

```shell
echo 2 > /proc/sys/kernel/randomize_va_space
```

### 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
- 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件、MindSpeed-Core-MS安装目录、多用户使用共享数据集等敏感内容做好权限管控，管控权限可参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)进行设置。
- MindSpeed 中各类融合算子通过调用 PyTorch 中的 cpp_extension 特性进行编译，编译结果会默认缓存到 `~/.cache/torch_extensions` 目录下，建议用户根据自身需要，参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)对生成文件做好权限控制。
- 原生 Megatron-LM 以及 PyTorch 框架运行中所生成的文件权限依赖系统设定，如 Megatron-LM 生成的数据集索引文件、torch.save 接口保存的文件等。建议当前执行脚本的用户根据自身需要，对生成文件做好权限控制，设定的权限可参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)进行设置。
- 运行时 CANN 可能会缓存算子编译文件，存储在运行目录下的`kernel_meta_*`文件夹内，加快后续训练的运行速度，用户可根据需要自行对生成后的相关文件进行权限控制。
- 用户安装和使用过程需要做好权限控制，建议参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)文件权限参考进行设置。如需要保存安装/卸载日志，可在安装/卸载命令后面加上参数 `--log <FILE>`， 注意对`<FILE>`文件及目录做好权限管控。

### 数据安全声明

- MindSpeed 依赖 CANN 的基础能力实现 AOE 性能调优、算子 dump、日志记录等功能，用户需要关注上述功能生成文件的权限控制。

### 运行安全声明

- 建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
- MindSpeed 在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看 CANN 日志、解析生成的 Core Dump 文件等方式。
- MindSpeed-Core-MS在模型运行时内部用到了MindSpore，可能会因为版本不匹配导致运行错误，具体可参考[MindSpore安全声明](https://gitee.com/mindspore/mindspore/blob/master/SECURITY.md)。

### 公网地址声明

| 类型 | 开源代码地址 | 文件名 | 公网IP地址/公网URL地址/域名/邮箱地址 | 用途说明                                  |
| ---- | ----- | ---- | ---- |---------------------------------------|
| 开发引入 | - | auto_convert.sh、auto_convert_rl.sh | https://gitcode.com/Ascend/MindSpeed.git | 用于拉取MindSpeed代码仓                      |
| 开发引入 | - | auto_convert.sh、auto_convert_rl.sh | https://gitcode.com/ascend/MindSpeed-LLM.git | 用于拉取MindSpeed-LLM代码仓                  |
| 开发引入 | - | auto_convert.sh | https://gitcode.com/ascend/MindSpeed-MM.git | 用于拉取MindSpeed-MM代码仓                   |
| 开发引入 | - | auto_convert_rl.sh | https://gitcode.com/ascend/MindSpeed-RL.git | 用于拉取MindSpeed-RL代码仓                   |
| 开发引入 | - | auto_convert.sh、auto_convert_rl.sh | https://gitee.com/mindspore/msadapter.git | 用于拉取msadapter代码仓                      |
| 开发引入 | - | auto_convert.sh | https://openi.pcl.ac.cn/OpenI/MSAdapter.git | 用于拉取MSAdapter代码仓                      |
| 开发引入 | - | auto_convert.sh、auto_convert_rl.sh | https://gitee.com/mirrors/Megatron-LM.git | 用于拉取Megatron代码仓                       |
| 开发引入 | - | auto_convert_rl.sh | https://gitee.com/mirrors/huggingface_transformers.git | 用于拉取transformers代码仓                   |
| 开发引入 | - | auto_convert_rl.sh | https://gitee.com/modelee/accelerate.git | 用于拉取accelerate代码仓                     |
| 开发引入 | - | auto_convert_rl.sh | https://gitee.com/mirrors/vllm.git | 用于拉取vllm代码仓                           |
| 开发引入 | - | auto_convert_rl.sh | https://gitee.com/mirrors/vllm-ascend.git | 用于拉取vllm-ascend代码仓                    |
| 开发引入 | - | auto_convert_rl.sh | https://gitee.com/mirrors/huggingface_hub.git | 用于拉取huggingface_hub代码仓                |
| 开发引入 | - | setup.py | https://gitcode.com/ascend/MindSpeed-Core-MS | setup脚本方式安装MindSpeed-Core-MS地址        |
| 开发引入 | - | setup.py | https://gitcode.com/ascend/MindSpeed-Core-MS/issues | setup脚本方式安装MindSpeed-Core-MS issues地址 |

### 附录

#### A-文件（夹）各场景权限管控推荐最大值

| 类型                | linux权限参考最大值   |
|-------------------|----------------|
| 用户主目录             | 750（rwxr-x---） |
| 程序文件(含脚本文件、库文件等)  | 550（r-xr-x---） |
| 程序文件目录            | 550（r-xr-x---） |
| 配置文件              | 640（rw-r-----） |
| 配置文件目录            | 750（rwxr-x---） |
| 日志文件(记录完毕或者已经归档)  | 440（r--r-----） |
| 日志文件(正在记录)        | 640（rw-r-----） |
| 日志文件目录            | 750（rwxr-x---） |
| Debug文件           | 640（rw-r-----） |
| Debug文件目录         | 750（rwxr-x---） |
| 临时文件目录            | 750（rwxr-x---） |
| 维护升级文件目录          | 770（rwxrwx---） |
| 业务数据文件            | 640（rw-r-----） |
| 业务数据文件目录          | 750（rwxr-x---） |
| 密钥组件、私钥、证书、密文文件目录 | 700（rwx—----）  |
| 密钥组件、私钥、证书、加密密文   | 600（rw-------） |
| 加解密接口、加解密脚本       | 500（r-x------） |