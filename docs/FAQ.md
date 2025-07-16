# FAQ

## 1 基础环境搭建遇到问题，无法快速上手，是否有镜像参考？

已知有[非正式商用镜像](http://mirrors.cn-central-221.ovaijisuan.com/detail/188.html)可协助上手，通过该镜像创建并进入容器后，可以试用本项目所提供的功能，具体信息和使用请参考下面链接:  
<http://mirrors.cn-central-221.ovaijisuan.com/detail/188.html>

## 2 数据转换时报错OSError:cannot load library 'libsndfile.so':libsndfile.so: cannot open sharedobject file: No such file or directory,该怎么解决？

该报错通常是由于操作系统中缺乏依赖库libsndfile1所致,可以执行以下命令安装相应的依赖库（根据具体操作系统类型选择）：
yum/apt install libsndfile1

## 3 运行过程中出现类似ModuleNotFoundError: No module named 'torch'/'torch_npu'的错误，该怎么解决？

该报错通常是由于环境变量PYTHONPATH未正确设置所致，请在执行命令前，先设置PYTHONPATH环境变量，其中LLM可以直接在终端中运行以下命令来设置:
source /test/scripts/set_path.sh

对于MM及RL，建议参考test_convert_mm/rl.sh脚本中的PYTHONPATH设置方式。


