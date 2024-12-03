mindspeed_ms.training.training.pretrain
============================================

.. py:function:: mindspeed_ms.training.training.pretrain(train_valid_test_datasets_provider, model_provider_func, model_type, forward_step_func=None, process_non_loss_data_func=None)

    主要的训练流程。

    该函数会按顺序执行以下流程：
        1）使用model_provider_func创建模型，优化器和学习率规划器
        2）使用forward_step_func训练模型。

    输入：
        - **train_valid_test_datasets_provider** (function) - 一个输入为train/valid/test数据集，输出为train，valid，test数据集的函数。当前未使能。
        - **model_provider_func** (function) - 一个会返回原版模型的函数，原版指的是不包含fp16和ddp的一个简单模型。
        - **model_type** (enum) - 训练模型的类型。
        - **forward_step_func** (function) - 单步训练的函数。默认值：None。
        - **process_non_loss_data_func** (function) - 一个后处理网络输出的函数。默认值：None。

    异常：
        - **ValueError** - all_config没有从kwargs中传入。
