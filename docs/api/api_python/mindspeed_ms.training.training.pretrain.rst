mindspeed_ms.training.training.pretrain
============================================

.. py:function:: mindspeed_ms.training.training.pretrain(train_valid_test_datasets_provider, model_provider_func, model_type, forward_step_func=None, process_non_loss_data_func=None, **kwargs)

    预训练接口。用户通过此接口传入数据迭代器、模型定义、配置项等即可开始模型训练。

    参数：
        - **train_valid_test_datasets_provider** (function) - 一个输入为train/valid/test数据集大小，输出为对应的数据集的函数。当前未使能。
        - **model_provider_func** (function) - 一个会返回模型的函数。
        - **model_type** (enum) - 表示训练模型的类型的枚举类。当前未使能。
        - **forward_step_func** (function，可选) - 前向函数，当前未使能。默认值：``None`` 。
        - **process_non_loss_data_func** (function，可选) - 对网络的输出进行后处理的函数。当前未使能。默认值：``None`` 。
        - **kwargs** (dict) - 其他输入。数据迭代器、配置项 `all_config` 等从这里传入。

    异常：
        - **ValueError** - `all_config` 没有从kwargs中传入。
