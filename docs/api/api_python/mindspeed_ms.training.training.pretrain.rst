mindspeed_ms.training.training.pretrain
============================================

.. py:function:: mindspeed_ms.training.training.pretrain(train_valid_test_datasets_provider, model_provider_func, model_type, forward_step_func=None, process_non_loss_data_func=None)

    预训练接口。

    该函数会按顺序执行以下流程：

    1. 使用model_provider_func创建模型，优化器和学习率规划器
    2. 使用forward_step_func训练模型。

    参数：
        - **train_valid_test_datasets_provider** (function) - 一个输入为train/valid/test数据集，输出为train，valid，test数据集的函数。当前未使能。
        - **model_provider_func** (function) - 一个会返回原版模型的函数，原版指的是不包含fp16和ddp的一个简单模型。
        - **model_type** (enum) - 训练模型的类型。
        - **forward_step_func** (function，可选) - 一个函数，它接受一个 `data iterator` 和 `model` ，并返回一个 `loss` 标量，该标量带有一个字典，其中 `key:value` 是我们在训练期间想要监视的信息，例如 `lm_loss: value` 。我们还要求该函数将 `batch generator` 添加到定时器类中。目前未使用。默认值：``None`` 。
        - **process_non_loss_data_func** (function，可选) - 用于将网络的进程输出后处理的函数。它可以用于将输出张量（例如图像）转储到 tensorboard。它接受 `collected data` （张量列表）、 `current iteration index` 和 `tensorboard writer` 作为参数。目前未使用。默认值：``None`` 。
        - **kwargs** (dict) - 其他输入。

    异常：
        - **ValueError** - all_config没有从kwargs中传入。
