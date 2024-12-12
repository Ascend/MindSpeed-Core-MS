# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Run ParallelLinear2D"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.communication import init
from mindspore.communication.comm_func import all_gather_into_tensor

from mindspeed_ms.core.parallel_state import (
    initialize_model_parallel,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tp_x_rank,
    get_tp_x_world_size,
    get_tp_y_rank,
    get_tp_y_world_size
)

from mindspeed_ms.core.comm_utils import (
    auto_grad_scatter_along_first_dim,
    auto_grad_scatter_along_last_dim,
    auto_grad_gather_along_first_dim,
    auto_grad_gather_along_last_dim,
)
from mindspeed_ms.training import (
    parse_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
from mindspeed_ms.core.tensor_parallel import ParallelLinear2D, ColumnParallelLinear
from mindspeed_ms.core.parallel_state import TPYCollectiveComm, TPXCollectiveComm
from mindspeed_ms.core.tensor_parallel import GatherFromModelParallelRegion


class TestParallelLinear2D:
    """Test ParallelLinear2D"""
    def __init__(self, config):
        self.config = config
        self.fp32_precision_threshold = 5e-4

    def get_column_linear_grad(self, dist_schedule, model_config, input_x_data, targets):
        """Get column linear grad"""
        param_dict = self.generate_ckpt(model_config, dist_schedule)
        # 创建模型实例 dense_h_to_4h
        config = self.config
        if dist_schedule:
            net = ParallelLinear2D(
                config.hidden_size,
                config.ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                add_bias=True,
                skip_bias_add=True,
                is_expert=False,
                ag_comm_intf=TPXCollectiveComm,
                rs_comm_intf=TPYCollectiveComm,
            )
        else:
            net = ColumnParallelLinear(
                config.hidden_size,
                config.ffn_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                gather_output=False,
                skip_bias_add=False,
                is_expert=False
            )
        ms.load_param_into_net(net, param_dict)
        # # 创建损失函数和优化器
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(dense_h_to_4h.parameters(), lr=0.01)
        # 前向传播：计算预测值
        if dist_schedule:
            # s,b,h -> s/x,b,h/y
            def forward(input_x, target_x):
                input_x = auto_grad_scatter_along_last_dim(input_x, TPYCollectiveComm)
                input_x = auto_grad_scatter_along_first_dim(input_x, TPXCollectiveComm)
                # s/x,b,h/y -> s/y, b, E/x
                output_x = net(input_x)
                # s/y, b, E/x -> s, b, E
                output_x = auto_grad_gather_along_first_dim(output_x, TPYCollectiveComm)
                output_x = auto_grad_gather_along_last_dim(output_x, TPXCollectiveComm)
                # print(output_x.shape, target_x.shape)
                loss = criterion(output_x, target_x)
                return loss
        else:
            def forward(input_x, target_x):
                # print('???????????', input_x, target_x)
                input_x = input_x.contiguous()
                output_x = net(input_x)[0]
                gather_from_tensor_model_parallel_region = GatherFromModelParallelRegion()
                output_x = gather_from_tensor_model_parallel_region(output_x)
                loss = criterion(output_x, target_x)
                return loss
        outputs, inputs_gradient = ms.value_and_grad(forward, grad_position=0,
                                                     weights=net.trainable_params())(input_x_data, targets)
        print('outputs', outputs)#, inputs_gradient)

        return outputs, inputs_gradient

    def test_column_linear_fp32_precision_threshold_give_same_input_when_linear_1d_2d(self, tp_x_y):
        """Test column linear gradient with fp32 precision threshold"""
        (tp, tp_x, tp_y) = tp_x_y
        # init
        init()
        initialize_model_parallel(tensor_model_parallel_size=tp,
                                  tp_2d=True,
                                  tp_x=tp_x,
                                  tp_y=tp_y)
        seed = 42
        ms.set_seed(seed)
        ds.set_seed(seed)
        model_config = self.config
        hidden_size = 64
        ffn_hidden_size = 256
        seq = 32
        batch_size = 16
        # 生成一些随机输入数据和目标输出
        np.random.seed(seed)
        # [s, b, h]
        input_x = ms.Tensor(np.random.randn(seq, batch_size, hidden_size), dtype=mstype.float32)
        # [s, b, E]
        targets = ms.Tensor(np.random.randn(seq, batch_size, ffn_hidden_size), dtype=mstype.float32)

        # 2d linear
        output_2d, weight_grad_2d = self.get_column_linear_grad(
            dist_schedule=1,
            model_config=model_config,
            input_x_data=input_x,
            targets=targets)

        # 生成一些随机输入数据和目标输出
        np.random.seed(seed)
        # [s, b, h]
        input_x = ms.Tensor(np.random.randn(seq, batch_size, hidden_size), dtype=mstype.float32)
        # [s, b, E]
        targets = ms.Tensor(np.random.randn(seq, batch_size, ffn_hidden_size), dtype=mstype.float32)

        # 1d linear
        output_1d, weight_grad_1d = self.get_column_linear_grad(dist_schedule=0,
                                                                model_config=model_config,
                                                                input_x_data=input_x,
                                                                targets=targets)
        assert np.allclose(output_2d.asnumpy(), output_1d.asnumpy(), rtol=self.fp32_precision_threshold,
                           atol=self.fp32_precision_threshold)
        weight_grad_1d, _ = all_gather_into_tensor(weight_grad_1d[1][0],
                                                   group=get_tensor_model_parallel_group())
        weight_grad_2d = auto_grad_gather_along_first_dim(weight_grad_2d[1][0], TPXCollectiveComm)
        weight_grad_2d = auto_grad_gather_along_last_dim(weight_grad_2d, TPYCollectiveComm)
        assert np.allclose(weight_grad_1d.asnumpy(), weight_grad_2d.asnumpy(), rtol=self.fp32_precision_threshold,
                           atol=self.fp32_precision_threshold)
    def generate_ckpt(self, config, dist_schedule):
        """Generate ckpt"""
        seed = 42
        ms.set_seed(seed)
        np.random.seed(seed)
        hidden_size = config.hidden_size
        ffn_hidden_size = config.ffn_hidden_size
        param_name = 'weight'
        param_dict = {}
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size, ffn_hidden_size)), mstype.float32),
            name=param_name,
        )
        # print(param_dict[param_name].asnumpy())
        param_name = 'bias'
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((ffn_hidden_size)), mstype.float32), name=param_name
        )
        if dist_schedule:
            tp_x_rank = get_tp_x_rank()
            tp_x_world_size = get_tp_x_world_size()
            tp_y_rank = get_tp_y_rank()
            tp_y_world_size = get_tp_y_world_size()

            new_params = {}

            name = 'net.weight'
            param = param_dict['weight']
            param = param.transpose()
            start_x = tp_x_rank * (param.shape[0] // tp_x_world_size)
            end_x = (tp_x_rank + 1) * (param.shape[0] // tp_x_world_size)
            start_y = tp_y_rank * (param.shape[1] // tp_y_world_size)
            end_y = (tp_y_rank + 1) * (param.shape[1] // tp_y_world_size)
            # new_param = param[start: end]
            new_param = param[start_x: end_x, start_y: end_y]
            # print(new_param.shape)
            new_params[name] = ms.Parameter(new_param)

            name = 'net.bias'
            param = param_dict['bias']
            start = tp_x_rank * (param.shape[0] // tp_x_world_size)
            end = (tp_x_rank + 1) * (param.shape[0] // tp_x_world_size)
            new_param = param[start:end]
            new_params[name] = ms.Parameter(new_param)
        else:
            tp_rank = get_tensor_model_parallel_rank()
            tp_world_size = get_tensor_model_parallel_world_size()

            new_params = {}

            name = 'net.weight'
            param = param_dict['weight']
            start = tp_rank * (param.shape[1] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[1] // tp_world_size)
            # new_param = param[:, start: end]
            new_param = param.transpose()[start: end]
            new_params[name] = ms.Parameter(new_param)

            name = 'net.bias'
            param = param_dict['bias']
            start = tp_rank * (param.shape[0] // tp_world_size)
            end = (tp_rank + 1) * (param.shape[0] // tp_world_size)
            new_param = param[start:end]
            new_params[name] = ms.Parameter(new_param)
        return new_params


if __name__ == '__main__':
    args = parse_args()
    args.data_layout = "BSH"
    if args.yaml_cfg is None:
        config_ = core_transformer_config_from_args(args)
    else:
        config_ = core_transformer_config_from_yaml(args)

    linear2d_test = TestParallelLinear2D(config_)
    linear2d_test.test_column_linear_fp32_precision_threshold_give_same_input_when_linear_1d_2d([4, 2, 2])
