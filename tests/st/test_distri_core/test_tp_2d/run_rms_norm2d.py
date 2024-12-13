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
"""Run RMSNorm2D"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import Tensor
from mindspore.communication import init

from mindspeed_ms.core.tensor_parallel import RMSNorm2D
from mindspeed_ms.legacy.model.norm import RMSNorm
from mindspeed_ms.core.parallel_state import TPYCollectiveComm, TPXCollectiveComm

from mindspeed_ms.core.parallel_state import (
    initialize_model_parallel,
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


class TestRMSNorm2D:
    """Test RMSNorm2D"""
    def __init__(self, config):
        self.config = config
        self.fp32_precision_threshold = 5e-4

    def get_rmsnorm_grad(self, dist_schedule, config, input_x_data, targets):
        """Get RMSNorm Grad"""
        param_dict = self.generate_ckpt(config, dist_schedule)
        # 创建模型实例 dense_h_to_4h
        if dist_schedule:
            net = RMSNorm2D(config.hidden_size, 1e-6, TPYCollectiveComm)
        else:
            net = RMSNorm(config.hidden_size, 1e-6)
        ms.load_param_into_net(net, param_dict)
        criterion = nn.MSELoss()
        # 前向传播：计算预测值
        if dist_schedule:
            # s,b,h -> s/x,b,h/y
            def forward(input_x, target_x):
                input_x = auto_grad_scatter_along_last_dim(input_x, TPYCollectiveComm)
                input_x = auto_grad_scatter_along_first_dim(input_x, TPXCollectiveComm)
                # s/x,b,h/y -> s/y, b, E/x
                output_x = net(input_x)
                # s/y, b, E/x -> s, b, E
                output_x = auto_grad_gather_along_first_dim(output_x, TPXCollectiveComm)
                output_x = auto_grad_gather_along_last_dim(output_x, TPYCollectiveComm)
                loss = criterion(output_x, target_x)
                return loss
        else:
            def forward(input_x, target_x):
                output_x = net(input_x)
                loss = criterion(output_x, target_x)
                return loss
        outputs, inputs_gradient = ms.value_and_grad(forward, grad_position=0,
                                                     weights=net.trainable_params())(input_x_data, targets)
        # print(outputs, inputs_gradient)
        return outputs, inputs_gradient

    def test_column_linear_fp32_precision_threshold_give_same_input_when_linear_1d_2d(self, tp_x_y):
        """Test column linear fp32 precision threshold"""
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
        hidden_size = model_config.hidden_size
        # ffn_hidden_size = 256
        seq = 32
        batch_size = 16
        # 生成一些随机输入数据和目标输出
        np.random.seed(seed)
        # [s, b, h]
        input_x = ms.Tensor(np.random.randn(seq, batch_size, hidden_size), dtype=mstype.float32)
        # [s, b, E]
        targets = ms.Tensor(np.random.randn(seq, batch_size, hidden_size), dtype=mstype.float32)

        # 2d linear
        output_2d, weight_grad_2d = self.get_rmsnorm_grad(
            dist_schedule=1,
            config=model_config,
            input_x_data=input_x,
            targets=targets)

        # 生成一些随机输入数据和目标输出
        np.random.seed(seed)
        # [s, b, h]
        input_x = ms.Tensor(np.random.randn(seq, batch_size, hidden_size), dtype=mstype.float32)
        # [s, b, E]
        targets = ms.Tensor(np.random.randn(seq, batch_size, hidden_size), dtype=mstype.float32)

        # 1d linear
        output_1d, weight_grad_1d = self.get_rmsnorm_grad(
            dist_schedule=0,
            config=model_config,
            input_x_data=input_x,
            targets=targets)
        assert np.allclose(output_2d.asnumpy(), output_1d.asnumpy(), rtol=self.fp32_precision_threshold,
                           atol=self.fp32_precision_threshold)
        # weight_grad_2d = all_reduce(weight_grad_2d[1][0], group=TPXCollectiveComm.get_comm_group())[0]
        weight_grad_2d = auto_grad_gather_along_last_dim(weight_grad_2d[1][0], TPYCollectiveComm)
        assert np.allclose(weight_grad_1d[1][0].asnumpy(), weight_grad_2d.asnumpy(), rtol=self.fp32_precision_threshold,
                           atol=self.fp32_precision_threshold)

    def generate_ckpt(self, config, dist_schedule):
        """Generate ckpt"""
        seed = 42
        ms.set_seed(seed)
        np.random.seed(seed)
        hidden_size = config.hidden_size
        param_name = 'weight'
        param_dict = {}
        param_dict[param_name] = ms.Parameter(
            Tensor(np.random.random((hidden_size)), mstype.float32),
            name=param_name,
        )
        if dist_schedule:
            tp_y_rank = get_tp_y_rank()
            tp_y_world_size = get_tp_y_world_size()

            new_params = {}

            name = 'net.weight'
            param = param_dict['weight'].asnumpy()
            start = tp_y_rank * (param.shape[0] // tp_y_world_size)
            end = (tp_y_rank + 1) * (param.shape[0] // tp_y_world_size)
            new_param = param[start: end]
            new_params[name] = ms.Parameter(new_param)
        else:
            new_params = {}
            name = 'net.weight'
            param = param_dict['weight']
            new_params[name] = ms.Parameter(param)
        return new_params


if __name__ == '__main__':
    args, defaults = parse_args()
    args.data_layout = "BSH"
    if args.yaml_cfg is None:
        config_ = core_transformer_config_from_args(args)
    else:
        config_ = core_transformer_config_from_yaml(args)
    a = TestRMSNorm2D(config_)
    a.test_column_linear_fp32_precision_threshold_give_same_input_when_linear_1d_2d([4, 2, 2])
