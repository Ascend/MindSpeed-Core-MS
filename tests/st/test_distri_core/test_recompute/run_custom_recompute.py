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
""" Test ParallelTransformer. """
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, grad
import mindspore.common.dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init

from mindspeed_ms.tools import DictConfig
from mindspeed_ms.training import parse_args, core_transformer_config_from_yaml
from mindspeed_ms.training.yaml_arguments import validate_yaml
from mindspeed_ms.training.global_vars import set_global_variables
from mindspeed_ms.core.parallel_state import initialize_model_parallel, get_rank
from mindspeed_ms.legacy.model import ParallelTransformer
from mindspeed_ms.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding


class ParallelTransformerNet(nn.Cell):
    """ ParallelTransformerNet. """
    def __init__(self, config, with_rope=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(kv_channels=config.hidden_size//config.num_attention_heads,
                                        rotary_percent=1.0)
        self.transformer = ParallelTransformer(config=config, post_norm=False, model_type=None)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """ construct. """
        x = x.swapaxes(0, 1)
        labels = labels.swapaxes(0, 1)
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output = self.transformer(x, attention_mask, rotary_pos_emb=emb)
        else:
            output = self.transformer(x, attention_mask)
        output = output.swapaxes(0, 1)
        labels = labels.swapaxes(0, 1)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def run_parallel_transformer(config, args, recompute_method):
    """ Test ParallelTransformer. """
    seed = 2024
    batch_size = 1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON', pynative_synchronize=True)

    full_recompute_list = None
    select_recompute_list = None
    select_comm_recompute_list = None
    recompute_config = None
    if recompute_method == 0:
        full_recompute_list = [config.num_layers]
    elif recompute_method == 1:
        select_recompute_list = [config.num_layers]
    elif recompute_method == 2:
        select_comm_recompute_list = [config.num_layers]
    if recompute_method != -1:
        recompute_config = {"recompute": full_recompute_list,
                            "select_recompute": select_recompute_list,
                            "select_comm_recompute": select_comm_recompute_list}
        recompute_config = DictConfig(**recompute_config)
    init()
    initialize_model_parallel(tensor_model_parallel_size=config.tensor_model_parallel_size,
                              use_sequence_parallel=config.sequence_parallel)

    ms.set_seed(seed)
    ms.manual_seed(seed)
    np.random.seed(seed)
    config.recompute_config = recompute_config
    network = ParallelTransformerNet(config=config, with_rope=False)
    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)
    grad_fn = grad(network.construct, grad_position=(0), weights=None)
    input_ids = Tensor(np.random.random((batch_size, args.seq_length, config.hidden_size)).astype(np.float32),
                       ms.float32)
    attn_mask = Tensor(np.tril(np.ones(shape=(1, 1, 2 * args.seq_length, 2 * args.seq_length))).astype(np.uint8),
                       ms.float32)
    labels = Tensor(np.zeros((batch_size, args.seq_length)).astype(np.float32), ms.float32)

    grad_value = grad_fn(input_ids, attn_mask, labels)
    if get_rank() == 0:
        if recompute_config is None:
            # without recompute
            np.save('grad_without_recompute.npy', grad_value.asnumpy())
            return
        grad_without_recompute = np.load('./grad_without_recompute.npy')
        print('grad_value:', grad_value.asnumpy())
        print('grad_without_recompute:', grad_without_recompute)
        assert np.allclose(
            grad_without_recompute, grad_value.asnumpy(), atol=1e-3
        ), "Gradient checkpointed recompute failed."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute_method', type=int, default=-1,
                        help="0 full, 1 select, 2 select comm, -1 not recompute")
    parser.add_argument('--yaml-cfg', default=None, type=str)
    extra_args = parser.parse_args()

    def extra_parser_provider(inner_parser):
        inner_parser.add_argument('--recompute_method', type=int, default=-1,
                                  help="0 full, 1 select, 2 select comm, -1 not recompute")
        return inner_parser


    main_args, defaults = parse_args(extra_args_provider=extra_parser_provider)
    main_args = validate_yaml(main_args, defaults, {})
    set_global_variables(main_args, False)
    used_config = core_transformer_config_from_yaml(main_args)
    run_parallel_transformer(used_config, main_args, extra_args.recompute_method)
