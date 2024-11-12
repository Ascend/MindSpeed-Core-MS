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
"""Language model test"""
import mindspore as ms
from mindspore import nn
from mindspore.communication import init
from mindspeed_ms.training import (
    parse_args,
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)
from mindspeed_ms.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from mindspeed_ms.training import get_model
from mindspeed_ms.core.transformer import TransformerConfig
from mindspeed_ms.core.parallel_state import initialize_model_parallel
from mindspeed_ms.legacy.model.module import Module


# pylint: disable=W0621
class RotaryEmbeddingNet(Module):
    """RotaryEmbedding net"""
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        args = get_args()
        rotary_dim = config.hidden_size // config.num_attention_heads
        self.rotary_embedding = RotaryEmbedding(
            rotary_dim,
            rotary_percent=args.rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )

    def construct(self, max_seq_len, offset=0):
        """forward"""
        output = self.rotary_embedding(max_seq_len, offset)
        return output


def run_rotary_embedding(config: TransformerConfig):
    """Run rotary embedding layer"""
    args = get_args()
    ms.set_seed(2024)
    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE, deterministic='ON')
    init()

    pp = config.pipeline_model_parallel_size
    tp = config.tensor_model_parallel_size
    if config.virtual_pipeline_model_parallel_size is not None and \
        config.virtual_pipeline_model_parallel_size > 1:
        vpp = config.virtual_pipeline_model_parallel_size
    else:
        vpp = None
    initialize_model_parallel(tensor_model_parallel_size=tp,
                              pipeline_model_parallel_size=pp,
                              virtual_pipeline_model_parallel_size=vpp)

    # init model
    # pylint: disable=W0613
    def model_provider_func(pre_process=True, post_process=True):
        network = RotaryEmbeddingNet(config)
        return network
    network = get_model(model_provider_func, config)
    if isinstance(network, nn.CellList) and len(network) == 1:
        network = network[0]
    network.set_train(True)

    # run model
    output = network(args.seq_length)
    print(f'Loss: {output.astype(ms.float32).abs().mean().asnumpy()}')


if __name__ == '__main__':
    args = parse_args()
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args)
    else:
        config = core_transformer_config_from_args(args)
    run_rotary_embedding(config)
