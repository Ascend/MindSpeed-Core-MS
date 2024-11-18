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
"""GPT Layer Specs"""

from mindspeed_ms.core.fusions.fused_bias_dropout import get_bias_dropout_add
from mindspeed_ms.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from mindspeed_ms.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from mindspeed_ms.core.transformer.dot_product_attention import DotProductAttention
from mindspeed_ms.core.transformer.enums import AttnMaskType
from mindspeed_ms.core.transformer.identity_op import IdentityOp
from mindspeed_ms.core.transformer.mlp import MLP, MLPSubmodules
from mindspeed_ms.core.transformer import ModuleSpec
from mindspeed_ms.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from mindspeed_ms.core.transformer.wrapped_layer_norm import WrappedLayerNorm


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
        num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
    ) -> ModuleSpec:
    """get gpt layer with transformer engine_spec."""

    raise NotImplementedError("Transformer engine is not implemented for now in core mode."
                              "args.transformer_impl should set as 'local'")


# Use this spec for an implementation using only modules in mindspeed_ms core
def get_gpt_layer_local_spec(
        num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
    ) -> ModuleSpec:
    """get gpt layer local spec."""

    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=WrappedLayerNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=WrappedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=WrappedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=WrappedLayerNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
# pylint: disable=W0613,R1705
def _get_mlp_module_spec(
        use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
    ) -> ModuleSpec:
    """get mlp module spec."""
    if use_te:
        raise NotImplementedError("Transformer engine is not implemented for now in mlp."
                                  "args.transformer_impl should set as 'local'")

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules.
        raise NotImplementedError("Mixture of experts with modules is not supported for now in core mode.")
