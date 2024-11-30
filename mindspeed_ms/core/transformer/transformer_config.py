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
"""Transformer config."""

import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from mindspore.common.initializer import _INITIALIZER_ALIAS
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator

from mindspeed_ms.training.model_parallel_config import ModelParallelConfig
from mindspeed_ms.core.utils import DictWithValueError

_SUPPORT_INIT_METHOD = DictWithValueError(_INITIALIZER_ALIAS)


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Transformer config class"""

    ####################
    # model architecture
    ####################
    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    num_query_groups: int = None
    """Number of query groups for group query attention. If None, normal attention is used."""

    ffn_hidden_size: int = None
    """Transformer Feed-Forward Network hidden size. This is set to 4*hidden_size if not provided."""

    kv_channels: int = None
    """Projection weights dimension in multi-head attention. This is set to hidden_size //
    num_attention_heads if not provided."""

    hidden_dropout: float = 0.1
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.1
    """Post attention dropout probability."""

    fp32_residual_connection: bool = False
    """If true, move residual connections to fp32."""

    apply_residual_connection_post_layernorm: bool = False
    """If True, uses the original BERT residule connection ordering."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """If set to True, the LayerNorm is adjusted to center the gamma values around 0. This improves
    numerical stability."""

    add_bias_linear: bool = True
    """Include a bias term in all linear layers (QKV projections, after core attention, and two in
    MLP layer)."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    gated_linear_unit: bool = False
    """Use a gated linear unit for the first linear layer in the MLP."""

    activation_func: str = "gelu"
    """Activation function to use for the non-linearity in the MLP."""

    activation_func_fp8_input_store: bool = False
    """Store the input of MLP activation function in FP8 for backprop to save memory.
    The stored input is casted back to the original precision before backprop computation."""

    num_moe_experts: int = None
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    rotary_interleaved: bool = False
    """True is rotate pairs of even and odd dimensions (RoFormer style), False is rotate pairs of
    first half and second half (LLaMa style). Default to False."""

    window_size: Optional[Tuple[int, int]] = None
    """If not None, then will use sliding window attention. The size of the window is specified by
    the numbers inside the tuple; -1 is special value meaning "infinite window size"."""

    normalization: str = "LayerNorm"
    """Which norm to use for normalization layers, valid options are `LayerNorm` and `RMSNorm`."""

    qk_layernorm: bool = False
    """Whether to apply LayerNorm to the query and key embeddings."""

    test_mode: bool = False
    """Whether to run real-time tests."""

    calculate_per_token_loss: bool = False
    """Whether cross entropy loss is calculated over the actual number of non-padded tokens in the
    global batch, versus the default behavior of assuming all tokens are non-padded."""

    # Additional argument
    mask_func_type: str = "attn_mask_add"
    """Attention mask compute method"""

    # Additional argument
    grad_clip_kwargs: dict = None
    """Grad clip arguments"""

    ####################
    # initialization
    ####################
    init_method: Callable = None
    """Method to initialize weights. Note that bias is always set to zero. Should be a function that
    takes a single Tensor and initializes it. If None, will be set to
    megatron.core.utils.init_method_normal(init_method_std) which is torch nn init normal with
    mean=0.0 and std=init_method_std."""

    # Placeholder. Do not use
    output_layer_init_method: Callable = None
    """Method to initialize weights of the output layer of both attention and MLP blocks. If None,
    will be set to megatron.core.utils.scaled_init_method_normal(init_method_std) which is torch nn
    init normal with mean=0.0 and std=init_method_std / math.sqrt(2.0 * num_layers)."""

    init_method_std: float = 0.02
    """Standard deviation of the zero mean normal for the default initialization method, not used if
    init_method and output_layer_init_method are provided."""

    # Additional argument
    bias_init: Callable = None
    """Bias init method, default is zeros. additional parameter"""

    ####################
    # mixed-precision
    ####################
    apply_query_key_layer_scaling: bool = False
    """If true, scale Q * K^T by 1 / layer-number. This improve numeric stability when training with
    fp16."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32. This should be True if
    apply_query_key_layer_scaling is True."""

    ####################
    # fusion
    ####################
    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    # Placeholder. Do not use
    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel. This kernel only supports a fixed set
    of hidden sizes."""

    # Placeholder. Do not use
    memory_efficient_layer_norm: bool = False
    """If True, and using local layers (not from TransformerEngine), tells Apex to use the memory
    efficient fused LayerNorm kernel. Ignored if not using LayerNorm."""

    bias_dropout_fusion: bool = False
    """If True, uses bias dropout fusion."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    ####################
    # activation recomputation
    ####################
    recompute_granularity: str = None
    """Determines which type of activation recompute to use.  Megatron-core supports 'selective'
    activation checkpointing where only the memory intensive part of attention is checkpointed.
    These memory intensive activations are also less compute intensive which makes activation
    checkpointing more efficient for LLMs (20B+).  See Reducing Activation Recomputation in Large
    Transformer Models (https://arxiv.org/abs/2205.05198) for more details.  'full' will checkpoint
    the entire transformer layer.  If None, no recompute is performed and all activations are saved.
    If set, must be 'selective' or 'full'. 'selective' always uses all layers.
    """

    recompute_method: str = None
    """Determines which transformer layers will be recomputed. uniform will uniformly divide the
    total number of transformer layers in a transformer block and recompute the input activation of
    each divided chunk at the specified granularity.  block will recompute the input activations for
    only a set number of transformer layers per pipeline stage.  The rest of the layers in the
    pipeline stage will not have any activations recomputed.  If None, and recompute is enabled, all
    layers will do recomputation. If set, must be 'uniform' or 'block'."""

    recompute_num_layers: int = None
    """When recompute_method is uniform, recompute_num_layers is the number of transformer layers in
    each uniformly divided recompute unit.  When recompute_method is block, recompute_num_layers is
    the number of transformer layers to recompute within each pipeline stage.  Must be None for
    'selective' activation checkpointing."""

    distribute_saved_activations: bool = None
    """If True, distribute recomputed activations across the model parallel group."""

    # Additional argument
    recompute_config: dict = None
    """Recompute strateges"""

    # Additional argument
    select_comm_recompute: bool = False
    """Whether to select commouncation recompute"""

    # Additional argument
    select_recompute: bool = False
    """Whether to select recompute"""

    ####################
    # fp8 related
    ####################
    fp8: str = None
    """If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
    choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
    activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

    # Placeholder. Do not use
    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    # Placeholder. Do not use
    fp8_interval: int = 1
    """Controls how often the scaling factor is recomputed."""

    # Placeholder. Do not use
    fp8_amax_history_len: int = 1
    """The length of the amax history window used for scaling factor computation."""

    # Placeholder. Do not use
    fp8_amax_compute_algo: str = "most_recent"
    """Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
    predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
    always chooses the most recently seen value.

    """

    # Placeholder. Do not use
    fp8_wgrad: bool = True
    """When set to False, override FP8 config options and do the wgrad computation in higher precision."""

    # Placeholder. Do not use
    fp8_dot_product_attention: bool = False
    """When set to True, use the FP8 implementation of Dot Product Attention."""

    # Placeholder. Do not use
    fp8_multi_head_attention: bool = False
    """When set to True, use the FP8 implementation of Multi Head Attention."""

    ####################
    # MoE related
    ####################
    moe_router_load_balancing_type: str = "aux_loss"
    """Determines the load balancing strategy for the router. "aux_loss" corresponds to the load
    balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds to the balancing
    algorithm used in S-BASE, and "none" implies no load balancing."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    # Placeholder. Do not use
    moe_router_pre_softmax: bool = False
    """Enable pre-softmax routing for MoE, which means the top-k selection is before the softmax. By default, top-k is done after the softmax."""

    moe_grouped_gemm: bool = False
    """When there are multiple experts per rank, compress multiple local (potentially small) gemms
    in a single kernel launch to improve the utilization and performance by leveraging the Grouped
    GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).

    """

    # 1e-2 would be a good start value for load balance loss.
    moe_aux_loss_coeff: float = 0.0
    """Scaling coefficient for the aux loss. A starting value of 1e-2 is recommended."""

    moe_z_loss_coeff: float = None  # 1e-3 would be a good start value for z-loss
    """Scaling coefficient for the z-loss. A starting value of 1e-3 is recommended."""

    moe_input_jitter_eps: float = None
    """Add noise to the input tensor by applying jitter with a specified epsilon value."""

    # Placeholder. Do not use
    moe_token_dropping: bool = False
    """This feature involves selectively dropping and padding tokens for each expert to achieve a
    specified capacity, similar to GShard, Switch-Transformer, and DeepSpeed-MoE. Note that this is
    currently unsupported so should remain False."""

    moe_token_dispatcher_type: str = "allgather"
    """The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather' and 'alltoall'."""

    # Placeholder. Do not use
    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE, currently supports auxiliary loss and z loss."""

    moe_expert_capacity_factor: float = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token will be dropped. The default is None."""

    moe_pad_expert_input_to_capacity: bool = False
    """moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False."""

    moe_token_drop_policy: str = 'probs'
    """The policy to drop tokens. Can be either "probs" or "position". If "probs", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
    """

    # Placeholder. Do not use
    moe_layer_recompute: bool = False
    """Memory optimization: checkpointing moe_layer to save actiavtion memory."""

    ####################
    # miscellaneous
    ####################
    clone_scatter_output_in_embedding: bool = True
    """When set to True, clone the output of scatter_to_sequence_parallel_region in embedding layer
    to facilitate garbage collection of input."""

    # Placeholder. Do not use
    disable_parameter_transpose_cache: bool = False
    """When set to true, the parameter transposes are not cached for subsequent iterations."""

    # Placeholder. Do not use
    enable_cuda_graph: bool = False
    """When set to true, TransformerLayer blocks are wrapped with CUDA graph."""

    ####################
    # LoRA
    ####################

    # Additional argument
    use_lora: bool = False
    """Apply LoRA to the pretrain model, additional parameter"""

    # Additional argument
    lora_rank: int = 8
    """The dimension for LoRA modules, additional parameter"""

    # Additional argument
    lora_alpha: int = 32
    """The dimension for LoRA modules, additional parameter"""

    # Additional argument
    lora_dropout: float = 0.0
    """The dropout rate for LoRA, additional parameter"""

    # Additional argument
    lora_target_cells: list = None
    """The names of the cells to build LoRA modules. If 'use_lora' is
       True, this argument should at least contains a dict with the key 'targets_cells' and
       the value of names of the cells to apply LoRA. In addition, if you want to set special
       rank or alpha for cells in target_cells, you can add dict to the list.
       For example:
            case 1:
                target_cells = [
                  {'target_cells':[
                      '.*.qkv_proj'
                  ]},
              ]
            In this case, cells which name end with '.qkv_proj' will be applied LoRA.

            case 2:
                target_cells = [
                  {'target_cells':[
                      'backbone.layers.layers.0.attention.qkv_proj'
                  ]},
              ]
            In this case, the cell 'backbone.layers.layers.0.attention.qkv_proj' will be applied LoRA.

            case 3:
                [
                  {'target_cells':[
                      '.*.qkv_proj',
                  ]},
                  {'cell':'backbone.layers.layers.0.attention.qkv_proj', 'rank':4, 'alpha':16},
              ]
            In this case, cells which name end with '.qkv_proj' will be applied LoRA. In addition, the rank
            and alpha of the cell 'backbone.layers.layers.0.attention.qkv_proj' is 4 and 32, the rank and
            alpha of other cells are set to 'lora_rank' and 'lora_alpha', additional parameter.
    """

    # Additional argument
    lora_module: dict = None
    """LoRA module, additional parameter"""

    def update_lora_config(self, cell_name):
        """Update lora config"""
        lora_module = self.lora_module
        self.lora_module = None if lora_module is None else lora_module.get(
            cell_name, None)

    def _validate_lora_target_cells(self, target_cells):
        """Validate and assign lora_target_cells"""
        if target_cells is None:
            return target_cells

        # valid target_cells
        target_cells_defined = False
        for item in target_cells:
            if 'target_cells' in item.keys():
                if target_cells_defined:
                    raise ValueError("'target_cells' cannot not be defined more than once.")
                target_cells_defined = True
                Validator.check_value_type("target_cells", item['target_cells'], list)
                target_cells_lst = item['target_cells']
                if not target_cells_lst:
                    raise ValueError("for 'target_cells', the list of target_cells name must be set.")
        if not target_cells_defined:
            raise ValueError("for 'target_cells', the list of target_cells name must be set.")

        def _check_in_target_cells(cell_name):
            target_cell_found = False
            for target_key in target_cells_lst:
                match = re.match(target_key, cell_name)
                if match is not None and match.group() == cell_name:
                    return target_key
            return target_cell_found

        # valid rank and alpha for specific cells
        specific_lora_cell = []
        for item in target_cells:
            if 'cell' in item.keys():
                cell_name = item['cell']
                if not _check_in_target_cells(cell_name):
                    raise ValueError(
                        f"The cell need to set rank or alpha should be in the range "
                        f"defined by target_cells, but got name '{cell_name}'.")
                specific_lora_cell.append(item)
        return target_cells_lst, specific_lora_cell

    # pylint: disable=C0330, C0301, R1720, W0105
    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError(
                f'num_moe_experts must be non None to use expert-parallel.')

        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError(f'num_moe_experts must be non-negative.')

        if self.moe_expert_capacity_factor is not None:
            if self.moe_token_dispatcher_type != "alltoall":
                raise ValueError(
                    f'moe_expert_capacity_factor only works with alltoall token dispatcher'
                )
            if self.moe_expert_capacity_factor < 0:
                self.moe_expert_capacity_factor = None
            if self.moe_router_load_balancing_type not in ["aux_loss", "none"]:
                raise ValueError(
                    f'moe_expert_capacity_factor only works with aux_loss or none load balancing'
                )

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    f'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity'
                )

        if self.cpu_offloading and (
            self.cpu_offloading_num_layers < 0 or self.cpu_offloading_num_layers >= self.num_layers
        ):
            raise ValueError(
                f'CPU offloading can be done only for layers less than {self.num_layers}'
            )

        if self.cpu_offloading and self.pipeline_model_parallel_size > 1:
            raise ValueError(
                f'Currently there is no support for Pipeline parallelism with CPU offloading'
            )

        if self.cpu_offloading and self.recompute_granularity is not None:
            raise ValueError(
                f'CPU offloading does not work when activation recomputation is enabled'
            )

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
                )

            if self.recompute_method is not None:
                if self.recompute_method not in ['block', 'uniform']:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != 'selective':
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} '
                    f'so recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be between '
                    f'1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}'
                )
            elif (
                self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be None.'
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel}'
                )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        """
        if self.bias_activation_fusion:
            if self.activation_func not in ["gelu", "swiglu"]:
                raise ValueError(
                    "When bias_activation_fusion is True, activation function should be either gelu or swiglu"
                )
            if (
                self.activation_func == "gelu"
                and not self.gated_linear_unit
                and not self.add_bias_linear
            ):
                raise ValueError(
                    "When bias_activation_fusion is True, gated_linear_unit is False, "
                    "and activation function is gelu, add_bias_linear must also be True."
                )
        """
        if self.activation_func_fp8_input_store:
            if self.activation_func != "swiglu" or not self.gated_linear_unit:
                raise ValueError(
                    "Storing activation input in FP8 is supported only for SwiGLU.")
        if self.apply_rope_fusion and self.rotary_interleaved:
            raise ValueError(
                f'rotary_interleaved does not work with apply_rope_fusion.')

        if self.init_method is None:
            self.init_method = _SUPPORT_INIT_METHOD["normal"]()

        if self.bias_init is None:
            self.bias_init = _SUPPORT_INIT_METHOD["zeros"]()

        if self.output_layer_init_method is None:
            self.output_layer_init_method = "normal"

        """
        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )
        """

        if self.moe_extended_tp:
            if self.moe_token_dispatcher_type != 'allgather':
                raise ValueError(
                    "Moe extended TP parallelism only applies to allgather based token dispatcher."
                )
            extended_tp_size = self.tensor_model_parallel_size * self.expert_model_parallel_size
            if self.ffn_hidden_size % extended_tp_size != 0:
                raise ValueError(
                    f'ffn_hidden_size: {self.ffn_hidden_size} must be divisible by extended_tp_size {extended_tp_size}'
                )

        self.lora_target_cells = self._validate_lora_target_cells(self.lora_target_cells)
