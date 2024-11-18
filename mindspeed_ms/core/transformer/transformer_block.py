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
# =========================================================================
""" transformer block """

from dataclasses import dataclass
from typing import List, Optional, Union

from collections import OrderedDict
import numpy as np
from mindspore import Tensor, mint, nn, Parameter

from mindspeed_ms.tools import logger
from mindspeed_ms.core.transformer import Module
from mindspeed_ms.core.transformer import ModuleSpec, build_module
from mindspeed_ms.core.transformer import TransformerConfig
from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.legacy.model.utils import get_num_layer_list, get_layers_and_offset
from mindspeed_ms.core.transformer.transformer_layer import BaseTransformerLayer, NoopTransformerLayer
from mindspeed_ms.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_pipeline_model_parallel_world_size,
    get_virtual_pipeline_model_parallel_rank,
    get_virtual_pipeline_model_parallel_world_size
)

from mindspeed_ms.core.transformer.wrapped_layer_norm import WrappedLayerNorm

# pylint: disable=C0103
LayerNormImpl = WrappedLayerNorm


def get_num_layers_to_build(config: TransformerConfig, with_offset: bool = False) -> int:
    """get transformer layers nums for current rank"""
    args = get_args()
    standalone_embedding_stage = args.standalone_embedding_stage
    pp_stage = get_pipeline_model_parallel_world_size()
    pp_rank = get_pipeline_model_parallel_rank()
    if standalone_embedding_stage:
        pp_stage = pp_stage - 1
        pp_rank = pp_rank - 1
    vpp_stage = (get_virtual_pipeline_model_parallel_world_size()
                if get_virtual_pipeline_model_parallel_world_size() is not None else 1)
    vpp_rank = (get_virtual_pipeline_model_parallel_rank()
                if get_virtual_pipeline_model_parallel_world_size() is not None else 0)
    pp_split_num = pp_stage * vpp_stage
    if config.num_layers < pp_split_num:
        raise RuntimeError(f"The number of model layers is {config.num_layers}, "
                        f"but using pipeline parallel requires at least "
                        f"'pp({pp_stage}) * vpp({vpp_stage}) = {pp_split_num}' "
                        f"layers for splitting")
    if get_pipeline_model_parallel_world_size() > 1:
        if standalone_embedding_stage and get_pipeline_model_parallel_rank() == 0:
            num_layers = 0
            offset = 0
        else:
            num_layer_list = get_num_layer_list(config)
            num_layer_array = np.array(num_layer_list)

            if num_layer_array.sum() != config.num_layers:
                raise ValueError(f"The number of model layers is {config.num_layers}, "
                                f"but the sum of num_layer_list "
                                f"{num_layer_array} is {num_layer_array.sum()}.")
            if not np.all(num_layer_array > 0):
                raise ValueError(f"All elements of num_layer_list should be larger than 0, "
                                f"but got {num_layer_array}.")
            num_layers, offset = get_layers_and_offset(num_layer_array,
                                                        pp_stage, pp_rank,
                                                        get_virtual_pipeline_model_parallel_world_size(), vpp_rank)
            if get_virtual_pipeline_model_parallel_world_size() is not None:
                logger.info(
                    f"Custom num layer list is {num_layer_array}. "
                    f"Num_layers in vpp_rank:{vpp_rank}, pp_rank:{pp_rank} is {num_layers}.")
            else:
                logger.info(
                    f"Custom num layer list is {num_layer_array}. "
                    f"Num_layers in pp_rank:{pp_rank} is {num_layers}.")
    else:
        num_layers = config.num_layers
        offset = get_pipeline_model_parallel_rank() * num_layers

    if with_offset:
        return num_layers, offset
    else:
        return num_layers


@dataclass
class TransformerBlockSubmodules:
    layer_specs: List[ModuleSpec] = None
    layer_norm: Optional[Union[ModuleSpec, nn.Cell]] = None


def _get_block_submodules(
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        with_offset: bool = False
    ) -> TransformerBlockSubmodules:
    """ get the submodules for current block """
    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        raise NotImplementedError("Initialize TransformerBlock with TransformerBlockSubmodules "
                                    "is not supported for now.")
    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    if isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            raise NotImplementedError("Initialize TransformerBlock with TransformerBlock ModuleSpec "
                                        "is not supported for now.")
        if issubclass(spec.module, BaseTransformerLayer):
            if with_offset:
                num_layers, offset = get_num_layers_to_build(config, with_offset=with_offset)
                return TransformerBlockSubmodules(
                    layer_specs=[spec] * num_layers,
                    layer_norm=LayerNormImpl,
                ), offset
            else:
                num_layers = get_num_layers_to_build(config, with_offset=with_offset)
                return TransformerBlockSubmodules(
                    layer_specs=[spec] * num_layers,
                    layer_norm=LayerNormImpl,
                )
        raise Exception(f"specialize for {spec.module.__name__}.")
    raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(Module):
    """Transformer class."""

    def __init__(
            self,
            config: TransformerConfig,
            spec: Union[TransformerBlockSubmodules, ModuleSpec],
            post_layer_norm: bool = True,
            pre_process: bool = True,
            post_process: bool = True,
    ):
        super().__init__(config=config)
        args = get_args()

        self.submodules, self.offset = _get_block_submodules(config, spec, with_offset=True)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.num_layers = len(self.submodules.layer_specs)

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        seq_length = args.seq_length
        if config.sequence_parallel:
            seq_length = seq_length // get_tensor_model_parallel_world_size()

        self.set_hidden_states = None
        self.pipeline_parallel = get_pipeline_model_parallel_world_size() > 1
        if self.pipeline_parallel:
            batch_size = args.micro_batch_size
            hidden_states_shape = (seq_length, batch_size, config.hidden_size)
            self.set_hidden_states = Parameter(
                mint.zeros(
                    hidden_states_shape, dtype=config.compute_dtype
                ),
                requires_grad=False,
                name="set_hidden_states",
            )

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

    def _build_layers(self):
        """ Transformer layers. """
        def build_layer(layer_spec, layer_number):
            return build_module(
                layer_spec,
                config=self.config,
                layer_number=layer_number,
            )

        args = get_args()
        # ensure the Parameter of each rank init as correct name
        layers_dict = OrderedDict()
        if not hasattr(args, "parameters_id_map"):
            layer_str_dict = {}
        else:
            layer_str_dict = args.parameters_id_map
        if self.num_layers == 0:
            self.num_layers = 1
            layers_dict[str(0)] = NoopTransformerLayer(1)
        else:
            for i, layer_spec in enumerate(self.submodules.layer_specs):
                global_id = i + self.offset
                layer_str = f"{global_id}"
                pp_size = get_pipeline_model_parallel_world_size()
                vpp_size = get_virtual_pipeline_model_parallel_world_size()
                if pp_size is not None and pp_size > 1:
                    layer_str += f"_pp{get_pipeline_model_parallel_rank()}"
                if vpp_size is not None and vpp_size > 1:
                    layer_str += f"_vpp{get_virtual_pipeline_model_parallel_rank()}"
                if pp_size is not None and pp_size > 1:
                    layer_str += f"_id{i}"
                if self.config.noop_layers:
                    if i + self.offset in self.config.noop_layers:
                        layers_dict[layer_str] = build_layer(
                            ModuleSpec(module=NoopTransformerLayer),
                            i + 1 + self.offset,
                        )
                        continue
                layer_str_dict[str(global_id)] = layer_str
                layers_dict[layer_str] = build_layer(layer_spec, i + 1 + self.offset)
        self.layers = nn.SequentialCell(layers_dict)
        args.parameters_id_map = layer_str_dict

        if LayerNormImpl and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                LayerNormImpl,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

    def set_input_tensor(self, input_tensor):
        """
        In pipeline parallel, the receiving data from previous stage will be set into class.
        Construct function's input will be replace by self.set_hidden_states.
        """
        self.set_hidden_states.set_data(input_tensor, slice_shape=True)

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor,
            context: Tensor = None,
            context_mask: Tensor = None,
            rotary_pos_emb: Tensor = None,
            inference_params=None,
            packed_seq_params=None,
    ):
        """ construct """
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]
        if self.config.fp8:
            raise NotImplementedError("fp8 is not supported for now.")
        if context is not None:
            raise NotImplementedError("context is not supported for now.")
        if context_mask is not None:
            raise NotImplementedError("context_mask is not supported for now.")
        if inference_params is not None:
            raise NotImplementedError("inference_params is not supported for now.")
        if packed_seq_params is not None:
            raise NotImplementedError("packed_seq_params is not supported for now.")

        layers = self.layers

        # self.hidden_states instead of input
        if not self.pre_process:
            hidden_states = self.set_hidden_states.value()

        for layer in layers:
            hidden_states, context = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
            )

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
