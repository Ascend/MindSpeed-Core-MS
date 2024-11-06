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
"""Test ParallelRandom"""
import sys

import mindspore.common.dtype as mstype
import pytest

from mindspeed_ms.training import (
    parse_args,
    get_args,
    core_transformer_config_from_args,
    core_transformer_config_from_yaml
)


VERIFY_YAML_DICT = {
    "num_layers": 24,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_query_groups": None,
    "ffn_hidden_size": 2688,
    "kv_channels": 64,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "fp32_residual_connection": False,
    "apply_residual_connection_post_layernorm": False,
    "layernorm_epsilon": 1.e-5,
    "layernorm_zero_centered_gamma": True,
    "add_bias_linear": False,
    "bias_activation_fusion": False,
    "add_qkv_bias": False,
    "gated_linear_unit": False,
    "activation_func": "swiglu",
    "num_moe_experts": None,
    "rotary_interleaved": False,
    "window_size": None,
    "init_method": None,
    "init_method_std": 0.02,
    "output_layer_init_method": None,
    "apply_query_key_layer_scaling": False,
    "attention_softmax_in_fp32": False,
    "bias_swiglu_fusion": True,
    "masked_softmax_fusion": True,
    "persist_layer_norm": False,
    "memory_efficient_layer_norm": False,
    "bias_dropout_fusion": True,
    "apply_rope_fusion": True,
    "recompute_granularity": None,
    "recompute_method": None,
    "recompute_num_layers": None,
    "distribute_saved_activations": None,
    "fp8": None,
    "clone_scatter_output_in_embedding": True,
    "normalization": "LayerNorm",
    "moe_router_load_balancing_type": "aux_loss",
    "moe_router_topk": 2,
    "moe_grouped_gemm": False,
    "moe_aux_loss_coeff": 0,
    "moe_z_loss_coeff": None,
    "moe_input_jitter_eps": None,
    "moe_token_dropping": False,
    "tensor_model_parallel_size": 1,
    "context_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "virtual_pipeline_model_parallel_size": None,
    "sequence_parallel": False,
    "expert_model_parallel_size": 1,
    "use_cpu_initialization": True,
    "fp16": False,
    "bf16": True,
    "pipeline_dtype": None,
    "params_dtype": mstype.bfloat16,
    "compute_dtype": mstype.bfloat16,
    "untie_embeddings_and_output_weights": True,
    "position_embedding_type": "rope",
    "rotary_percent": 0.5,
    "transformer_impl": "local",
    "use_flash_attn": False,
    "seed": 1234,
    "optimizer": "adam",
    "lr": 2.5e-4,
    "lr_decay_style": "cosine",
    "lr_decay_iters": None,
    "lr_decay_samples": 255126953,
    "lr_warmup_fraction": None,
    "lr_warmup_iters": 0,
    "lr_warmup_samples": 81381,
    "lr_warmup_init": 0.0,
    "min_lr": 2.5e-5,
    "weight_decay": 0.1,
    "start_weight_decay": 0.1,
    "end_weight_decay": 0.1,
    "weight_decay_incr_style": "constant",
    "clip_grad": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
}


VERIFY_CONFIG_DICT = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
    "expert_model_parallel_size": 1,
    "sequence_parallel": False,
    "pipeline_dtype": mstype.float32,
    "params_dtype": mstype.float32,
    "compute_dtype": mstype.float32,
    "softmax_compute_dtype": mstype.float32,
    "num_layers": 4,
    "num_attention_heads": 4,
    "num_query_groups": 4,
    "hidden_size": 64,
    "ffn_hidden_size": 256,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "init_method": 'normal',
    "add_qkv_bias": True,
    "add_out_proj_bias": False,
    "add_mlp_bias": True,
    "mask_func_type": "attn_mask_add",
    "normalization": "FusedRMSNorm",
    "layernorm_epsilon": 0.00001,
    "activation_func": "gelu",
    "num_moe_experts": 1,
    "masked_softmax_fusion": False,
    "bias_dropout_fusion": False,
    "clone_scatter_output_in_embedding": False,
    "attention_softmax_in_fp32": True,
    "apply_rope_fusion": False,
    "add_bias_linear": False,
}


def extra_args_provider(parser):
    parser.add_argument('--extra-flag', type=str, default='')
    return parser


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
class TestConfig:
    """A test class for testing config."""

    def test_get_args_before_parse(self):
        error = False
        try:
            _ = get_args()
        except AssertionError as e:
            assert str(e) == "global arguments is not initialized."
            error = True
        assert error


    def test_parse_args_from_yaml(self):
        """test parse args from yaml"""
        yaml_file = "gpt_config.yaml"
        sys.argv = ['test_config.py', '--yaml-cfg', yaml_file]
        args = parse_args()
        config = core_transformer_config_from_yaml(args)

        for key, val in VERIFY_YAML_DICT.items():
            assert hasattr(args, key), f"args has not attr {key}"
            assert val == getattr(args, key), f"args.{key} = {getattr(args, key)}, not equal to {val}"

        assert config.pipeline_dtype == mstype.bfloat16
        assert config.deallocate_pipeline_outputs
        assert config.gated_linear_unit
        assert config.bias_activation_fusion

        # complex struct verify
        assert config.fa_config.input_layout == "BSH"
        assert config.fa_config.nest.a == 1
        assert config.fa_config.nest.b == 2
        assert config.num_layer_list == [1, 2, 3]


    def test_parse_args_from_command_line(self):
        """test parse args from command line"""
        sys.argv = [
            "test_config.py",
            "--tensor-model-parallel-size", "1",
            "--pipeline-model-parallel-size", "1",
            "--context-parallel-size", "1",
            "--expert-model-parallel-size", "1",
            "--no-rope-fusion",
            "--normalization", "FusedRMSNorm",
            "--num-layers", "4",
            "--num-attention-heads", "4",
            "--num-query-groups", "1",
            "--hidden-size", "64",
            "--ffn-hidden-size", "256",
            "--hidden-dropout", "0.0",
            "--attention-dropout", "0.0",
            "--num-experts", "1",
            "--add-qkv-bias",
            "--disable-out-proj-bias",
            "--no-masked-softmax-fusion",
            "--no-bias-dropout-fusion",
            "--no-clone-scatter-output-in-embedding",
            "--attention-softmax-in-fp32",
            "--disable-bias-linear",
            "--seq-length", "32",
            "--vocab-size", "128",
            "--position-embedding-type", "rope",
            "--max-position-embeddings", "4096",
            "--epochs", "1",
            "--train-iters", "2",
            "--global-batch-size", "8",
            "--micro-batch-size", "1",
            "--clip-grad", "0.0",
            "--train-samples", "0",
            "--transformer-impl", "local",
            "--disable-post-norm",
            "--initial-loss-scale", "1",
            "--log-interval", "1",
            "--extra-flag", "test",
            "--absent-flag", "aaa"
        ]

        args_default = {
            "epochs": 333,
            "a": 1,
            "b": 2
        }

        args = parse_args(
            extra_args_provider=extra_args_provider,
            args_defaults=args_default,
            ignore_unknown_args=True
        )
        config = core_transformer_config_from_args(args)

        assert not hasattr(args, "absent_flag")
        assert args.extra_flag == "test"
        assert args.epochs == 1
        assert args.a == 1
        assert args.b == 2

        for key, val in VERIFY_CONFIG_DICT.items():
            assert getattr(config, key) == val, f"{getattr(config, key)} != {val}"


    def test_parse_illegal_yaml(self):
        """test parse args illegal yaml file"""
        yaml_file = "illegal.yaml"
        sys.argv = ['test_config.py', '--yaml-cfg', yaml_file]
        error = False
        try:
            _ = parse_args()
        except TypeError as _:
            error = True
        assert error
