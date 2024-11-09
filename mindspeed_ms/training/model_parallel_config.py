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
"""Model parallel config."""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import mindspore.common.dtype as mstype
from mindspore._c_expression.typing import Float, BFloat

from mindspeed_ms.core.utils import DictWithValueError


_SUPPORT_DTYPE_DICT = DictWithValueError(
    {"float16": mstype.float16, "float32": mstype.float32, "bfloat16": mstype.bfloat16})


@dataclass
class ModelParallelConfig:
    """Parallel config class."""

    ###################
    # Model parallelism
    ###################
    tensor_model_parallel_size: int = 1
    """Intra-layer model parallelism. Splits tensors across GPU ranks."""

    pipeline_model_parallel_size: int = 1
    """Inter-layer model parallelism. Splits transformer layers across GPU ranks."""

    virtual_pipeline_model_parallel_size: Optional[int] = None
    """Interleaved pipeline parallelism is used to improve performance by reducing the pipeline
       bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
       The number of virtual blocks per pipeline model parallel rank is the virtual model parallel
       size.  See Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM:
       arxiv.org/pdf/2104.04473.pdf for more details.
    """

    sequence_parallel: bool = False
    """Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer norms
       and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer Models
       (https://arxiv.org/abs/2205.05198) for more details.
    """

    context_parallel_size: int = 1
    """Splits network input along sequence dimension across GPU ranks."""

    expert_model_parallel_size: int = 1
    """Distributes Moe Experts across sub data parallel dimension."""

    # Placeholder. Do not use
    moe_extended_tp: bool = False
    """Alternative parallelization strategy for expert parallelism. Instead of distributing experts
       across expert_model_parallel_size, each expert is sharded along extendended tensor parallel
       domain (tensor_model_paralle_size * expert_model_parallel_size). It avoids the load balancing
       problem with MOE training.
    """

    ###################
    # Initialization
    ###################
    # Placeholder. Do not use
    perform_initialization: bool = True
    """If true, weights are initialized. This option can be useful when you know you are going to
       load values from a checkpoint.
    """

    use_cpu_initialization: bool = False
    """When set to False, we initialize the weights directly on the GPU. CPU initialization is the
       same regardless of tensor model parallelism, but GPU initialization is not. Transferring
       weights from CPU to GPU can take a significant amount of time for large models.
    """

    ###################
    # Training
    ###################
    fp16: bool = False
    """If true, train with fp16 mixed precision training."""

    bf16: bool = False
    """If true, train with bf16 mixed precision training."""

    params_dtype: Union[Float, BFloat] = mstype.float32
    """dtype used when initializing the weights."""

    # Placeholder. Do not use
    timers: Callable = None
    """Timers object to call for various timing functions. See megatron.core.timers.Timers"""

    # Placeholder. Do not use
    grad_scale_func: Callable = None
    """If using loss scaling, this function should take the loss and return the scaled loss. If
       None, no function is called on the loss.
    """

    # Placeholder. Do not use
    no_sync_func: Callable = None
    """Function that creates a context that suppresses asynchronous data-parallel communication. If
       the model is an instance of core.distributed.DistributedDataParallel, the default is to use
       core.distributed.DistributedDataParallel.no_sync.
    """

    # Placeholder. Do not use
    grad_sync_func: Callable = None
    """Function that launches asynchronous gradient reductions (e.g. distributed optimizer gradient
       reduce-scatters). The function should take one argument: an iterable of parameters whose
       gradients are to be synchronized.
    """

    # Placeholder. Do not use
    param_sync_func: Callable = None
    """Function that launches asynchronous parameter synchronizations (e.g. distributed optimizer
       parameter all-gathers). The function should take one argument: an iterable of parameters to
       be synchronized.
    """

    deterministic_mode: bool = False
    """If true, code that has deterministic execution will be chosen. This usually
       means slower execution, but is good for debugging and testing."""

    # Placeholder. Do not use
    enable_autocast: bool = False
    """If true runs the forward step function inside torch.autocast context."""

    # Placeholder. Do not use
    autocast_dtype: Union[Float, BFloat] = None
    """dtype to pass to torch.amp.autocast when enabled. If None, is set to pipeline_dtype."""

    # Placeholder. Do not use
    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None
    """If int, set the number of microbatches where not all of the layers will be checkpointed and
       recomputed. The rest of the microbatches within the window of maximum outstanding
       microbatches will recompute all layers (either full recompute or selective recompute). If
       None, the checkpoint and recompute will be left up to the forward_step function.

    """

    # Additional argument
    compute_dtype: Union[Float, BFloat] = mstype.float32
    """Compute data type of linear module"""

    # Additional argument
    softmax_compute_dtype: Union[Float, BFloat] = mstype.float32
    """Compute data type of softmax layer"""

    # Additional argument
    zero_level: str = None
    """Zero level for ZeRO optimizer, if None, will not use ZeRO optimizer"""

    # Additional argument
    num_layer_list: list = None
    """User-defined pipeline parallel model layer division"""

    ###################
    # Optimizations
    ###################
    gradient_accumulation_fusion: bool = False
    """If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA extension
       fused_weight_gradient_mlp_cuda module. To use gradient_accumulation_fusion you must install
       APEX with --cpp_ext and --cuda_ext. For example: "pip install --global-option=\"--cpp_ext\"
       --global-option=\"--cuda_ext\" ". Note that the extension requires CUDA>=11. Otherwise, you
       must turn off gradient accumulation fusion.
    """

    # Placeholder. Do not use
    async_tensor_model_parallel_allreduce: bool = False
    """NOTE: Deprecated. This flag is ignored."""

    # Placeholder. Do not use
    use_te_rng_tracker: bool = False
    """If true, uses RNG state tracker in TransformerEngine if exists.
    """

    # Placeholder. Do not use
    tp_comm_overlap: bool = False
    """If true, allows overlapping of Linear layer execution with tensor parallel communication
       collectives like AllGather/ReduceScatter. Overlapping is done for the linear layers wherever
       possible during the forward and the backward pass.
    """

    # Placeholder. Do not use
    tp_comm_bulk_wgrad: bool = True
    """If true, allows All-Gather overlap with Bprop activation gradient GEMM. Don't care if
       tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_bulk_dgrad: bool = True
    """If true, allows Reduce-Scatter overlap with Bprop weight gradient GEMM. Don't care if
       tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_overlap_ag: bool = True
    """If true, allows All-Gather overlap with GEMM by pipelining the GEMM and All-Gather.
       Don't care if tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_overlap_rs: bool = True
    """If true, allows Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter.
       Don't care if tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_overlap_rs_dgrad: bool = False
    """If true, allows Reduce-Scatter overlap with DGRAD GEMM by pipelining the
       GEMM and Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_split_ag: bool = True
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather
       splits. Don't care if tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_atomic_ag: bool = False
    """Deprecated from TransformerEngine v1.6.0.
        If true, allows All-Gather overlap with Fprop GEMM by pipelining the GEMM and All-Gather both
       done atomically. Don't care if tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_split_rs: bool = True
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
       Reduce-Scatter splits. Don't care if tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    tp_comm_atomic_rs: bool = False
    """Deprecated from TransformerEngine v1.6.0.
       If true, allows Reduce-Scatter overlap with Fprop GEMM by pipelining the GEMM and
       Reduce-Scatter both done atomically. Don't care if tp_comm_overlap is False.
    """

    # Placeholder. Do not use
    cross_entropy_loss_fusion: bool = False
    """If this is enabled, the fused cross entropy implementation would be used.
       Defaults to False.
    """

    ###################
    # Pipeline Parallel
    ###################
    pipeline_dtype: Union[Float, BFloat] = None
    """dtype used in p2p communication, usually params_dtype"""

    variable_seq_lengths: bool = False
    """Support for variable sequence lengths across microbatches. Setting this communicates the size
        of tensors during pipeline parallelism communication, because of this extra overhead it
        should only be set if the sequence length varies by microbatch within a global batch.
    """

    overlap_p2p_comm: bool = False
    """When True some of the peer to peer communication for pipeline parallelism will overlap with
       computation. Must be False if batch_p2p_comm is true.
    """

    # Placeholder. Do not use
    batch_p2p_comm: bool = True
    """Use batch_isend_irecv instead of individual isend/irecv calls. Must be False if
       overlap_p2p_comm is True.
    """

    # Placeholder. Do not use
    batch_p2p_sync: bool = True
    """When using batch_isend_irecv, do a cuda.device.synchronize afterward to work around a bug in
       older version of PyTorch.
    """

    # Placeholder. Do not use
    use_ring_exchange_p2p: bool = False
    """Use custom ring_exchange kernel instead of torch.distributed.batch_isend_irecv(). Requires
       custom built torch with torch.distributed.ring_exchange.
    """

    # Placeholder. Do not use
    deallocate_pipeline_outputs: bool = False
    """If True, output data is deallocated after the tensor is sent to the next pipeline stage.
       Helps with saving memory, does nothing when pipeline parallel is not used.
    """

    # Placeholder. Do not use
    defer_embedding_wgrad_compute: bool = False
    """If true, defers the embedding WGRAD GEMMs while pipeline flush is
       taking place enabling us to hide pipeline flush latency. Defaults to False.
    """

    # Placeholder. Do not use
    wgrad_deferral_limit: int = 0
    """This value tunes the number of micro-batches for which the embedding weight gradient compute
       needs to be deferred to pipeline flush, this argument is invalid if `defer_embedding_wgrad_compute` is False.
       Defaults to 0, which means all micro-batches are deferred.
    """

    # Placeholder. Do not use
    pipeline_model_parallel_split_rank: Optional[int] = None
    """If int, rank where encoder and decoder should be split in cases where the model has both an
       encoder and decoder (e.g., T5). Ignored if None.
    """

    ###################
    # CPU Offloading
    ###################

    # Placeholder. Do not use
    cpu_offloading: bool = False
    """When set to True, all the activations are offloaded to the CPU asynchronously."""

    # Placeholder. Do not use
    cpu_offloading_num_layers: int = 0
    """Tells the number of transformer layers for which activations has to be offloaded."""

    # Placeholder. Do not use
    _cpu_offloading_context = None
    """For internal use only, do not set."""

    # Placeholder. Do not use
    cpu_offloading_activations: bool = True
    """If True, offloads the activations to CPU."""

    # Placeholder. Do not use
    cpu_offloading_weights: bool = True
    """If True, offloads the weights to CPU."""

    ###################
    # Timing
    ###################
    # Placeholder. Do not use
    barrier_with_L1_time: bool = True  # pylint: disable=C0103
    """If true, use barrier with level 1 time measurements. It is up to the user to make sure
       calling barrier with their timers will not result in hangs. This can happen if for example
       the user adds a level 1 timer that is not called by all ranks.
    """


    def _init_dtype(self, name, value):
        if value is None:
            return None
        if isinstance(value, str):
            return _SUPPORT_DTYPE_DICT[value]
        if not isinstance(value, (Float, BFloat)):
            raise TypeError(f"{name} type must be None, str or mstype, but got {type(value)}")
        return value


    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.sequence_parallel:
            if self.tensor_model_parallel_size <= 1:
                raise ValueError("Can not use sequence paralllelism without tensor parallelism")

        self.params_dtype = self._init_dtype("params_dtype", self.params_dtype)
        self.pipeline_dtype = self._init_dtype("pipeline_dtype", self.pipeline_dtype)
        self.compute_dtype = self._init_dtype("compute_dtype", self.compute_dtype)
        self.softmax_compute_dtype = self._init_dtype("softmax_compute_dtype", self.softmax_compute_dtype)
        self.autocast_dtype = self._init_dtype("autocast_dtype", self.autocast_dtype)

        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError(
                    "When using pipeline parallelism, pipeline_dtype must be specified"
                )

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype

        if self.defer_embedding_wgrad_compute and self.pipeline_model_parallel_size == 1:
            raise ValueError(
                "Cannot defer embedding wgrad compute when pipeline model parallel is not used"
            )

        if self.defer_embedding_wgrad_compute and not self.gradient_accumulation_fusion:
            raise ValueError(
                "Cannot defer embedding wgrad compute when gradient accumulation fusion is not used"
            )

        if self.defer_embedding_wgrad_compute and self.wgrad_deferral_limit < 0:
            raise ValueError(
                "Wgrad deferral limit should be greater than or equal to 0 when this optimization is enabled!"
            )

        if self.expert_model_parallel_size > 1 and self.tensor_model_parallel_size > 1:
            if self.sequence_parallel is False:
                raise ValueError(
                    "When using expert parallelism and tensor parallelism, sequence parallelism must be used"
                )


    def __str__(self):
        gap = 2 * " "
        attributes = vars(self)
        print_str = "\n" + self.__class__.__name__ + "\n"
        for name, val in attributes.items():
            new_str = str(val)
            new_str = new_str.replace("\n", "\n" + gap)
            print_str += f"{gap}{name}: {new_str}\n"

        return print_str
