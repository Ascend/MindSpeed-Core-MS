import torch
from torch.nn.functional import rms_norm, fast_gelu, swiglu
import mindspore as ms
from mindspore import ops
from mindspore.ops import auto_generate as gen
from mindspore.ops import rotary_position_embedding

from . import npu
from . import profiler


def npu_rms_norm(x, gamma, epsilon=1e-5):
    output = rms_norm(x, gamma, epsilon)
    return output


def npu_swiglu(x, dim=-1):
    return swiglu(x, dim)


def npu_rotary_position_embedding(x, cos, sin, mode=0):
    """
    Inputs:
        - **x** (Tensor) - The input tensor.
        - **cos** (Tensor) - The input cos tensor.
        - **sin** (Tensor) - The input sin tensor.
        - **mode** (int) - Optional mode value: 0 rotate half, 1 rotate interleaved.

    Outputs:
        - **y** (Tensor) - The output tensor.

    """
    return rotary_position_embedding(x, cos, sin, mode)


def npu_rotary_mul(x, cos, sin):
    """
    Inputs:
        - **x** (Tensor) - The input tensor.
        - **cos** (Tensor) - The input cos tensor.
        - **sin** (Tensor) - The input sin tensor.

    Outputs:
        - **y** (Tensor) - The output tensor.

    """
    return rotary_position_embedding(x, cos, sin, mode=0)


def npu_incre_flash_attention(query, key, value, *, padding_mask=None, atten_mask=None, pse_shift=None,
                              actual_seq_lengths=None, antiquant_scale=None, antiquant_offset=None, block_table=None,
                              dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None,
                              quant_offset2=None, kv_padding_size=None, num_heads=1, scale_value=1.0,
                              input_layout="BSH", num_key_value_heads=0, block_size=0, inner_precise=1):
    key = [key]
    value = [value]
    output = ops.incre_flash_attention(
        query, key, value, attn_mask=atten_mask, actual_seq_lengths=actual_seq_lengths, pse_shift=pse_shift,
        dequant_scale1=dequant_scale1, quant_scale1=quant_scale1, dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2, quant_offset2=quant_offset2, antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset, block_table=block_table, num_heads=num_heads, input_layout=input_layout,
        scale_value=scale_value, num_key_value_heads=num_key_value_heads, block_size=block_size,
        inner_precise=inner_precise, kv_padding_size=kv_padding_size
    )
    return output


def npu_prompt_flash_attention(query, key, value, *, pse_shift=None, padding_mask=None, atten_mask=None,
                               actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None,
                               quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0,
                               pre_tokens=2147473647, next_tokens=0, input_layout="BSH", num_key_value_heads=0,
                               actual_seq_lengths_kv=None, sparse_mode=0):

    output = ops.prompt_flash_attention(
        query, key, value, attn_mask=atten_mask, actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv, pse_shift=pse_shift, deq_scale1=deq_scale1,
        quant_scale1=quant_scale1, deq_scale2=deq_scale2, quant_scale2=quant_scale2, quant_offset2=quant_offset2,
        num_heads=num_heads, scale_value=scale_value, pre_tokens=pre_tokens, next_tokens=next_tokens,
        input_layout=input_layout, num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode, inner_precise=1
    )
    return output


def npu_fusion_attention(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
                         scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0,
                         drop_mask=None, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                         gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None):
    output = gen.flash_attention_score_impl(
        query, key, value, real_shift=pse, padding_mask=padding_mask, drop_mask=drop_mask,
        attn_mask=atten_mask, prefix=prefix, actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen, head_num=head_num, keep_prob=keep_prob,
        scale_value=scale, pre_tokens=pre_tockens, next_tokens=next_tockens,
        inner_precise=inner_precise, input_layout=input_layout, sparse_mode=sparse_mode
    )
    sfm_max, sfm_sum, _, atten_out = output

    return atten_out, sfm_max, sfm_sum


def npu_fusion_attention_grad(
        query, key, value, dy, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None,
        drop_mask=None, softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None, prefix=None,
        actual_seq_qlen=None, actual_seq_kvlen=None, keep_prob=1.0, scale_value=1.0,
        pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, sparse_mode=0,
        gen_mask_parallel=True, sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
        seed=1234, offset=0, numels=0
    ):
    output = gen.flash_attention_score_grad_impl(
        query, key, value, dy, pse_shift=pse, padding_mask=padding_mask, atten_mask=atten_mask, drop_mask=drop_mask,
        softmax_max=softmax_max, softmax_sum=softmax_sum, softmax_in=softmax_in, attention_in=attention_in,
        prefix=prefix, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen, head_num=head_num,
        keep_prob=keep_prob, scale_value=scale_value, pre_tokens=pre_tockens, next_tokens=next_tockens,
        inner_precise=inner_precise, input_layout=input_layout, sparse_mode=sparse_mode
    )
    dq, dk, dv, _ = output

    return dq, dk ,dv


adamw_opt = gen.ApplyAdamW()

def npu_apply_adam_w(beta1_power, beta2_power, lr, weight_decay, beta1, beta2,
                     epsilon, grad, max_grad_norm, amsgrad, maximize, out):

    var, m, v = out
    var, m, v = adamw_opt(var, m, v, beta1_power=beta1_power, beta2_power=beta2_power, lr=lr, weight_decay=weight_decay,
                          beta1=beta1, beta2=beta2, epsilon=epsilon, grad=grad, max_grad_norm=max_grad_norm,
                          amsgrad=amsgrad, maximize=maximize)
    return var, m, v


def npu_all_gather_base_mm(
        input_: ms.Tensor,
        x2: ms.Tensor,
        _: str,
        world_size: int,
        bias: None = None,
        gather_index: int = 0,
        gather_output: bool = True,
        comm_turn: int = 0,
    ) -> None:
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    return ms.ops.all_gather_matmul(
        input_,
        x2,
        group,
        world_size,
        bias=bias,
        gather_index=gather_index,
        gather_output=gather_output,
        comm_turn=comm_turn,
    )


def npu_mm_reduce_scatter_base(
        input_: ms.Tensor,
        x2: ms.Tensor,
        _: str,
        world_size: int,
        reduce_op: str = ops.ReduceOp.SUM,
        bias: None = None,
        comm_turn: int = 0,
    ) -> None:
    group = ms.communication.GlobalComm.WORLD_COMM_GROUP
    return ms.ops.matmul_reduce_scatter(
        input_,
        x2,
        group,
        world_size,
        reduce_op=reduce_op,
        bias=bias,
        comm_turn=comm_turn,
    )

__version__ = "2.5"


