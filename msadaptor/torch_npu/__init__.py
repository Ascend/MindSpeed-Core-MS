from . import npu
from . import profiler
from torch.nn.functional import rms_norm, fast_gelu, swiglu
import mindspore as ms
from mindspore import ops
from mindspore.ops import auto_generate as gen


def npu_rms_norm(x, gamma, epsilon=1e-5):
    output = rms_norm(x, gamma, epsilon)
    return output


def npu_swiglu(x, dim=-1):
    return swiglu(x, dim)


from mindspore.ops import rotary_position_embedding


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


# def npu_fusion_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None,
#                          scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0,
#                          prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
#                          gen_mask_parallel=True, sync=False):
#     output = ops.flash_attention_score(query, key, value, real_shift=pse, padding_mask=padding_mask,
#                                        attn_mask=atten_mask, prefix=prefix, actual_seq_qlen=actual_seq_qlen,
#                                        actual_seq_kvlen=actual_seq_kvlen, head_num=head_num, keep_prob=keep_prob,
#                                        scalar_value=scale, pre_tokens=pre_tockens, next_tokens=next_tockens,
#                                        inner_precise=inner_precise, input_layout=input_layout, sparse_mode=sparse_mode)

#     return (output,)


from mindspore.ops.operations.nn_ops import FlashAttentionScore


def npu_fusion_attention(query, key, value, head_num, input_layout, *,
                         pse=None, padding_mask=None, atten_mask=None, scale=1.,
                         keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647,
                         inner_precise=0, prefix=None, actual_seq_qlen=None,
                         actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True,
                         sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None):
    """
    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.
        - **softmax_max** (Tensor[float32]) - (B, N1, S1, 8) when input_layout is not `TND` else (T1, N1, D)
        - **softmax_sum** (Tensor[float32]) - (B, N1, S1, 8) when input_layout is not `TND` else (T1, N1, D)
    """
    fa_op = FlashAttentionScore(head_num=head_num,
                                scale_value=scale,
                                keep_prob=keep_prob,
                                pre_tokens=pre_tockens,
                                next_tokens=next_tockens,
                                inner_precise=inner_precise,
                                input_layout=input_layout,
                                sparse_mode=sparse_mode)
    # create drop_mask.
    enable_dropout = keep_prob < 1.0
    drop_mask_bits = None
    if input_layout != "TND":
        if input_layout == "BSH":
            bsz, q_seq_len, _ = query.shape
            _, kv_seq_len, _ = key.shape
        elif input_layout == "SBH":
            q_seq_len, bsz, _ = query.shape
            kv_seq_len, _, _ = key.shape
        elif input_layout == "BNSD":
            bsz, _, q_seq_len, _ = query.shape
            _, _, kv_seq_len, _ = key.shape
        elif input_layout == "BSND":
            bsz, q_seq_len, _, _ = query.shape
            _, kv_seq_len, _, _ = key.shape
        else:
            raise ValueError(f"input_layout is invalid")
        if enable_dropout:
            drop_gen_mask = DropoutGenMask()
            keep_prob_tensor = Tensor(keep_prob, dtype=mstype.float16)
            drop_mask_bits = F.reshape(
                drop_gen_mask((bsz, head_num, q_seq_len, kv_seq_len), keep_prob_tensor),
                (bsz, head_num, q_seq_len, kv_seq_len // 8))
    out = fa_op(query, key, value,
                real_shift=pse,
                drop_mask=drop_mask_bits,
                padding_mask=padding_mask,
                attn_mask=atten_mask,
                prefix=prefix,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen)
    sfm_max, sfm_sum, sfm_out, atten_out = out
    outputs = []
    outputs.append(atten_out)
    outputs.append(sfm_max)
    outputs.append(sfm_sum)
    # return atten_out, sfm_max, sfm_sum
    return outputs


from mindspore.ops.auto_generate import FlashAttentionScoreGrad


def npu_fusion_attention_grad(query, key, value, dy, head_num, input_layout, *,
                              pse=None, padding_mask=None, atten_mask=None, scale=1.,
                              keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647,
                              inner_precise=0,
                              softmax_max=None, softmax_sum=None, softmax_in=None, attention_in=None,
                              prefix=None, actual_seq_qlen=None,
                              actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True,
                              sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None,
                              seed=None, offset=None, numels=None):
    """
    Outputs:
        - **dq** (Tensor[float16, bfloat16]) - The gradient of the Query vector.
        - **dk** (Tensor[float16, bfloat16]) - The gradient of the Key vector.
        - **dv** (Tensor[float16, bfloat16]) - The gradient of the Value vector.
    """
    fag_op = FlashAttentionScoreGrad(head_num=head_num,
                                     scale_value=scale,
                                     keep_prob=keep_prob,
                                     pre_tokens=pre_tockens,
                                     next_tokens=next_tockens,
                                     inner_precise=inner_precise,
                                     input_layout=input_layout,
                                     sparse_mode=sparse_mode)
    # create drop_mask.
    enable_dropout = keep_prob < 1.0
    drop_mask_bits = None
    if input_layout != "TND":
        if input_layout == "BSH":
            bsz, q_seq_len, _ = query.shape
            _, kv_seq_len, _ = key.shape
        elif input_layout == "SBH":
            q_seq_len, bsz, _ = query.shape
            kv_seq_len, _, _ = key.shape
        elif input_layout == "BNSD":
            bsz, _, q_seq_len, _ = query.shape
            _, _, kv_seq_len, _ = key.shape
        elif input_layout == "BSND":
            bsz, q_seq_len, _, _ = query.shape
            _, kv_seq_len, _, _ = key.shape
        else:
            raise ValueError(f"input_layout is invalid")
        if enable_dropout:
            drop_gen_mask = DropoutGenMask()
            keep_prob_tensor = Tensor(keep_prob, dtype=mstype.float16)
            drop_mask_bits = F.reshape(
                drop_gen_mask((bsz, head_num, q_seq_len, kv_seq_len), keep_prob_tensor),
                (bsz, head_num, q_seq_len, kv_seq_len // 8))
    out = fag_op(query, key, value,
                 dy,
                 pse_shift=pse,
                 drop_mask=drop_mask_bits,
                 padding_mask=padding_mask,
                 atten_mask=atten_mask,
                 softmax_max=softmax_max,
                 softmax_sum=softmax_sum,
                 softmax_in=softmax_in,
                 attention_in=attention_in,
                 prefix=prefix,
                 actual_seq_qlen=actual_seq_qlen,
                 actual_seq_kvlen=actual_seq_kvlen)
    dq, dk, dv, dpse = out
    outputs = []
    outputs.append(dq)
    outputs.append(dk)
    outputs.append(dv)

    # return dq, dk, dv, dpse
    return outputs


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
