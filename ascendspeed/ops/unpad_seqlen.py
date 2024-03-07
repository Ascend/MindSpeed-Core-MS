import os
import json
import torch
import torch_npu
import numpy as np
import time
from ascendspeed.op_builder import PadSeqLenOpBuilder
from ascendspeed.op_builder import UnpadSeqLenOpBuilder
torch.set_printoptions(profile="full")

class UnpadSeqLenFunction(torch.autograd.Function):

    ascendspeed_ops_pad = PadSeqLenOpBuilder().load()
    ascendspeed_ops_unpad = UnpadSeqLenOpBuilder().load()

    @staticmethod
    def forward(ctx, input_tensor, seqlen, maxseqlen):
        """unpad forward function
        
        Args:
            input_tensor(tensor): float16, The input matrix, shape（batch,max_seq_len,hidden_size）
            seqlen(list): int32, shape param
            maxseqlen: int32, shape param

        Return:
            output_tensor(tensor): float16, The output matrix, shape（sum(seq_len),hidden_size）
        """
        ctx.maxseqlen = maxseqlen
        ctx.seqlen = seqlen
        qseqlen = torch.tensor(seqlen, device=input_tensor.device, dtype=torch.int32)
        output_tensor = UnpadSeqLenFunction.ascendspeed_ops_unpad.npu_unpad_seqlen(input_tensor, seqlen, maxseqlen)
        return output_tensor

    @staticmethod
    def backward(ctx, input_tensor):
        """unpad backward function
        
        Args:
            input_tensor(tensor): float16, The input matrix, shape（sum(seq_len),hidden_size）
            seq_len(list): int32, shape param
            maxseqlen: int32, shape param

        Return:
            output_tensor(tensor): float16, The output matrix, shape（batch,max_seq_len,hidden_size）
        """
        output_tensor = UnpadSeqLenFunction.ascendspeed_ops_pad.npu_pad_seqlen(input_tensor.contiguous(), ctx.seqlen, ctx.maxseqlen)
        return output_tensor, None, None

class UnpadSeqLen(torch.nn.Module):
    def __init__(self, maxseqlen):
        super(UnpadSeqLen, self).__init__()
        self.maxseqlen = maxseqlen

    def forward(self, input_tensor, seqlen):
        output_tensor = UnpadSeqLenFunction.apply(input_tensor, seqlen, self.maxseqlen)
        return output_tensor