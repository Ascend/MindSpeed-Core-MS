import os
import json
import torch
import torch_npu
import time
from ascendspeed.op_builder import UnpadGenAttentionMaskOpBuilder

class UnpadGenAttentionMask(torch.nn.Module):

    ascendspeed_ops = UnpadGenAttentionMaskOpBuilder().load()

    def __init__(self, num_attention_heads_per_partition):
        super(UnpadGenAttentionMask, self).__init__()
        self.num_attention_heads_per_partition = num_attention_heads_per_partition

    def forward(self, attention_mask, seq_lengths):
        """gen attention mask function
        
        Args:
            attention_mask(tensor): float16, The input matrix, shape（batch,1,max_seq_len,max_seq_len）
            seq_lengths(list): int32, shape param
            head_num: int32, shape param

        Return:
            output(tensor): float16, The output matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
        """
        input_mask = attention_mask.half() #  self.input_tensor

        headNum = self.num_attention_heads_per_partition
        output_size = sum([s**2 for s in seq_lengths]) * headNum

        out_attention_mask = UnpadGenAttentionMask.ascendspeed_ops.npu_unpad_gen_attention_mask(input_mask, seq_lengths, headNum)
        output = out_attention_mask > 0.5
        return output
