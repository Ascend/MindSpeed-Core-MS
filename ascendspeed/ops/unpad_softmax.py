import torch
import torch_npu
from ascendspeed.op_builder import UnpadSoftMaxOpBuilder

class UnpadSoftMaxFunction(torch.autograd.Function):

    ascendspeed_ops = UnpadSoftMaxOpBuilder().load()

    @staticmethod
    def forward(ctx, data_input, seq_len, head_num):
        """unpad softmax forward function
        
        Args:
            data_input(tensor): float16, The input matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
            seq_len(list): int32, shape param
            head_num: int32, shape param

        Return:
            data_output(tensor): float16, The output matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
        """
        data_output = UnpadSoftMaxFunction.ascendspeed_ops.npu_unpad_softmax(data_input, seq_len, head_num)
        ctx.save_for_backward(data_output)
        ctx.seq_len = seq_len
        ctx.head_num = head_num
        return data_output

    @staticmethod
    def backward(ctx, y_grad):
        """Unpad softmax backward function
        
        Args:
            y_input(tensor): float16, The input matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
            y_grad(tensor): float16, The input matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
            seq_len(list): int32, shape param
            head_num: int32, shape param

        Return:
            x_grad(tensor): float16, The output matrix, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）
        """
        y_input, = ctx.saved_tensors
        x_grad = UnpadSoftMaxFunction.ascendspeed_ops.npu_unpad_softmax_grad(y_input, y_grad, ctx.seq_len, ctx.head_num)
        return x_grad, None, None

class UnpadSoftMax(torch.nn.Module):
    def __init__(self):
        super(UnpadSoftMax, self).__init__()

    def forward(self, x: torch.Tensor, seq_len: list, head_num: int):
        return UnpadSoftMaxFunction.apply(x, seq_len, head_num)
