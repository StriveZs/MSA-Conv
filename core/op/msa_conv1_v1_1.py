import torch
from torch.autograd import Function
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import math
import msa_conv1

"""
    实现任意卷积结构接口化
"""

class MSA_ConvF_1(Function):
    @staticmethod
    def forward(ctx,
                Dot: torch.Tensor,
                V: torch.Tensor,
                kernel_size_h=1,
                kernel_size_w=1,
                stride=1,
                dilation=1,
                padding_h=0,
                padding_w=0,
                im2col_step=128):
        if V is not None and V.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as V, got {V.dim()}D tensor \
                  instead.')

        ctx.stride_h, ctx.stride_w = _pair(stride)
        ctx.pad_h, ctx.pad_w = padding_h, padding_w
        ctx.kernel_h, ctx.kernel_w = kernel_size_h, kernel_size_w
        ctx.dilation_h, ctx.dilation_w = _pair(dilation)

        ctx.out_h = int(
            math.floor(
                torch.true_divide((V.size(2) + 2 * ctx.pad_h -
                                   (ctx.kernel_h - 1) - 1), ctx.stride_h) + 1))
        ctx.out_w = int(
            math.floor(
                torch.true_divide((V.size(3) + 2 * ctx.pad_w -
                                   (ctx.kernel_w - 1) - 1), ctx.stride_w) + 1))
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(Dot, V) # 保存张量，可以用backward

        output = V.new_empty((V.size(0), V.size(1), V.size(2), V.size(3)))
        ctx.bufs_ = [V.new_empty(0), V.new_empty(0)]  # columns, ones

        ctx.cur_im2col_step = min(ctx.im2col_step, V.size(0))
        #assert (V.size(0) % ctx.cur_im2col_step
        #       ) == 0, 'batch size must be divisible by im2col_step'
        if V.size(0) % ctx.cur_im2col_step != 0:
            ctx.cur_im2col_step = V.size(0)

        if Dot.is_contiguous() == False:
            Dot = Dot.contiguous()
        if V.is_contiguous()  == False:
            V = V.contiguous()
        #print('Dot:'+str(Dot.dtype))
        #print('V:'+str(V.dtype))
        msa_conv1.msa_conv1_forward(
            Dot,
            V,
            output,
            ctx.bufs_[0],
            kW=ctx.kernel_w,
            kH=ctx.kernel_h,
            dW=ctx.stride_w,
            dH=ctx.stride_h,
            padW=ctx.pad_w,
            padH=ctx.pad_h,
            dilationW=ctx.dilation_w,
            dilationH=ctx.dilation_h,
            im2col_step=ctx.cur_im2col_step)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None, ) * 9
        if grad_output.is_contiguous() == False:
            grad_output = grad_output.contiguous()
        Dot, V = ctx.saved_tensors # 读取张量
        grad_Dot = Dot.new_empty((Dot.size(0), Dot.size(1), Dot.size(2), Dot.size(3)), dtype=torch.float32)
        grad_V = V.new_empty((V.size(0), V.size(1), V.size(2), V.size(3)), dtype=torch.float32)
        ctx.bufs_ = [Dot.new_empty(0), V.new_empty(0), grad_output.new_empty(0)]
        if Dot.is_contiguous() == False:
            Dot = Dot.contiguous()
        if V.is_contiguous()  == False:
            V = V.contiguous()
        # print(Dot.dtype) # Dot的dtype为float16
        # print(V.dtype) # Dot的dtype为float16
        # print(grad_output.dtype) # grad_output的dtype为float16
        # 问题: RuntimeError: expected scalar type Float but found Half, 输入的tensor dtype为float16
        msa_conv1.msa_conv1_backward(
            Dot,
            V,
            grad_output,
            grad_Dot,
            grad_V,
            ctx.bufs_[0],
            ctx.bufs_[1],
            ctx.bufs_[2],
            kW=ctx.kernel_w,
            kH=ctx.kernel_h,
            dW=ctx.stride_w,
            dH=ctx.stride_h,
            padW=ctx.pad_w,
            padH=ctx.pad_h,
            dilationW=ctx.dilation_w,
            dilationH=ctx.dilation_h,
            im2col_step=ctx.cur_im2col_step)
        return grad_Dot, grad_V, None, None, None, None, None, None, None
    
    @staticmethod
    def backward_pytorch(ctx, grad_output):
        Dot, V = ctx.saved_tensors #[B, K, H, W], [B, C, H, W]
        b, c, h, w = V.shape
        if grad_output is None:
            return (None, ) * 6
        slide_window = nn.Unfold(kernel_size=(ctx.kernel_w, ctx.kernel_h), stride=1, padding= (ctx.kernel_w - 1) // 2)
        Unfold_Dot = slide_window(Dot) # [B, KK, HW]
        Unfold_V = slide_window(V) # [B, CK, HW]
        Unfold_Output = slide_window(grad_output)

        # grad_Dot
        grad_output_reshape = grad_output.unsqueeze(2) #[B, C, H, W] -> [B, C, 1, H, W]
        Unfold_V_reshape = Unfold_V.reshape(b, c, ctx.kernel_w * ctx.kernel_h, h, w) # [B, C, K, H, W]
        grad_Dot = torch.sum(torch.mul(grad_output_reshape, Unfold_V_reshape), dim=1)# [B, K, H, W]
        
        # grad_V
        Dot_reshape = Dot.reshape(b, ctx.kernel_h, ctx.kernel_w, h, w) # [B, K, H, W]
        Unfold_Output_reshape = Unfold_Output.reshape(b, c, ctx.kernel_h*ctx.kernel_w, h, w) # [B, C, K, H, W]
        Unfold_Dot_reshape = Unfold_Dot.reshape(b, ctx.kernel_h, ctx.kernel_w, ctx.kernel_h, ctx.kernel_w, h, w) # [B, K, K, K, K, H, W]
        Unfold_Dot_Mul = torch.zeros_like(Dot_reshape) # [B, K, K, H, W]
        # 这个是卷积公式的变形版本, 从[B,K,K,K,K,H,W]中取出K×K个[B,H,W],
        # 对应关系为: ctx.kernel_h-1-i, ctx.kernel_w-1-j, i, j 对应的是 i,j
        for i in range(ctx.kernel_h):
            for j in range(ctx.kernel_w):
                Unfold_Dot_Mul[:, i, j, :, :] = Unfold_Dot_reshape[:, ctx.kernel_h-1-i, ctx.kernel_w-1-j, i, j, :, :]
        # 取出来之后的结果点积相加就好了
        Unfold_Dot_Mul = Unfold_Dot_Mul.reshape(b, ctx.kernel_h*ctx.kernel_w, h, w).unsqueeze(1)
        grad_V = torch.mul(Unfold_Dot_Mul, Unfold_Output_reshape).sum(2)
        return grad_Dot, grad_V, None, None, None, None # 这个对应的input的输入参数数量，如果不需要返回梯度，则用None补齐


msaconv_op_1 = MSA_ConvF_1.apply


class MSA_Conv2d_1_v1(nn.Module):

    def __init__(self,
                 kernel_size: int or tuple,
                 dilation: int = 1,
                 im2col_step: int = 32):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size_h = kernel_size
            self.kernel_size_w = kernel_size
        else:
            self.kernel_size_h = kernel_size[0]
            self.kernel_size_w = kernel_size[1]
        self.stride = 1
        self.dilation = dilation
        self.padding_h = dilation * (self.kernel_size_h - 1) // 2
        self.padding_w = dilation * (self.kernel_size_w - 1) // 2
        # print(self.dilation, self.padding)
        # import pdb;pdb.set_trace()
        self.im2col_step = im2col_step

    def forward(self,
                Dot: torch.Tensor,
                V: torch.Tensor,) -> torch.Tensor:
        return msaconv_op_1(Dot, V, self.kernel_size_h, self.kernel_size_w, self.stride, self.dilation,
                                 self.padding_h, self.padding_w, self.im2col_step)