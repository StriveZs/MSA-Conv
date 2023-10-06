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

class MSA_ConvF(Function):
    @staticmethod
    def forward(ctx,
                Q: torch.Tensor,
                K: torch.Tensor,
                kernel_size_h=3,
                kernel_size_w=3,
                stride=1,
                dilation=1,
                padding_h=0,
                padding_w=0,
                im2col_step=128,):
        if K is not None and K.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as K, got {K.dim()}D tensor \
                  instead.')
        assert Q.dim() == K.dim()

        ctx.stride_h, ctx.stride_w = _pair(stride)
        ctx.pad_h, ctx.pad_w = padding_h, padding_w
        ctx.kernel_h, ctx.kernel_w = kernel_size_h, kernel_size_w
        ctx.dilation_h, ctx.dilation_w = _pair(dilation)

        ctx.out_h = int(
            math.floor(
                torch.true_divide((K.size(2) + 2 * ctx.pad_h -
                                   (ctx.kernel_h - 1) - 1), ctx.stride_h) + 1))
        ctx.out_w = int(
            math.floor(
                torch.true_divide((K.size(3) + 2 * ctx.pad_w -
                                   (ctx.kernel_w - 1) - 1), ctx.stride_w) + 1))
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(Q, K)

        output = K.new_empty((K.size(0), ctx.kernel_h*ctx.kernel_w, K.size(2), K.size(3)))
        ctx.bufs_ = [K.new_empty(0), K.new_empty(0)]  # columns, ones

        ctx.cur_im2col_step = min(ctx.im2col_step, K.size(0))
        #assert (K.size(0) % ctx.cur_im2col_step
        #         ) == 0, 'batch size must be divisible by im2col_step'
        if K.size(0) % ctx.cur_im2col_step != 0:
            ctx.cur_im2col_step = K.size(0)

        if Q.is_contiguous() == False:
            Q = Q.contiguous()
        if K.is_contiguous()  == False:
            K = K.contiguous()
        #print('Q:'+str(Q.dtype))
        #print('K:'+str(K.dtype))
        msa_conv1.msa_conv_forward(
            Q,
            K,
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
        Q, K = ctx.saved_tensors
        grad_Q = Q.new_empty((Q.size(0), Q.size(1), Q.size(2), Q.size(3)), dtype=torch.float32)
        grad_K = Q.new_empty((K.size(0), K.size(1), K.size(2), K.size(3)), dtype=torch.float32)
        ctx.bufs_ = [Q.new_empty(0), K.new_empty(0), grad_output.new_empty(0)]
        if Q.is_contiguous() == False:
            Q = Q.contiguous()
        if K.is_contiguous()  == False:
            K = K.contiguous()
        msa_conv1.msa_conv_backward(
            Q,
            K,
            grad_output,
            grad_Q,
            grad_K,
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
        return grad_Q, grad_K, None, None, None, None, None, None, None
        
    @staticmethod
    def backward_python(ctx, grad_output):
        Q, K = ctx.saved_tensors
        b, c, h, w = Q.shape
        if grad_output is None:
            return (None, ) * 6
        slide_window = nn.Unfold(kernel_size=(ctx.kernel_w, ctx.kernel_h), stride=1, padding= (ctx.kernel_w - 1) // 2)
        Unfold_K = slide_window(K) # [B, CK, HW]
        Unfold_Q = slide_window(Q) # [B, CK, HW]
        Unfold_Output = slide_window(grad_output)  # [B, KK, HW ]

        # grad_Q calculation
        grad_output_reshape = grad_output.unsqueeze(1) #[B, K, H, W] -> [B, 1, K, H, W]
        Unfold_K_reshape = Unfold_K.reshape(b, c, ctx.kernel_w * ctx.kernel_h, h, w) # [B, C, K, HW]
        grad_Q = torch.sum(torch.mul(grad_output_reshape, Unfold_K_reshape), dim=2)# [B, C, H, W]
        
        # grad_K
        grad_output_reshape = grad_output.reshape(b, ctx.kernel_h, ctx.kernel_w, h, w)
        Unfold_Q_reshape = Unfold_Q.reshape(b, c, ctx.kernel_h*ctx.kernel_w, h, w) # [B, C, K, H, W]
        Unfold_output_reshape = Unfold_Output.reshape(b, ctx.kernel_h, ctx.kernel_w, ctx.kernel_h, ctx.kernel_w, h, w) # [B, K, K, K, K, H, W]
        Unfold_Dot_Mul = torch.zeros_like(grad_output_reshape) # [B, K, K, H, W]
        # 这个是卷积公式的变形版本, 从[B,K,K,K,K,H,W]中取出K×K个[B,H,W],
        # 对应关系为: ctx.kernel_h-1-i, ctx.kernel_w-1-j, i, j 对应的是 i,j
        for i in range(ctx.kernel_h):
            for j in range(ctx.kernel_w):
                Unfold_Dot_Mul[:, i, j, :, :] = Unfold_output_reshape[:, ctx.kernel_h-1-i, ctx.kernel_w-1-j, i, j, :, :]
        # 取出来之后的结果点积相加就好了
        Unfold_Dot_Mul = Unfold_Dot_Mul.reshape(b, ctx.kernel_h*ctx.kernel_w, h, w).unsqueeze(1) # [B, 1, K, H, W]
        grad_K = torch.mul(Unfold_Dot_Mul, Unfold_Q_reshape).sum(2)

        return grad_Q, grad_K, None, None, None, None


msaconv_op = MSA_ConvF.apply


class MSA_Conv2d_v1(nn.Module):

    def __init__(self,
                 kernel_size: int or tuple,
                 dilation: int = 1,
                 im2col_step: int = 32):
        super().__init__()
        # print(kernel_size)
        if isinstance(kernel_size, int):
            self.kernel_size_h = kernel_size
            self.kernel_size_w = kernel_size
        else:
            self.kernel_size_h = kernel_size[0]
            self.kernel_size_w = kernel_size[1]
        self.stride = 1
        self.dilation = dilation
        # 这个一定要设置，而不是默认为1，C++padding为1也会给你取出来你需要形状的tensor，可能就是别的地址上的数值了
        # 设置不正确的话，会影响最终结果
        self.padding_h = dilation * (self.kernel_size_h - 1) // 2
        self.padding_w = dilation * (self.kernel_size_w - 1) // 2
        self.im2col_step = im2col_step

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,) -> torch.Tensor:
        return msaconv_op(Q, K, self.kernel_size_h, self.kernel_size_w, self.stride, self.dilation,
                                 self.padding_h, self.padding_w, self.im2col_step)