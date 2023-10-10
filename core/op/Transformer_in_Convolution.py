import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 将算子添加到当前目录下
import sys
sys.path.append('/file/msa_conv_cuda')

from core.op.msa_conv_v1_1 import MSA_Conv2d_v1
from core.op.msa_conv1_v1_1 import MSA_Conv2d_1_v1


class MSA_Conv_2d(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernel_size: int or tuple,
                 stride: int = 1,
                 dilation: int = 1,
                 dropout: float = 0.1,
                 im2col_step: int = 7,
                 padding=1,
                 bias=False):
        super().__init__()
        self.scale = input_channel ** -0.5

        self.conv1 = MSA_Conv2d_v1(kernel_size, dilation=dilation, im2col_step=im2col_step)
        self.conv2 = MSA_Conv2d_1_v1(kernel_size, dilation=dilation, im2col_step=im2col_step)
        
        self.qkv_conv = nn.Conv2d(input_channel, input_channel*3, kernel_size=1, stride=1, padding=0)
        # 这里的升维可以理解为一种embedding？
        self.attend = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(dropout)
        self.stride = stride
        
        self.mlp = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    
    def forward(self, x):
        """
        x: shape [B, C, H, W]
        """
        qkv = self.qkv_conv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: t.float(), qkv)

        # q × k
        dots = self.conv1(q, k) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # dot × v
        out = self.conv2(attn, v)

        out += x  # residual connection

        # mlp
        out = self.mlp(out)
        nn.ReLU(inplace=True)(out)
        
        return out
        
