import torch
import torch.nn as nn
from core.op.msa_conv_v1_1 import MSA_Conv2d_v1
from core.op.msa_conv1_v1_1 import MSA_Conv2d_1_v1
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def cascade_test():
    device_name = 'cuda:0'
    # device_name = 'cpu'
    batch_size = 32
    channel = 16
    height = 32
    width = 32
    l1,l2,l3,l4 = 1,1,1,1
    Q = torch.rand([batch_size, channel, height, width]).to(device_name)
    K = torch.rand([batch_size, channel, height, width]).to(device_name)
    V = torch.rand([batch_size, channel, height, width]).to(device_name)
    Q1 = Q.clone().detach()
    K1 = K.clone().detach()
    V1 = V.clone().detach()
    Q.requires_grad = True
    K.requires_grad = True
    Q1.requires_grad = True
    K1.requires_grad = True
    V.requires_grad = True
    V1.requires_grad = True
    # ts = torch.Tensor(out_channels)
    ks=7
    ks1=5
    ks2=1
    dr=2
    my_layer = MSA_Conv2d_v1((ks1,ks2), dilation=dr).to(device_name)
    #my_layer_1 = MSA_Conv2d_v1((ks2,ks1), dilation=dr).to(device_name)
    my_layer1 = MSA_Conv2d_1_v1((ks2,ks1), dilation=dr).to(device_name)
    #my_layer1_1 = MSA_Conv2d_1_v1((ks2,ks1), dilation=dr).to(device_name)

    # 验证结果
    slide_window = nn.Unfold(kernel_size=(ks1, ks2), stride=1, dilation=dr, padding= (dr*(ks1 - 1) // 2, dr*(ks2 - 1) // 2))
    kv_transpose = nn.Sequential(Rearrange('b (c h w) l -> (h w) c (b l)', h=ks1, w=ks2, c=channel))
    q_transpose = nn.Sequential(Rearrange('b c h w -> c (b h w)'))
    dot_transpose = nn.Sequential(Rearrange('k (b h w) -> b k h w', h=height, w=width))
    #dot1_transpose = nn.Sequential(Rearrange('b k h w -> k (b h w)', h=height, w=width))
    #out_transpose = nn.Sequential(Rearrange('c (b h w) -> b c h w', h=height, w=width))

    k_unfold = slide_window(K1) # [B, C* kH * kW, L], 其中kH是核的高，kW是核宽 [2, 9216=1024*3*3, 4096]
    # v_unfold = slide_window(V) # L= H*W 窗口数量 [2, 9216=1024*3*3, 4096]
    #print(k_unfold.shape)
    k = kv_transpose(k_unfold) # [K, C, BHW]
    # v = kv_transpose(v_unfold) # [K, C, BHW]

    q = q_transpose(Q1).unsqueeze(0) # [B, 1, H*W, C] [2, 1, 4096, 1024]
    #print(q.shape)
    dot = torch.sum(torch.mul(q, k), dim=1) # [B, K, H, W]
    dot = dot_transpose(dot) # [K, BHW]

    dropout =  0.
    attend_use = nn.Softmax(dim = 1)
    dropout_use = nn.Dropout(dropout)

    attn = attend_use(dot) # [B, C, H*W, kH*kW] [2, 4096, 9, 9]
    attn = dropout_use(attn) # [B, C, H*W, kH*kW] [2, 4096, 9, 9]

    #V1 = V.clone().detach()


    # 验证结果
    slide_window = nn.Unfold(kernel_size=(ks2, ks1), stride=1, dilation=dr, padding= (dr*(ks2 - 1) // 2, dr*(ks1 - 1) // 2))
    kv_transpose = nn.Sequential(Rearrange('b (c h w) l -> (h w) c (b l)', h=ks2, w=ks1, c=channel))
    dot1_transpose = nn.Sequential(Rearrange('b k h w -> k (b h w)', h=height, w=width))
    out_transpose = nn.Sequential(Rearrange('c (b h w) -> b c h w', h=height, w=width))

    v_unfold = slide_window(V1) # [B, C* kH * kW, L], 其中kH是核的高，kW是核宽 [2, 9216=1024*3*3, 4096]
    v = kv_transpose(v_unfold) # [K, C, BHW]

    dot1 = dot1_transpose(attn).unsqueeze(1) # [K, 1, BHW]
    out = torch.sum(torch.mul(dot1, v), dim=0)
    out = out_transpose(out)

    Dot = my_layer(Q, K)
    Atten = attend_use(Dot) # [B, C, H*W, kH*kW] [2, 4096, 9, 9]
    Atten = dropout_use(Atten) # [B, C, H*W, kH*kW] [2, 4096, 9, 9]

    res = my_layer1(Atten, V)
    res.sum().backward()
    print('-----')
    t = res - out
    print(t.max())
    print(t.min())
    print('-----')
    #print(Q.grad)
    print(torch.sum(Q.grad))
    print('-----')
    out.sum().backward()
    #print(Q1.grad)
    print(torch.sum(Q1.grad))
    t = Q.grad - Q1.grad
    print(t.max())
    print(t.min())
    print('-----')
    print(torch.sum(K1.grad))
    t = K.grad - K1.grad
    print(t.max())
    print(t.min())
    print('-----')
    print(torch.sum(V1.grad))
    t = V.grad - V1.grad
    # print(t)
    print(t.max())
    print(t.min())
    #print(t)

if __name__ == '__main__':
    cascade_test()