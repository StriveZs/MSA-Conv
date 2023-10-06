#include "pytorch_cpp_helper.hpp"
#include <torch/torch.h> // 这个不引用 声明Tensor的时候会报错
// #include "pytorch_cuda_helper.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
// 使用PYBIND11_MODULE必须加下面两行！！！！ 
// 要不会报PYBIND11_MODULE error: expected constructor, destructor, or type conversion before ‘(’ token
#include <iostream>
using namespace std;

void msa_conv_im2col_cpu(Tensor data_im,
                        const int channels, const int height,
                        const int width, const int ksize_h,
                        const int ksize_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w,
                        const int dilation_h, const int dilation_w,
                        const int parallel_imgs, Tensor data_col);

void msa_conv_im2col_cuda(Tensor data_im,
                        const int channels, const int height,
                        const int width, const int ksize_h,
                        const int ksize_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w,
                        const int dilation_h, const int dilation_w,
                        const int parallel_imgs, Tensor data_col);

void msa_conv_shape_check(at::Tensor Q,
                         at::Tensor K, int kH,
                         int kW, int dH, int dW, int padH, int padW,
                         int dilationH, int dilationW)
{
    TORCH_CHECK(
        Q.ndimension() == 4,
        "4D Q tensor (nOutputPlane,nInputPlane,kH,kW) expected, but got: %s",
        Q.ndimension());

    TORCH_CHECK(
        K.ndimension() == 4,
        "4D K tensor (nOutputPlane,nInputPlane,kH,kW) expected, but got: %s",
        K.ndimension());

    TORCH_CHECK(kW > 0 && kH > 0,
                "kernel size should be greater than zero, but got kH: %d kW: %d",
                kH, kW);

    TORCH_CHECK(dW > 0 && dH > 0,
                "stride should be greater than zero, but got dH: %d dW: %d", dH,
                dW);
    
    TORCH_CHECK(
        dilationW > 0 && dilationH > 0,
        "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
        dilationH, dilationW);

    int ndim = K.ndimension();
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4)
    {
        dimf++;
        dimh++;
        dimw++;
    }

    TORCH_CHECK(ndim == 3 || ndim == 4,
                "3D or 4D input tensor expected but got: %s", ndim);

    // K Check
    long K_inputHeight = K.size(dimh);
    long K_inputWidth = K.size(dimw);
    long K_outputHeight =
        (K_inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    long K_outputWidth =
        (K_inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

    if (K_outputWidth < 1 || K_outputHeight < 1)
        AT_ERROR(
            "K Output size is too small"
        );

    //TORCH_CHECK((K_inputHeight >= kH && K_inputWidth >= kW), "input K is smaller than kernel");
    
    // Q Check
    long Q_inputHeight = Q.size(dimh);
    long Q_inputWidth = Q.size(dimw);
    long Q_outputHeight =
        (Q_inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    long Q_outputWidth =
        (Q_inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

    if (Q_outputWidth < 1 || Q_outputHeight < 1)
        AT_ERROR(
            "Q Output size is too small"
        );

    //TORCH_CHECK((Q_inputHeight >= kH && Q_inputWidth >= kW), "input Q is smaller than kernel");

}

void msa_conv_forward(Tensor Q, Tensor K, Tensor output,
                     Tensor columns, int kW,
                     int kH, int dW, int dH, int padW, int padH,
                     int dilationW, int dilationH,
                     int im2col_step)
{   
    bool isCuda = false;
    if (K.device().is_cuda())
    {
        CHECK_CUDA_INPUT(Q);
        CHECK_CUDA_INPUT(K);
        CHECK_CUDA_INPUT(output);
        CHECK_CUDA_INPUT(columns);
        isCuda = true;
    }
    else
    {
        CHECK_CPU_INPUT(Q);
        CHECK_CPU_INPUT(K);
        CHECK_CPU_INPUT(output);
        CHECK_CPU_INPUT(columns);
    }

    msa_conv_shape_check(Q, K, kH, kW, dH, dW, padH, padW, dilationH, dilationW); // 输入检查

    at::DeviceGuard guard(K.device());

    long batchSize = K.size(0);
    long nInputPlane = K.size(1);
    long inputHeight = K.size(2);
    long inputWidth = K.size(3);

    // long nOutputPlane = nInputPlane;

    long outputWidth = inputWidth;
    long outputHeight = inputHeight;
    
    // 如果batchSize不能够整除im2col_step，则直接使用
    

    output = output.view({batchSize / im2col_step, im2col_step, kH*kW,
                          outputHeight, outputWidth});
    columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
        K.options());
    Tensor Q_columns = at::zeros({1, nInputPlane, im2col_step * outputHeight * outputWidth}, Q.options());

    K = K.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth});
    Q = Q.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth});

    Tensor output_buffer = at::zeros({batchSize / im2col_step, kH*kW,
                                      im2col_step * outputHeight * outputWidth},
                                     output.options());
    // 这里的at::zeros 以及下面的at::sum就是pytorch在C++中的接口
    // 用法等同于torch.zeros(), torch.sum()
    // output_buffer = output_buffer.view(
    //     {output_buffer.size(0), output_buffer.size(1),
    //      output_buffer.size(2)});

    for (int elt = 0; elt < batchSize / im2col_step; elt++)
    {
        if (isCuda)
        {
            msa_conv_im2col_cuda(K[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, columns);
        }
        else
        {
            msa_conv_im2col_cpu(K[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, columns);
        }

        // Tensor Q_columns = Q[elt].view({1, nInputPlane, im2col_step * outputHeight * outputWidth});
        Q_columns.copy_(Q[elt].clone().transpose_(0, 1).reshape({1, nInputPlane, im2col_step * outputHeight * outputWidth}));
        Tensor tt = columns.reshape({nInputPlane, kH*kW, im2col_step * outputHeight * outputWidth}).transpose_(0, 1);
        Tensor res = at::sum(tt.mul_(Q_columns), 1);
        // Q × V
        output_buffer[elt] = res;
    }

    output_buffer = output_buffer.view({batchSize / im2col_step, kH*kW,
                                        im2col_step, outputHeight, outputWidth});
    output_buffer.transpose_(1, 2);

    output.copy_(output_buffer);
    output = output.view({batchSize, kH*kW, outputHeight, outputWidth});
    K = K.view({batchSize, nInputPlane, inputHeight, inputWidth});
    Q = Q.view({batchSize, nInputPlane, inputHeight, inputWidth});
}

void msa_conv1_forward(Tensor Dot, Tensor V, Tensor output,
                     Tensor columns, int kW,
                     int kH, int dW, int dH, int padW, int padH,
                     int dilationW, int dilationH,
                     int im2col_step)
{   
    // CUDA测试 这个后面再添加上
    bool isCuda = false;
    if (V.device().is_cuda())
    {
        CHECK_CUDA_INPUT(Dot);
        CHECK_CUDA_INPUT(V);
        CHECK_CUDA_INPUT(output);
        CHECK_CUDA_INPUT(columns);
        isCuda = true;
    }
    else
    {
        CHECK_CPU_INPUT(Dot);
        CHECK_CPU_INPUT(V);
        CHECK_CPU_INPUT(output);
        CHECK_CPU_INPUT(columns);
    }
    

    msa_conv_shape_check(Dot, V, kH, kW, dH, dW, padH, padW, dilationH, dilationW); // 输入检查

    at::DeviceGuard guard(V.device());

    long batchSize = V.size(0);
    long nInputPlane = V.size(1);
    long inputHeight = V.size(2);
    long inputWidth = V.size(3);

    // long nOutputPlane = nInputPlane;

    long outputWidth = inputWidth;
    long outputHeight = inputHeight;

    output = output.view({batchSize / im2col_step, im2col_step, nInputPlane,
                          outputHeight, outputWidth});
    columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
        V.options());
    Tensor Dot_columns = at::zeros({kH*kW, 1, im2col_step * outputHeight * outputWidth}, V.options());

    Dot = Dot.view({batchSize / im2col_step, im2col_step, kH*kW,
                        inputHeight, inputWidth});
    V = V.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth});

    Tensor output_buffer = at::zeros({batchSize / im2col_step, nInputPlane,
                                      im2col_step * outputHeight * outputWidth},
                                     output.options());
    // 这里的at::zeros 以及下面的at::sum就是pytorch在C++中的接口
    // 用法等同于torch.zeros(), torch.sum()
    // output_buffer = output_buffer.view(
    //     {output_buffer.size(0), output_buffer.size(1),
    //      output_buffer.size(2)});

    for (int elt = 0; elt < batchSize / im2col_step; elt++)
    {   
        if (isCuda)
        {
            msa_conv_im2col_cuda(V[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, columns);
        }
        else
        {
            msa_conv_im2col_cpu(V[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW,
                        dilationH, dilationW,
                        im2col_step, columns);
        }

        // Tensor Q_columns = Q[elt].view({1, nInputPlane, im2col_step * outputHeight * outputWidth});
        Dot_columns.copy_(Dot[elt].clone().transpose_(0, 1).reshape({kH*kW, 1, im2col_step * outputHeight * outputWidth}));
        Tensor t1 = columns.reshape({nInputPlane, kH*kW, im2col_step * outputHeight * outputWidth}).transpose_(0, 1);
        Tensor res = at::sum(t1.mul_(Dot_columns), 0);
        // Q × V
        output_buffer[elt] = res;
    }

    output_buffer = output_buffer.view({batchSize / im2col_step, nInputPlane,
                                        im2col_step, outputHeight, outputWidth});
    output_buffer.transpose_(1, 2);

    output.copy_(output_buffer);
    output = output.view({batchSize, nInputPlane, outputHeight, outputWidth});
    Dot = Dot.view({batchSize, kH*kW, inputHeight, inputWidth});
    V = V.view({batchSize, nInputPlane, inputHeight, inputWidth});
}

void msa_conv_backward(Tensor Q, Tensor K, Tensor grad_output, 
                     Tensor grad_Q, Tensor grad_K,
                     Tensor Q_columns, Tensor K_columns, Tensor Output_columns, 
                     int kW, int kH, int dW, int dH, int padW, int padH,
                     int dilationW, int dilationH,
                     int im2col_step)
{
    // CUDA测试 这个后面再添加上
    bool isCuda = false;
    if (Q.device().is_cuda())
    {
        CHECK_CUDA_INPUT(Q);
        CHECK_CUDA_INPUT(K);
        CHECK_CUDA_INPUT(grad_output);
        CHECK_CUDA_INPUT(grad_Q);
        CHECK_CUDA_INPUT(grad_K);
        CHECK_CUDA_INPUT(Q_columns);
        CHECK_CUDA_INPUT(K_columns);
        CHECK_CUDA_INPUT(Output_columns);
        isCuda = true;
    }
    else
    {
        CHECK_CPU_INPUT(Q);
        CHECK_CPU_INPUT(K);
        CHECK_CUDA_INPUT(grad_output);
        CHECK_CPU_INPUT(grad_Q);
        CHECK_CPU_INPUT(grad_K);
        CHECK_CPU_INPUT(Q_columns);
        CHECK_CPU_INPUT(K_columns);
        CHECK_CPU_INPUT(Output_columns);
    }

    at::DeviceGuard guard(Q.device());

    long batchSize = Q.size(0);
    long nInputPlane = Q.size(1);
    long inputHeight = Q.size(2);
    long inputWidth = Q.size(3);

    grad_Q = grad_Q.view({batchSize / im2col_step, im2col_step, nInputPlane,
                          inputHeight, inputWidth}); // [B/b, b, C, H, W]
    grad_K = grad_K.view({batchSize / im2col_step, im2col_step, nInputPlane,
                          inputHeight, inputWidth}); // [B/b, b, C, H, W]

    Q_columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * inputHeight * inputWidth},
        Q.options()); // [CKK, bHW]
    K_columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * inputHeight * inputWidth},
        Q.options()); // [CKK, bHW]
    Output_columns = at::zeros(
        {kW * kH * kW * kH, im2col_step * inputHeight * inputWidth},
        Q.options()); // [KKKK, bHW]

    Q = Q.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth}); // [B/b, b, C, H, W]
    K = K.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth}); // [B/b, b, C, H, W]
    grad_output = grad_output.view({batchSize / im2col_step, im2col_step, kW * kH,
                        inputHeight, inputWidth}); // [B/b, b, KK, H, W]

    Tensor grad_output_reshape = at::zeros({1, kH*kW, im2col_step * inputHeight * inputWidth},
                                     grad_Q.options());

    Tensor grad_Q_buffer = at::zeros({batchSize / im2col_step, nInputPlane,
                                      im2col_step * inputHeight * inputWidth},
                                     grad_Q.options()); // [B/b, C, bHW]
    Tensor grad_K_buffer = at::zeros({batchSize / im2col_step, nInputPlane,
                                      im2col_step * inputHeight * inputWidth},
                                     grad_K.options()); // [B/b, C, bHW]

    for (int elt = 0; elt < batchSize / im2col_step; elt++)
    {   
        if (isCuda)
        {
            msa_conv_im2col_cuda(Q[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, Q_columns);
            
            msa_conv_im2col_cuda(K[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, K_columns);

            msa_conv_im2col_cuda(grad_output[elt], kH*kW, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, Output_columns);
        }
        else
        {
            msa_conv_im2col_cpu(Q[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW,
                        dilationH, dilationW,
                        im2col_step, Q_columns);
            
            msa_conv_im2col_cpu(K[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW,
                        dilationH, dilationW,
                        im2col_step, K_columns);
            
            msa_conv_im2col_cpu(grad_output[elt], kH*kW, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW,
                        dilationH, dilationW,
                        im2col_step, Output_columns);
        }

        // grad_Q
        // K_columns: [CKK, bHW], grad_Q_buffer: [C, bHW]
        // 涉及到对输入tensor进行变形操作，比如view，transpose等需要在变形回来，要不就要用到clone()生成一个新的，否则会在backward计算时报错!!!
        grad_output_reshape.copy_(grad_output[elt].clone().transpose_(0, 1).unsqueeze(0).reshape({1, kH*kW, im2col_step * inputHeight * inputWidth})); // [b, KK, H, W] -> [KK, b, H, W] -> [1, KK, bHW]
        Tensor Unfold_K_reshape = K_columns.reshape({nInputPlane, kH*kW, im2col_step * inputHeight * inputWidth}); // [C, KK, bHW]
        grad_Q_buffer[elt] = at::sum(at::mul(grad_output_reshape, Unfold_K_reshape), 1);

        // grad_K
        Tensor Unfold_Q_reshape = Q_columns.reshape({nInputPlane, kH*kW, im2col_step * inputHeight * inputWidth}); // [C, KK, bHW]
        Tensor Unfold_Dot_Mul = at::zeros({kH, kW, im2col_step * inputHeight * inputWidth}, Q.options()); // [K, K, bHW]
        Tensor Unfold_output_reshape = Output_columns.reshape({kH, kW, kH, kW, im2col_step * inputHeight * inputWidth}); // [K, K, K, K, bHW]
        for (int i = 0; i < kH; i++){
            for (int j = 0; j < kW; j++){
                // Unfold_Dot_Mul(:, i, j, :, :) = Unfold_output_reshape(:, kH-1-i, kW-1-j, i, j, :, :);
                using namespace torch::indexing;
                auto unfold_index = Unfold_output_reshape.index({kH-1-i, kW-1-j, i, j, Slice(None)});
                Unfold_Dot_Mul.index({i, j, Slice(None)}) = unfold_index;
            }
        }
        Unfold_Dot_Mul = Unfold_Dot_Mul.reshape({1, kH*kW, im2col_step * inputHeight * inputWidth}); // [1, KK, bHW]
        grad_K_buffer[elt] = at::sum(at::mul(Unfold_Dot_Mul, Unfold_Q_reshape), 1); // [C, bHW]
    }


    grad_Q_buffer = grad_Q_buffer.view({batchSize / im2col_step, nInputPlane,
                                        im2col_step, inputHeight, inputWidth});
    grad_K_buffer = grad_K_buffer.view({batchSize / im2col_step, nInputPlane,
                                        im2col_step, inputHeight, inputWidth});
    grad_Q_buffer.transpose_(1, 2);
    grad_K_buffer.transpose_(1, 2);

    grad_Q.copy_(grad_Q_buffer);
    grad_K.copy_(grad_K_buffer);

    grad_Q = grad_Q.view({batchSize, nInputPlane, inputHeight, inputWidth});
    grad_K = grad_K.view({batchSize, nInputPlane, inputHeight, inputWidth});
    
    Q = Q.view({batchSize, nInputPlane, inputHeight, inputWidth}); // [B/b, b, C, H, W]
    K = K.view({batchSize, nInputPlane, inputHeight, inputWidth}); // [B/b, b, C, H, W]
    grad_output = grad_output.view({batchSize, kW * kH, inputHeight, inputWidth}); // [B/b, b, KK, H, W]
}

void msa_conv1_backward(Tensor Dot, Tensor V, Tensor grad_output, 
                     Tensor grad_Dot, Tensor grad_V,
                     Tensor Dot_columns, Tensor V_columns, Tensor Output_columns, 
                     int kW, int kH, int dW, int dH, int padW, int padH,
                     int dilationW, int dilationH,
                     int im2col_step)
{
    // CUDA测试 这个后面再添加上
    bool isCuda = false;
    if (V.device().is_cuda())
    {
        CHECK_CUDA_INPUT(Dot);
        CHECK_CUDA_INPUT(V);
        CHECK_CUDA_INPUT(grad_output);
        CHECK_CUDA_INPUT(grad_Dot);
        CHECK_CUDA_INPUT(grad_V);
        CHECK_CUDA_INPUT(Dot_columns);
        CHECK_CUDA_INPUT(V_columns);
        CHECK_CUDA_INPUT(Output_columns);
        isCuda = true;
    }
    else
    {
        CHECK_CPU_INPUT(Dot);
        CHECK_CPU_INPUT(V);
        CHECK_CUDA_INPUT(grad_output);
        CHECK_CPU_INPUT(grad_Dot);
        CHECK_CPU_INPUT(grad_V);
        CHECK_CPU_INPUT(Dot_columns);
        CHECK_CPU_INPUT(V_columns);
        CHECK_CPU_INPUT(Output_columns);
    }

    at::DeviceGuard guard(V.device());

    long batchSize = V.size(0);
    long nInputPlane = V.size(1);
    long inputHeight = V.size(2);
    long inputWidth = V.size(3);

    grad_Dot = grad_Dot.view({batchSize / im2col_step, im2col_step, kH*kW,
                          inputHeight, inputWidth}); // [B/b, b, KK, H, W]
    grad_V = grad_V.view({batchSize / im2col_step, im2col_step, nInputPlane,
                          inputHeight, inputWidth}); // [B/b, b, C, H, W]

    Dot_columns = at::zeros(
        {kW * kH * kW * kH, im2col_step * inputHeight * inputWidth},
        V.options()); // [KKKK, bHW]
    V_columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * inputHeight * inputWidth},
        V.options()); // [CKK, bHW]
    Output_columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * inputHeight * inputWidth},
        V.options()); // [CKK, bHW]

    Dot = Dot.view({batchSize / im2col_step, im2col_step, kH*kW,
                        inputHeight, inputWidth}); // [B/b, b, KK, H, W]
    V = V.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth}); // [B/b, b, C, H, W]
    grad_output = grad_output.view({batchSize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth}); // [B/b, b, C, H, W]

    Tensor grad_output_reshape = at::zeros({nInputPlane, 1, im2col_step * inputHeight * inputWidth},
                                     grad_Dot.options());

    Tensor grad_Dot_buffer = at::zeros({batchSize / im2col_step, kH*kW,
                                      im2col_step * inputHeight * inputWidth},
                                     grad_Dot.options()); // [B/b, KK, bHW]
    Tensor grad_V_buffer = at::zeros({batchSize / im2col_step, nInputPlane,
                                      im2col_step * inputHeight * inputWidth},
                                     grad_V.options()); // [B/b, C, bHW]

    for (int elt = 0; elt < batchSize / im2col_step; elt++)
    {   
        if (isCuda)
        {
            msa_conv_im2col_cuda(Dot[elt], kH*kW, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, Dot_columns);
            
            msa_conv_im2col_cuda(V[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, V_columns);

            msa_conv_im2col_cuda(grad_output[elt], nInputPlane, inputHeight,
                            inputWidth, kH, kW, padH, padW, dH, dW,
                            dilationH, dilationW,
                            im2col_step, Output_columns);
        }
        else
        {
            msa_conv_im2col_cpu(Dot[elt], kH*kW, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW,
                        dilationH, dilationW,
                        im2col_step, Dot_columns);
            
            msa_conv_im2col_cpu(V[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW,
                        dilationH, dilationW,
                        im2col_step, V_columns);
            
            msa_conv_im2col_cpu(grad_output[elt], nInputPlane, inputHeight,
                        inputWidth, kH, kW, padH, padW, dH, dW,
                        dilationH, dilationW,
                        im2col_step, Output_columns);
        }
        
        // grad_Dot
        // V_columns: [CKK, bHW], grad_Dot_buffer: [KK, bHW], grad_output: [B/b, b, C, H, W]
        grad_output_reshape.copy_(grad_output[elt].clone().transpose_(0, 1).unsqueeze(1).reshape({nInputPlane, 1, im2col_step * inputHeight * inputWidth})); // [b, C, H, W] -> [C, b, H, W] -> [C, 1, bHW]
        Tensor Unfold_V_reshape = V_columns.reshape({nInputPlane, kH*kW, im2col_step * inputHeight * inputWidth}); // [C, KK, bHW]
        grad_Dot_buffer[elt] = at::sum(at::mul(grad_output_reshape, Unfold_V_reshape), 0); // [KK, bHW]

        // grad_V
        Tensor Unfold_Dot_reshape = Dot_columns.reshape({kH, kW, kH, kW, im2col_step * inputHeight * inputWidth}); // [K, K, K, K, bHW]
        Tensor Unfold_Dot_Mul = at::zeros({kH, kW, im2col_step * inputHeight * inputWidth}, Dot.options()); // [K, K, bHW]
        Tensor Unfold_output_reshape = Output_columns.reshape({nInputPlane, kH*kW, im2col_step * inputHeight * inputWidth}); // [C, KK, bHW]
        for (int i = 0; i < kH; i++){
            for (int j = 0; j < kW; j++){
                // Unfold_Dot_Mul(:, i, j, :, :) = Unfold_output_reshape(:, kH-1-i, kW-1-j, i, j, :, :);
                using namespace torch::indexing;
                auto unfold_index = Unfold_Dot_reshape.index({kH-1-i, kW-1-j, i, j, Slice(None)});
                Unfold_Dot_Mul.index({i, j, Slice(None)}) = unfold_index;
            }
        }
        Unfold_Dot_Mul = Unfold_Dot_Mul.reshape({1, kH*kW, im2col_step * inputHeight * inputWidth}); // [1, KK, bHW]
        grad_V_buffer[elt] = at::sum(at::mul(Unfold_Dot_Mul, Unfold_output_reshape), 1); // [C, bHW]
    }


    grad_Dot_buffer = grad_Dot_buffer.view({batchSize / im2col_step, kH*kW,
                                        im2col_step, inputHeight, inputWidth});
    grad_V_buffer = grad_V_buffer.view({batchSize / im2col_step, nInputPlane,
                                        im2col_step, inputHeight, inputWidth});
    grad_Dot_buffer.transpose_(1, 2);
    grad_V_buffer.transpose_(1, 2);

    grad_Dot.copy_(grad_Dot_buffer);
    grad_V.copy_(grad_V_buffer);

    grad_Dot = grad_Dot.view({batchSize, kH*kW, inputHeight, inputWidth});
    grad_V = grad_V.view({batchSize, nInputPlane, inputHeight, inputWidth});

    Dot = Dot.view({batchSize, kH*kW, inputHeight, inputWidth}); // [B/b, b, KK, H, W]
    V = V.view({batchSize, nInputPlane, inputHeight, inputWidth}); // [B/b, b, C, H, W]
    grad_output = grad_output.view({batchSize, nInputPlane, inputHeight, inputWidth}); // [B/b, b, C, H, W]
}

template <typename T>
void msa_conv_im2col_cpu_kernel(
    const int n, const T *data_im, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int batch_size,
    const int num_channels, const int height_col,
    const int width_col, T *data_col)
{
    for (int index = 0; index < n; index++)
    {
        // index index of output matrix
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int b_col = (index / width_col / height_col) % batch_size;
        const int c_im = (index / width_col / height_col) / batch_size;
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        T *data_col_ptr =
            data_col +
            ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
        const T *data_im_ptr =
            data_im + (b_col * num_channels + c_im) * height * width;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                T val = static_cast<T>(0);
                const int h_im = h_in + i * dilation_h;
                const int w_im = w_in + j * dilation_w;
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
                {
                    val = data_im_ptr[h_im * width + w_im];
                }
                *data_col_ptr = val;
                data_col_ptr = data_col_ptr + batch_size * height_col * width_col;
            }
        }
    }
}

void msa_conv_im2col_cpu(Tensor data_im,
                        const int channels, const int height,
                        const int width, const int ksize_h,
                        const int ksize_w, const int pad_h, const int pad_w,
                        const int stride_h, const int stride_w,
                        const int dilation_h, const int dilation_w,
                        const int parallel_imgs, Tensor data_col)
{
    int height_col =
        (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col * parallel_imgs;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_im.scalar_type(), "", [&]
        { msa_conv_im2col_cpu_kernel<scalar_t>(
              num_kernels, data_im.data_ptr<scalar_t>(),
              height, width, ksize_h, ksize_w,
              pad_h, pad_w, stride_h, stride_w,
              dilation_h, dilation_w,
              parallel_imgs, channels,
              height_col, width_col, data_col.data_ptr<scalar_t>()); });
}

// void test(at::Tensor input){}

// PYBIND11_MODULE(myop, m) {
//     //第一个参数表示定义的函数的名称，第二个参数表示关联的是哪个实际的c++函数
// 		//py::arg用法是告诉python调用层能看到定义的函数中的参数名称是什么
// 		//后面的= 表示默认的参数，这就相当于我们在python中定义一个函数指定好默认值是一样的
//       m.def("test", test,
//             py::arg("input"));
// }
PYBIND11_MODULE(msa_conv1, m) {
    //第一个参数表示定义的函数的名称，第二个参数表示关联的是哪个实际的c++函数
		//py::arg用法是告诉python调用层能看到定义的函数中的参数名称是什么
		//后面的= 表示默认的参数，这就相当于我们在python中定义一个函数指定好默认值是一样的
      m.def("msa_conv_forward", msa_conv_forward,
            py::arg("Q"), 
            py::arg("K"), 
            py::arg("output"),
            py::arg("columns"), 
            py::arg("kW"),
            py::arg("kH"), 
            py::arg("dW"), 
            py::arg("dH"), 
            py::arg("padW"),
            py::arg("padH"), 
            py::arg("dilationW"),
            py::arg("dilationH"),
            py::arg("im2col_step"));

        m.def("msa_conv1_forward", msa_conv1_forward,
            py::arg("Dot"), 
            py::arg("V"), 
            py::arg("output"),
            py::arg("columns"), 
            py::arg("kW"),
            py::arg("kH"), 
            py::arg("dW"), 
            py::arg("dH"), 
            py::arg("padW"),
            py::arg("padH"), 
            py::arg("dilationW"),
            py::arg("dilationH"),
            py::arg("im2col_step"));
        
        m.def("msa_conv_backward", msa_conv_backward,
            py::arg("Q"), 
            py::arg("K"), 
            py::arg("grad_output"),
            py::arg("grad_Q"), 
            py::arg("grad_K"), 
            py::arg("Q_columns"), 
            py::arg("K_columns"), 
            py::arg("Output_columns"), 
            py::arg("kW"),
            py::arg("kH"), 
            py::arg("dW"), 
            py::arg("dH"), 
            py::arg("padW"),
            py::arg("padH"), 
            py::arg("dilationW"),
            py::arg("dilationH"),
            py::arg("im2col_step"));

        m.def("msa_conv1_backward", msa_conv1_backward,
            py::arg("Dot"), 
            py::arg("V"), 
            py::arg("grad_output"),
            py::arg("grad_Dot"), 
            py::arg("grad_V"), 
            py::arg("Dot_columns"), 
            py::arg("V_columns"), 
            py::arg("Output_columns"), 
            py::arg("kW"),
            py::arg("kH"), 
            py::arg("dW"), 
            py::arg("dH"), 
            py::arg("padW"),
            py::arg("padH"), 
            py::arg("dilationW"),
            py::arg("dilationH"),
            py::arg("im2col_step"));
}