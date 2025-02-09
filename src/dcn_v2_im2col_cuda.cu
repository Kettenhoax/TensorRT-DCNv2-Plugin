#include "dcn_v2_im2col_cuda.h"
#include <cstdio>
#include <algorithm>
#include <cstring>

const int CUDA_NUM_THREADS = 512;

dim3 GET_BLOCKS(uint n)
{
    uint k = (n - 1) / CUDA_NUM_THREADS + 1;
    uint x = k;
    uint y = 1;
    if (x > 65535)
    {
        x = ceil(sqrt(x));
        y = (n - 1) / (x * CUDA_NUM_THREADS) + 1;
    }
    dim3 d = {x, y, 1};
    return d;
}

__device__ float dmcn_im2col_bilinear(const float *bottom_data,
                                      const int width, const int height,
                                      float h, float w)
{
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
    {
        v1 = bottom_data[h_low * width + w_low];
    }
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
    {
        v2 = bottom_data[h_low * width + w_high];
    }
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
    {
        v3 = bottom_data[h_high * width + w_low];
    }
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
    {
        v4 = bottom_data[h_high * width + w_high];
    }

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int num_channels, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n)
        return;
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = index / width_col / height_col;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    float *data_col_ptr = data_col + ((c_col * height_col) + h_col) * width_col + w_col;

    const float *input_ptr = data_im + c_im * height * width;
    const float *offset_ptr = data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *mask_ptr = data_mask + deformable_group_index * kernel_h * kernel_w * height_col * width_col;

    const int in_v = h_col * stride_h - pad_h;
    const int in_u = w_col * stride_w - pad_w;

    for (int i = 0; i < kernel_h; ++i)
    {
        for (int j = 0; j < kernel_w; ++j)
        {
            const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
            const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
            const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
            const float offset_h = offset_ptr[data_offset_h_ptr];
            const float offset_w = offset_ptr[data_offset_w_ptr];
            const float mask = mask_ptr[data_mask_hw_ptr];

            float val = static_cast<float>(0);
            // TODO swap u and v
            const float in_v_off = in_v + i * dilation_h + offset_h;
            const float in_u_off = in_u + j * dilation_w + offset_w;
            if (in_v_off > -1 && in_u_off > -1 && in_v_off < height && in_u_off < width)
            {
                val = dmcn_im2col_bilinear(input_ptr, width, height, in_v_off, in_u_off);
            }
            *data_col_ptr = val * mask;
            data_col_ptr += height_col * width_col;
        }
    }
}

void modulated_deformable_im2col_cuda(cudaStream_t stream,
                                      const float *data_im, const float *data_offset, const float *data_mask,
                                      const int batch_size, const int channels, const int height_im, const int width_im,
                                      const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
                                      const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                      const int dilation_h, const int dilation_w,
                                      const int deformable_group, float *data_col)
{
    // assert(batch_size == 1)
    // num_axes should be smaller than block size
    const int channel_per_deformable_group = channels / deformable_group;
    const int num_kernels = channels * height_col * width_col;
    modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
                                             0, stream>>>(
        num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kenerl_w,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
        channels, deformable_group, height_col, width_col, data_col);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}
