---
title: Image Convolution on CUDA
subtitle: image conv.
date: 2025/5/27 22:46:00
tags: CUDA
---

## Image Convolution Formula

$$
(I * K)(x, y) = \sum^{a}_{i=1}\sum^{b}_{j=1} I(x + i, y + j) \cdot K(i, j)
$$

## Normal Image Convolution on CUDA

```cpp
__global__ void _conv2d(float *output, const float *input, const float *kernel, dim3 output_size, dim3 input_size, dim3 kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_size.x || y >= output_size.y) return;

    float sum = 0.0f;

    for (int i = 0; i < kernel_size.y; ++i) {
        for (int j = 0; j < kernel_size.x; ++j) {
            int px = x + j, py = y + i;
            sum += input[py * input_size.x + px] * kernel[i * kernel_size.x + j];
        }
    }

    output[y * output_size.x + x] = sum;
}
```

## Image Convolution using Shared Memory on CUDA

```cpp
__global__ void _conv2d_shared(float *output, const float *input, const float *kernel, dim3 output_size, dim3 input_size, dim3 kernel_size) {
    __shared__ float tile[32 + 3 - 1][32 + 3 - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_size.x || y >= output_size.y) return;

    for (int ky = 0; ky < kernel_size.y; ++ky) {
        for (int kx = 0; kx < kernel_size.x; ++kx) {
            int px = x + kx, py = y + ky;
            tile[ty + ky][tx + kx] = input[py * input_size.x + px];
        }
    }

    __syncthreads();

    float sum = 0.0f;

    for (int ky = 0; ky < kernel_size.y; ++ky) {
        for (int kx = 0; kx < kernel_size.x; ++kx) {
            sum += tile[ty + ky][tx + kx] * kernel[ky * kernel_size.x + kx];
        }
    }

    output[y * output_size.x + x] = sum;
}
```
