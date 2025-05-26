---
title: Matrix Multiplication on CUDA
subtitle: CUDA
date: 2025/5/26 21:46:00
tags: CUDA
---

We implement 3 kinds of matrix multiplication to execute the matrix multiplication:

$$
C_{K \times M} = A_{K \times N} \times B_{N \times M}
$$

## Matrix Multiplication on CPU
```cpp
int main() {
    float *A, *B, *C;

    A = new float[K * N];
    B = new float[N * M];
    C = new float[K * M];

    for (int i = 0; i < K * N; ++i) {
        A[i] = static_cast<float>(i);
    }

    for (int i = 0; i < N * M; ++i) {
        B[i] = static_cast<float>(i);
    }

    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            C[k * M + m] = 0.0f;
            for (int n = 0; n < N; n++) {
                C[k * M + m] += A[k * N + n] * B[n * M + m];
            }
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
```

## Matrix Multiplication using Global Memory on CUDA
```cpp
__global__ void cuda_matmul(float* C, const float *A, const float *B, int K, int N, int M) {
    int row = blockIdx.y + blockDim.y + threadIdx.y;
    int col = blockIdx.x + blockDim.x + threadIdx.x;

    if (row < K && col < M) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * M + col];
        }
        C[row * M + col] = sum;
    }
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = new float[N * M];
    B = new float[N * M];
    C = new float[N * M];
    cudaMalloc((void**)&d_A, N * M * sizeof(float));
    cudaMalloc((void**)&d_B, N * M * sizeof(float));
    cudaMalloc((void**)&d_C, N * M * sizeof(float));

    for (int i = 0; i < N * M; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    cuda_matmul<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

## Matrix Multiplication using Shared Memory on CUDA

```cpp
__global__ void cuda_matmul_shared(float* C, const float *A, const float *B, int K, int N, int M) {
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < M) {
        float value = 0.0f;
        for (int i = 0; i < (N + 31) / 32; ++i) {
            // Memory access pattern: load 32x32 tiles from A and B
            if (i * 32 + threadIdx.x < N && row < K) {
                shared_A[threadIdx.y][threadIdx.x] = A[row * N + i * 32 + threadIdx.x];
            } else {
                shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            }
            if (i * 32 + threadIdx.y < N && col < M) {
                shared_B[threadIdx.y][threadIdx.x] = B[(i * 32 + threadIdx.y) * M + col];
            } else {
                shared_B[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();

            // Compute the partial sum for this tile
            for (int j = 0; j < 32; ++j) {
                value += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
            }
            __syncthreads();
        }

        C[row * M + col] = value;
    }
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    A = new float[N * M];
    B = new float[N * M];
    C = new float[N * M];
    cudaMalloc((void**)&d_A, N * M * sizeof(float));
    cudaMalloc((void**)&d_B, N * M * sizeof(float));
    cudaMalloc((void**)&d_C, N * M * sizeof(float));

    for (int i = 0; i < N * M; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    cuda_matmul_shared<<<grid, block>>>(d_A, d_B, d_C, K, N, M);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```
