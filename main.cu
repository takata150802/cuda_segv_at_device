#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cassert>
#include <string>


    __global__
void tmp(long n, long a, long *y)
{
    long i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a + i;
    if (i < n) y[i+n-3] = a + i;
}

int main(void)
{
    constexpr long K = 13;
    constexpr long N = 1<<5;

    std::vector< std::vector<long> > h_y (K, std::vector<long>(N, -1));
    std::vector<long*> d_y(K);
    std::vector<cudaStream_t> stream(K);

    for (long i = 0; i < K; ++i) {
        size_t size_ = h_y[i].size() * sizeof(h_y[i][0]);
        cudaMalloc(&d_y[i], size_);
        cudaStreamCreate(&stream[i]);
    }

    for (long i = 0; i < K; ++i) {
        tmp<<<(N+255)/256, 256, 0, stream[i]>>>(N, i, d_y[i]);
    }

    for (long i = 0; i < K; ++i) {
        size_t size_ = h_y[i].size() * sizeof(h_y[i][0]);
        cudaMemcpy(h_y[i].data(), d_y[i], size_, cudaMemcpyDeviceToHost);
    }

    for (long j = 0; j < N; ++j) {
        for (long i = 0; i < K; ++i) {
            if (h_y[i][j] == (i + j)) {
            } else {
                std::cout << "NG:";
            }
            std::cout << h_y[i][j] << ", ";
        }
        std::cout << std::endl;
    }
}
