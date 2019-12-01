/**
 * cuda-memcheckによるメモリ破壊バグ検出のサンプル
 **/
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cassert>
#include <string>
#include <sstream>

#define CHECK(call)                                                  \
{                                                                    \
    const cudaError_t error = call;                                  \
    std::stringstream ss;                                            \
    if (error != cudaSuccess)                                        \
    {                                                                \
        ss        << "CHECK cudaError_t: "                           \
                  << __FILE__                                        \
                  << "("                                             \
                  << __LINE__                                        \
                  << ")"                                             \
                  << ": "                                            \
                  << "Error"                                         \
                  << std::endl;                                      \
        ss        << "code: "                                        \
                  << error                                           \
                  << ", "                                            \
                  << "reason: "                                      \
                  << cudaGetErrorString(error)                       \
                  << std::endl;                                      \
        std::cerr << ss.str();                                       \
        /*std::exit(EXIT_FAILURE);*/                                 \
    }                                                                \
}

// 本サンプル用のダミー関数. 動作はy[0]...y[N-1]に1を書き込むだけ
__global__ void convolutionFowradDummy(long *y)
{
    long i = blockIdx.x*blockDim.x + threadIdx.x;
    y[i] = 1;
}

// 本サンプル用のダミー関数. 動作はy[0]...y[N-1]に2を書き込むだけ
__global__ void poolingFowradDummy(long *y)
{
    long i = blockIdx.x*blockDim.x + threadIdx.x;
    y[i] = 2;
}

int main(void)
{
    // 事前にError をcheckするAPIを1度実行しておく
    // (Errorが検出されないことが期待)
    CHECK(cudaGetLastError ());

    // convolutionFowradDummy. poolingFowradDummy用のデバイスメモリを確保する
    // サイズはどちらもN*8Byte確保することが期待.
    // ただし、わざとメモリ破壊バグを発生されるため、
    // convolutionFowradDummyのほうだけ期待より少ないサイズを確保する
    constexpr long N = 1<<5;
    long *out_conv, *out_pool;
    size_t size_ = N * sizeof(long);
    CHECK(cudaMalloc(&out_conv, size_ - 1));
    CHECK(cudaMalloc(&out_pool, size_));

    // convolutionFowradDummy,poolingFowradDummyの順で実行する
    // cuda-memcheckなるツールを使うと,
    // convolutionFowradDummyでメモリ破壊(Invalid __global__ write of size 8)
    // していることを検出できる
    convolutionFowradDummy<<<1, N>>>(out_conv);
    poolingFowradDummy<<<1, N>>>(out_pool);
    CHECK(cudaDeviceSynchronize());

    // Error をcheckするAPIを実行すると
    // 不明なエラー(code: 719, reason: unspecified launch failur)と表示される
    // つまり、このAPIだけでも何らかのエラーが起きたことは検出できる
    CHECK(cudaGetLastError ());
    return 0;
}
