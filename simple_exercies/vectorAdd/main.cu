#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <random>
#include <vector>
#include <iostream>

// ==============
// Main Function
// ==============

__global__ void vector_add(int* a, int* b, int* res, int num_elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements) {
        res[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <vector_length>" << std::endl;
        exit(1);
    }
    // Initialize vectors
    int vector_len = atoi(argv[1]);
    int* operand_a = new int[vector_len];
    int* operand_b = new int[vector_len];
    int* sum_ab = new int[vector_len];
    for (int i = 0; i < vector_len; ++i) {
        operand_a[i] = i;
    }
    for (int i = 0; i < vector_len; ++i) {
        operand_b[i] = i*i;
    }

    int* d_operand_a;
    int* d_operand_b;
    int* d_sum_ab;
    cudaMalloc((void**)&d_operand_a, vector_len * sizeof(int));
    cudaMalloc((void**)&d_operand_b, vector_len * sizeof(int));
    cudaMalloc((void**)&d_sum_ab, vector_len * sizeof(int));
    cudaMemcpy(d_operand_a, operand_a, vector_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_operand_b, operand_b, vector_len * sizeof(int), cudaMemcpyHostToDevice);
    
    // Algorithm
    vector_add<<<1, vector_len>>>(d_operand_a, d_operand_b, d_sum_ab, vector_len);
    cudaMemcpy(sum_ab, d_sum_ab, vector_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaError_t error_code = cudaGetLastError();
    if (error_code != cudaSuccess) {
        std::cerr << "cudaGetLastError failed: " << cudaGetErrorString(error_code) << std::endl;
        exit(1);
    }

    // Finalize
    for (int i = 0; i < vector_len; i++) {
        std::cout << operand_a[i] << " + " << operand_b[i] << " = " << sum_ab[i] << std::endl;
        if (sum_ab[i] != operand_a[i] + operand_b[i]) {
            std::cerr << "Error: incorrect result at index " << i << std::endl;
            exit(1);
        }
    }
    cudaFree(d_operand_a);
    cudaFree(d_operand_b);
    cudaFree(d_sum_ab);
    delete[] operand_a;
    delete[] operand_b;
    delete[] sum_ab;
}
