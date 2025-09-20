#include <iostream>
#include "TheEmployeesSalary.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gpu_salary_incrementer(const double* original_salary, double* new_salary, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        new_salary[i] = original_salary[i] * 1.15 + 5000;
    }
}

int main() {
    int size = sizeof(TheArrayOfSalaries) / sizeof(double);

    // CPU Computation for Reference
    double cpu_TheArrayOfNewSalaries[size] = {0};
    cpu_salary_incrementer(TheArrayOfSalaries, cpu_TheArrayOfNewSalaries, size);

    // GPU Computation
    // 1. Allocation device memory
    double* d_original_salary;
    double* d_new_salary;
    cudaMalloc((void**)&d_original_salary, size * sizeof(double));
    cudaMalloc((void**)&d_new_salary, size * sizeof(double));

    // 2. Copy data from host to device
    cudaMemcpy(d_original_salary, TheArrayOfSalaries, size * sizeof(double), cudaMemcpyHostToDevice);

    // 3. Kernel launch
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    gpu_salary_incrementer<<<blocks_per_grid, threads_per_block>>>(d_original_salary, d_new_salary, size);
    cudaDeviceSynchronize();

    // 4. Copy data from device to host
    double gpu_TheArrayOfNewSalaries[size] = {0};
    cudaMemcpy(gpu_TheArrayOfNewSalaries, d_new_salary, size * sizeof(double), cudaMemcpyDeviceToHost);

    // 5. Free device memory)
    cudaFree(d_original_salary);
    cudaFree(d_new_salary);

    // Compare
    bool comparison_result = compare_results(cpu_TheArrayOfNewSalaries, gpu_TheArrayOfNewSalaries, size);
    std::cout << "Comparison result: " << (comparison_result ? "Match" : "Mismatch") << std::endl;
    
    return 0;
}