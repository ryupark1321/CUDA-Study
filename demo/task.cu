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

cudaError_t device_function_calls(double* cpu_TheArrayOfNewSalaries, double* d_original_salary, double* d_new_salary, int size) {
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        return cuda_status;
    }

    // 1. Allocation device memory
    cuda_status = cudaMalloc((void**)&d_original_salary, size * sizeof(double));
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_original_salary!" << std::endl;
        return cuda_status;
    }

    cuda_status = cudaMalloc((void**)&d_new_salary, size * sizeof(double));
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_new_salary!" << std::endl;
        return cuda_status;
    }

    // 2. Copy data from host to device
    cuda_status = cudaMemcpy(d_original_salary, TheArrayOfSalaries, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMemcopy failed for d_original_salary!" << std::endl;
        return cuda_status;
    }

    // 3. Kernel launch
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    gpu_salary_incrementer<<<blocks_per_grid, threads_per_block>>>(d_original_salary, d_new_salary, size);
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed!" << std::endl;
    }

    cuda_status = cudaGetLastError(); // Check for any kernel launch errors
    if (cuda_status != cudaSuccess) {
        std::cerr << "Error Detected!" << std::endl;
    }
    return cuda_status;
}

void cpu_salary_incrementer(const double original_salary[], double new_salary[], int size) {
    for (int i = 0; i < size; i++) {
        new_salary[i] = original_salary[i] * 1.15 + 5000;
    }
}

int main() {
    int size = sizeof(TheArrayOfSalaries) / sizeof(double);
    std::cout << "Size of TheArrayOfSalaries : " << size << std::endl;

    // CPU Computation for Reference
    double* cpu_TheArrayOfNewSalaries = new double[size](); // Define an array to hold new salaries, all 0's
    cpu_salary_incrementer(TheArrayOfSalaries, cpu_TheArrayOfNewSalaries, size);

    // GPU Computation
    double* d_original_salary;
    double* d_new_salary;
    cudaError_t cuda_success = device_function_calls(cpu_TheArrayOfNewSalaries, d_original_salary, d_new_salary, size);

    if (cuda_success != cudaSuccess) {
        std::cerr << "Device Function Calls Failed! CUDA error: " << cudaGetErrorString(cuda_success) << std::endl;
        cudaFree(d_original_salary);
        cudaFree(d_new_salary);
        delete[] cpu_TheArrayOfNewSalaries;
        return 1;
    }

    // 4. Copy data from device to host
    double gpu_TheArrayOfNewSalaries[size] = {0};
    cuda_status = cudaMemcpy(gpu_TheArrayOfNewSalaries, d_new_salary, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaMemCpy failed!" << std::endl;
    }

    // 5. Free device memory)
    cuda_status = cudaFree(d_original_salary);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaFree failed for d_original_salary!" << std::endl;
    }

    cuda_status = cudaFree(d_new_salary);
    if (cuda_status != cudaSuccess) {
        std::cerr << "cudaFree failed for d_new_salary!" << std::endl;
    }

    // Compare
    for (int i = 0; i < size; i++) {
        std::cout << TheArrayOfSalaries[i] << " -> " << cpu_TheArrayOfNewSalaries[i] << " = " << gpu_TheArrayOfNewSalaries[i] << std::endl;
    }
    
    // Free host memory
    delete[] cpu_TheArrayOfNewSalaries;
    return 0;
}