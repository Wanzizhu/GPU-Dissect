#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <type_traits>
#include <typeinfo>


#define DATA_SIZE (160 * 1024) // Size of data to load into L1 cache
#define THREADS_PER_BLOCK 1024

template <typename DATA_TYPE>
__global__ void l1_cache_throughput_test(DATA_TYPE *data,
                                         unsigned long long *duration,
                                         unsigned int iterations) {
  unsigned int tid = threadIdx.x;
  DATA_TYPE value;
  DATA_TYPE sum;
  unsigned long long start_time, end_time;

  // Load memory into L1 cache with the "ca" modifier
  // This ensures the data is cached in L1
  // Repeatedly access L1 cache
  for (unsigned int i = tid; i < DATA_SIZE; i += THREADS_PER_BLOCK) {
    value = data[i];
    sum += value;
  }

  // Record the start time
  start_time = clock64();

  // Repeatedly access L1 cache
  for (int iter = 0; iter < iterations; iter++) {
    for (unsigned int i = tid; i < DATA_SIZE; i += THREADS_PER_BLOCK) {
      value = data[i];
      sum += value;
    }
  }

  // Record the end time
  end_time = clock64();

  // Store the duration (end_time - start_time) per thread
  if (tid == 0) {
    *duration = end_time - start_time;
    data[0] = sum;
  }
}

template <typename DATA_TYPE> int measure_throughput(int iterations) {
  DATA_TYPE *d_data;
  unsigned long long *d_duration, h_duration;

  // Allocate data on device
  cudaMalloc(&d_data, DATA_SIZE * sizeof(DATA_TYPE));
  cudaMalloc(&d_duration, sizeof(unsigned long long));

  // Initialize data on device (dummy data)
  cudaMemset(d_data, 1, DATA_SIZE * sizeof(DATA_TYPE));

  // Launch kernel with one block and 1024 threads (targeting L1 cache on one
  // SM)
  l1_cache_throughput_test<DATA_TYPE>
      <<<1, THREADS_PER_BLOCK>>>(d_data, d_duration, iterations);

  // Copy duration back to host
  cudaMemcpy(&h_duration, d_duration, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  // Calculate bandwidth (throughput) in GB/s
  double bandwidth =
      (double)(DATA_SIZE * iterations * sizeof(DATA_TYPE)) / (double)h_duration;

  std::string data_lable = typeid(DATA_TYPE).name();
  printf("L1 Cache Throughput: %.2f bytes/clock, using data_type: %s, with %d "
         "iterations\n",
         bandwidth, data_lable.c_str(), iterations);

  // Free memory
  cudaFree(d_data);
  cudaFree(d_duration);

  return 0;
}

int main() {

  measure_throughput<float>(32);
  measure_throughput<double>(32);
}