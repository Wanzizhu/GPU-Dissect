#include "cuda_runtime.h"
#include "repeat.h"
#include <numeric>
#include <stdio.h>
#include <vector>

__global__ void shared_latency_single_thread(int *my_array, int array_length,
                                             int N, long long *duration) {

  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  for (int i = tid; i < array_length; i += blockDim.x) {
    sdata[i] = my_array[i];
  }
  // syncthreads
  __syncthreads();

  int sum = 0;

  int j = 0;
  for (int i = 0; i < N - 1; ++i) {
    auto start = clock();
    j = sdata[j];
    sum += j;
    auto end = clock();
    duration[i] = end - start;
  }
  my_array[array_length - 1] = sum;
}

void parametric_measure_shared(int N, int stride) {
  int nelems = N * stride;
  if (nelems * sizeof(int) > 164 * 1024) {
    printf("Error, Shared memory size exceeds the limit\n");
    return;
  }

  int *h_a = (int *)malloc(sizeof(int) * nelems);
  long long *h_latency = (long long *)malloc(N * sizeof(long long));

  /* initialize array elements on CPU */
  memset(h_a, 0, sizeof(int) * nelems);
  for (int i = 0; i < N; i++) {
    h_a[i * stride] = (i + 1) * stride;
  }
  for (int i = 0; i < N; ++i) {
    h_a[i * stride + 1] = (i + 1) * stride + 1;
  }

  /* allocate arrays on GPU */
  cudaError_t error_id;
  int *d_a;
  long long *d_latency;
  cudaMalloc((void **)&d_a, sizeof(int) * nelems);
  cudaMalloc((void **)&d_latency, N * sizeof(long long));

  cudaDeviceSynchronize();
  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error 1 is %s\n", cudaGetErrorString(error_id));
  }

  /* copy array elements from CPU to GPU */
  cudaMemcpy((void *)d_a, (void *)h_a, sizeof(int) * nelems,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_latency, (void *)h_latency, N * sizeof(long long),
             cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error 2 is %s\n", cudaGetErrorString(error_id));
  }

  /* launch kernel*/
  dim3 Db = dim3(1);
  dim3 Dg = dim3(1, 1, 1);

  int sharedMemSize = sizeof(int) * nelems;
  shared_latency_single_thread<<<Dg, Db, sharedMemSize>>>(d_a, nelems, N,
                                                          d_latency);
  cudaDeviceSynchronize();

  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error 3 is %s\n", cudaGetErrorString(error_id));
  }

  /* copy results from GPU to CPU */
  cudaDeviceSynchronize();

  cudaMemcpy((void *)h_latency, (void *)d_latency, N * sizeof(long long),
             cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  /* print results, last element is dummy data */
  int pre_address = h_a[0];
  for (int i = 0; i < N - 1; i++) {
    int bank_id = pre_address % 32;
    printf("address: %ld, bank id: %d, latency: %llu\n",
           pre_address * sizeof(int), bank_id, h_latency[i]);
    pre_address = h_a[pre_address];
  }
  float avg_latency =
      std::accumulate(h_latency, h_latency + N - 1, 0.0) / (N - 1);
  printf("Access total %d num with stride %ld bytes, Average latency: %.3f "
         "cycles\n",
         N, stride * sizeof(int), avg_latency);

  /* free memory on GPU */
  cudaFree(d_a);
  cudaFree(d_latency);
  cudaDeviceSynchronize();

  /*free memory on CPU */
  free(h_a);
  free(h_latency);
}

int main() {

  cudaSetDevice(0);

  int N = 160;
  int stride = 2;
  printf("Using single thread to measure shared memory latency for with stride "
         "%ld.\n",
         stride * sizeof(int));
  parametric_measure_shared(N, stride);

  return 0;
}
