#include <algorithm>
#include <numeric>
#include <stdint.h>
#include <stdio.h>
#include <string>

#include "cuda_runtime.h"

#define Elems 160

#define CacheLevel 0

__forceinline__ __device__ void ptx_load(unsigned int *ptr,
                                         unsigned int *value) {
#if CacheLevel == 0 // loading from global memory
  asm volatile("ld.global.cv.u32 %0, [%1];\n\t" : "=r"(*value) : "l"(ptr));
#elif CacheLevel == 1 // loading from L2
  asm volatile("ld.shared.cg.u32 %0, [%1];\n\t" : "=r"(*value) : "l"(ptr));
#elif CacheLevel == 2
  asm volatile("ld.global.ca.u32 %0, [%1];\n\t" : "=r"(*value) : "l"(ptr));
#endif
}

__global__ void global_latency(unsigned int *my_array, int array_length,
                               int iterations, unsigned int *duration,
                               unsigned int *index, int warm_up_stride,
                               bool do_tlb_warmup) {

  unsigned int start_time, end_time;

  __shared__ unsigned int s_tvalue[Elems];
  __shared__ unsigned int s_index[Elems];

  int k;

  for (k = 0; k < Elems; k++) {
    s_index[k] = 0;
    s_tvalue[k] = 0;
  }

  // first round to warm up TLB
  if (do_tlb_warmup) {
    unsigned int j = warm_up_stride;
    for (k = 0; k < iterations * Elems; k++) {
      // write ptx for below two lines
      start_time = clock();
      ptx_load(my_array + j, &j);
      unsigned int m = j + 17;
      s_index[k] = m;
      s_tvalue[k] = end_time - start_time;
    }
  }

  // second round
  unsigned int j = 0;
  for (k = 0; k < iterations * Elems; k++) {
    // write ptx for below two lines
    start_time = clock();
    ptx_load(my_array + j, &j);
    unsigned int m = j + 17;
    end_time = clock();
    s_index[k] = m;
    s_tvalue[k] = end_time - start_time;
  }

  my_array[array_length] = j;
  my_array[array_length + 1] = my_array[j];

  for (k = 0; k < Elems; k++) {
    index[k] = s_index[k];
    duration[k] = s_tvalue[k];
  }
}

void parametric_measure_latency(int N, int iterations, int access_stride,
                               int warm_up_stride, bool do_tlb_warmup, float unit, std::string unit_label) {
  cudaDeviceReset();

  cudaError_t error_id;

  int i;
  unsigned int *h_a;
  /* allocate arrays on CPU */
  h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N + 2));
  unsigned int *d_a;
  /* allocate arrays on GPU */
  error_id = cudaMalloc((void **)&d_a, sizeof(unsigned int) * (N + 2));
  if (error_id != cudaSuccess) {
    printf("Error 1.0 is %s\n", cudaGetErrorString(error_id));
  }

  /* initialize array elements*/
  for (i = 0; i < N; i++)
    h_a[i] = 0;

  for (int i = 0; i < Elems; ++i) {
    h_a[i * access_stride] = (i + 1) * access_stride;
    h_a[i * access_stride + warm_up_stride] =
        (i + 1) * access_stride + warm_up_stride;
  }

  /* copy array elements from CPU to GPU */
  error_id =
      cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
  if (error_id != cudaSuccess) {
    printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
  }

  unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int) * Elems);
  unsigned int *h_timeinfo =
      (unsigned int *)malloc(sizeof(unsigned int) * Elems);

  unsigned int *duration;
  error_id = cudaMalloc((void **)&duration, sizeof(unsigned int) * Elems);
  if (error_id != cudaSuccess) {
    printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
  }

  unsigned int *d_index;
  error_id = cudaMalloc((void **)&d_index, sizeof(unsigned int) * Elems);
  if (error_id != cudaSuccess) {
    printf("Error 1.3 is %s\n", cudaGetErrorString(error_id));
  }

  cudaDeviceSynchronize();
  /* launch kernel*/
  dim3 Db = dim3(1);
  dim3 Dg = dim3(1, 1, 1);

  global_latency<<<Dg, Db>>>(d_a, N, iterations, duration, d_index,
                             warm_up_stride, do_tlb_warmup);

  cudaDeviceSynchronize();

  error_id = cudaGetLastError();
  if (error_id != cudaSuccess) {
    printf("Error kernel is %s\n", cudaGetErrorString(error_id));
  }

  /* copy results from GPU to CPU */
  cudaDeviceSynchronize();

  error_id = cudaMemcpy((void *)h_timeinfo, (void *)duration,
                        sizeof(unsigned int) * Elems, cudaMemcpyDeviceToHost);
  if (error_id != cudaSuccess) {
    printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
  }
  error_id = cudaMemcpy((void *)h_index, (void *)d_index,
                        sizeof(unsigned int) * Elems, cudaMemcpyDeviceToHost);
  if (error_id != cudaSuccess) {
    printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
  }

  cudaDeviceSynchronize();

  unsigned int pre_idx = 0;
  printf("stride %s, latency\n", unit_label.c_str());
  for (i = 0; i < Elems; i++) {
    printf("%.1f, %d \n", float(pre_idx) * sizeof(unsigned int) / unit,
           h_timeinfo[i]);
    pre_idx = h_a[pre_idx];
  }
  auto max = std::max_element(h_timeinfo, h_timeinfo + Elems);
  auto min = std::min_element(h_timeinfo, h_timeinfo + Elems);
  unsigned int avg = std::accumulate(h_timeinfo, h_timeinfo + Elems, 0) / Elems;
  printf("max_latency: %d, stride: %.1f\n", *max,
         float(max - h_timeinfo) * access_stride / 1024.0f / 1024.0f);
  printf("min_latency: %d, stride: %.1f\n", *min,
         float(min - h_timeinfo) * access_stride / 1024.0f / 1024.0f);
  printf("avg_latency: %d\n", avg);

  /* free memory on GPU */
  cudaFree(d_a);
  cudaFree(d_index);
  cudaFree(duration);

  /*free memory on CPU */
  free(h_a);
  free(h_index);
  free(h_timeinfo);

  cudaDeviceReset();
}
void measure_latency() {
  int warm_up_stride = 64;
  
#if CacheLevel == 0
  std::vector<int> access_stride_list{1024 * 1024, 512 * 1024, 256 * 1024};
  float unit = 1024 * 1024;
  std::string unit_label = "MB";
#elif CacheLevel == 1
  std::vector<int> access_stride_list{1024, 512, 16};
  float unit = 1024;
  std::string unit_label = "KB";
#elif CacheLevel == 2
  std::vector<int> access_stride_list{32, 64, 128};
  float unit = 1;
  std::string unit_label = "B";
#endif

  for (auto access_stride : access_stride_list) {
    int N = Elems * access_stride;
    int iterations = 1;
    printf("\nMeauring global memory Latency, %.4f %s, array, with stride %.3f %s"
           "M ====\n",
           sizeof(int) * (float)N / unit, unit_label.c_str(),
           float(access_stride * sizeof(int)) / unit, unit_label.c_str());

    parametric_measure_latency(N, iterations, access_stride, warm_up_stride,
                              false, unit, unit_label);
    parametric_measure_latency(N, iterations, access_stride, warm_up_stride,
                              true, unit, unit_label);
    printf("===============================================\n\n");
  }
}

int main() {
  cudaSetDevice(0);
  measure_latency();
  cudaDeviceReset();
  return 0;
}