// nvcc -o laplacian laplacian.cu -std=c++11
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <cmath>

__global__
void gpu_laplacian(unsigned int N, float* values, float* laplacian) {
}

__host__
void cpu_laplacian(unsigned int N, float* values, float* laplacian) {

  // For efficiently computing the laplacian we will move 5 pointers on values
  // pointing : on the pixel, on its north, on its east, on its west, on its south
  float* v_ptr, *vN_ptr, *vE_ptr, *vS_ptr, *vW_ptr;
  float* l_ptr;

  // We handle the corners
  // #oooooo#
  // oooooooo
  // oooooooo
  // oooooooo
  // #oooooo#
  laplacian[0] = -(4 * values[0]) + (values[(N-1)*N] + values[1] + values[N] + values[(N-1)]);// top left
  laplacian[N-1] = -(4 * values[N-1]) + (values[(N-1)*N + (N-1)] + values[0] + values[N + (N-1)] + values[N-2]); // top right
  laplacian[(N-1)*N] = -(4 * values[(N-1)*N]) + (values[(N-2)*N] + values[(N-1)*N + 1] + values[0] + values[(N-1)*N + (N-1)]); // bottom left
  laplacian[(N-1)*N + (N-1)] = -(4 * values[(N-1)*N + (N-1)]) + (values[(N-2)*N + (N-1)]+values[(N-1)*N] + values[N-1] + values[(N-1)*N + (N-2)]); // bottom right

  // We handle the borders
  // o########o
  // oooooooooo
  // oooooooooo
  // oooooooooo
  // oooooooooo
  v_ptr = values + 1;
  vN_ptr = values + (N-1)*N + 1;
  vE_ptr = values + 2;
  vS_ptr = values + N + 1;
  vW_ptr = values;
  l_ptr = laplacian + 1;
  for(unsigned int i = 1 ; i < N-1 ; ++i, ++v_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++l_ptr) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

  // oooooooooo
  // oooooooooo
  // oooooooooo
  // oooooooooo
  // o########o
  v_ptr = values + (N-1)*N + 1;
  vN_ptr = values + (N-2)*N + 1;
  vE_ptr = values + (N-1)*N + 2;
  vS_ptr = values + 1 ;
  vW_ptr = values + (N-1)*N;
  l_ptr = laplacian + (N-1)*N + 1;
  for(unsigned int i = 1 ; i < N-1 ; ++i, ++v_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++l_ptr) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

  // oooooooooo
  // #ooooooooo
  // #ooooooooo
  // #ooooooooo
  // #ooooooooo
  // oooooooooo
  v_ptr = values + N;
  vN_ptr = values ;
  vE_ptr = values + N + 1;
  vS_ptr = values + 2*N ;
  vW_ptr = values + N + (N-1);
  l_ptr = laplacian + N;
  for(unsigned int i = 1 ; i < N-1 ; ++i, v_ptr+=N, vN_ptr+=N, vE_ptr+=N, vS_ptr+=N, vW_ptr+=N, l_ptr+=N) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

 
  // oooooooooo
  // ooooooooo#
  // ooooooooo#
  // ooooooooo#
  // ooooooooo#
  // oooooooooo
  // 
  v_ptr = values + N + (N-1);
  vN_ptr = values + (N-1);
  vE_ptr = values + N;
  vS_ptr = values + 2*N + (N-1) ;
  vW_ptr = values + N + (N-2);
  l_ptr = laplacian + N + (N-1);
  for(unsigned int i = 1 ; i < N-1 ; ++i, v_ptr+=N, vN_ptr+=N, vE_ptr+=N, vS_ptr+=N, vW_ptr+=N, l_ptr+=N) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

  // We handle the region inside the array exlucding a border of size 1,
  // i.e. the pixels # below
  // oooooooooo
  // o########o
  // o########o
  // o########o
  // oooooooooo
  v_ptr = values + (1*N + 1);
  vN_ptr = values + 1;
  vE_ptr = values + (1*N + 2);
  vS_ptr = values + (2*N + 1);
  vW_ptr = values + (1*N + 0);

  l_ptr = laplacian + (1*N + 1);
  for(unsigned int i = 1; i < N-1; ++i) {
    for(unsigned int j = 1 ; j < N-1 ; ++j, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++v_ptr, ++l_ptr) 
      *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

    // For switching to the next line we must move the pointers forward by 2 pixels
    v_ptr += 2;
    l_ptr += 2;
    vN_ptr += 2;
    vE_ptr += 2;
    vS_ptr += 2;
    vW_ptr += 2;
  }
} 

__host__
float diffNorm(float* v1, float* v2, unsigned int N) {
  float res = 0.0;
  float* v1ptr = v1;
  float* v2ptr = v2;
  float d;
  for(unsigned int i = 0 ; i < N*N; ++i, ++v1ptr, ++v2ptr) {
    d = (*v1ptr) - (*v2ptr);
    res += d*d;
  }
  return sqrt(d);
}

__host__
int main(int argc, char * argv[]) {
  unsigned int N = 256;
  unsigned int nbcalls = 100;

  float *I, *lcpu, *lgpu, *dI, *dlgpu;
  I = (float*) malloc(N*N*sizeof(float));
  lcpu = (float*) malloc(N*N*sizeof(float));
  lgpu = (float*) malloc(N*N*sizeof(float));

  cudaMalloc(&dI, N*N*sizeof(float));
  cudaMalloc(&dlgpu, N*N*sizeof(float));

  // Initialize the input image
  float *Iptr = I;
  for(unsigned int i = 0 ; i < N*N; ++i, ++Iptr)
    (*Iptr) = std::rand() / ((float)RAND_MAX);

  //************* CPU *****************//
  std::chrono::time_point<std::chrono::system_clock> start_cpu, end_cpu;
  start_cpu = std::chrono::system_clock::now();

  for(unsigned int i = 0 ; i < nbcalls; ++i) 
    cpu_laplacian(N, I, lcpu);
  
  end_cpu = std::chrono::system_clock::now();
  int elapsed_cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu-start_cpu).count();
  printf("CPU elapsed : %f ms per call \n", ((float)elapsed_cpu_ms)/nbcalls);
  //************* GPU *****************//

  std::chrono::time_point<std::chrono::system_clock> start_gpu, end_gpu;
  start_gpu = std::chrono::system_clock::now();
  
  // Copy the input to the GPU
  cudaMemcpy(dI, I, N*N*sizeof(float), cudaMemcpyHostToDevice);
  
  // Call the kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = 1;
  gpu_laplacian<<<blocksPerGrid, threadsPerBlock>>>(N, dI, dlgpu);
  
  // Get the result
  cudaMemcpy(lgpu, dlgpu, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  
  end_gpu = std::chrono::system_clock::now();
  int elapsed_gpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu-start_gpu).count();
  printf("GPU elapsed : %f ms per call \n", ((float)elapsed_gpu_ms)/nbcalls);

  //********** Comparison *************//
  printf("Difference : %f \n", diffNorm(lcpu, lgpu, N));
  
  //***********************************//
  // Free the memory
  cudaFree(dI);
  cudaFree(dlgpu);

  delete[] I;
  delete[] lcpu;
  delete[] lgpu;
}
