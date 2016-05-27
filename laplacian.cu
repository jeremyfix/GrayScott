// nvcc -o laplacian laplacian.cu -std=c++11
// nvcc -o laplacian laplacian.cu -std=c++11 -O3 -g -D_FORCE_INLINES
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>

#define WIDTH 256 
#define HEIGHT 256 

__device__
float* get_adr(float * base, size_t pitch, unsigned int row, unsigned int col) {
	return (float*)((char*) base + row * pitch) + col;
}

__global__
void gpu_laplacian( float* I, size_t pitch_I, float* laplacian,  size_t pitch_L) {
	// We suppose that width = height = number of threads

	float *vN_ptr, *vE_ptr, *vS_ptr, *vW_ptr;
	float *l_ptr,  *I_ptr;

	switch(threadIdx.x){
		case(0) :
			// Handles the first row
			// The first element
			l_ptr  = get_adr(laplacian, pitch_L, 0, 0);
			I_ptr  = get_adr(I, pitch_I, 0, 0) ;
			vN_ptr = get_adr(I, pitch_I, HEIGHT-1, 0);//(float*)((char*)I + (HEIGHT-1)*pitch_I);
			vE_ptr = get_adr(I, pitch_I, 0, 1);
			vS_ptr = get_adr(I, pitch_I, 1, 0);
			vW_ptr = get_adr(I, pitch_I, 0, WIDTH-1);
			*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4.);

			// The elements excluding the first and last
			l_ptr = get_adr(laplacian, pitch_L, 0, 1);
			I_ptr = get_adr(I, pitch_I, 0, 1);
			vN_ptr = get_adr(I, pitch_I, HEIGHT-1, 1);
			vE_ptr = get_adr(I, pitch_I, 0, 2);
			vS_ptr = get_adr(I, pitch_I, 1, 1);
			vW_ptr = get_adr(I, pitch_I, 0, 0);
			for(unsigned int i = 1 ; i < WIDTH - 1; ++i,  ++I_ptr, ++l_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr) 
				*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4);
			
			// The last element
			l_ptr = get_adr(laplacian, pitch_L, 0, WIDTH-1);
			I_ptr = get_adr(I, pitch_I, 0, WIDTH-1);
			vN_ptr = get_adr(I, pitch_I, HEIGHT-1, WIDTH-1);
			vE_ptr = get_adr(I, pitch_I, 0, 0);
			vS_ptr = get_adr(I, pitch_I, 1, WIDTH-1);
			vW_ptr = get_adr(I, pitch_I, 0, WIDTH-2);
			*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4.);

			break;
		case (HEIGHT-1):
			// Handles the last row
			// The first element
			l_ptr  = get_adr(laplacian, pitch_L, HEIGHT-1, 0);
			I_ptr  = get_adr(I, pitch_I, HEIGHT-1, 0) ;
			vN_ptr = get_adr(I, pitch_I, HEIGHT-2, 0);//(float*)((char*)I + (HEIGHT-1)*pitch_I);
			vE_ptr = get_adr(I, pitch_I, HEIGHT-1, 1);
			vS_ptr = get_adr(I, pitch_I, 0, 0);
			vW_ptr = get_adr(I, pitch_I, HEIGHT-1, WIDTH-1);
			*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4.);

			// The elements excluding the first and last
			l_ptr = get_adr(laplacian, pitch_L, HEIGHT-1, 1);
			I_ptr = get_adr(I, pitch_I, HEIGHT-1, 1);
			vN_ptr = get_adr(I, pitch_I, HEIGHT-2, 1);
			vE_ptr = get_adr(I, pitch_I, HEIGHT-1, 2);
			vS_ptr = get_adr(I, pitch_I, 0, 1);
			vW_ptr = get_adr(I, pitch_I, HEIGHT-1, 0);
			for(unsigned int i = 1 ; i < WIDTH - 1; ++i,  ++I_ptr, ++l_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr) 
				*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4);
			
			// The last element
			l_ptr = get_adr(laplacian, pitch_L, HEIGHT-1, WIDTH-1);
			I_ptr = get_adr(I, pitch_I, HEIGHT-1, WIDTH-1);
			vN_ptr = get_adr(I, pitch_I, HEIGHT-2, WIDTH-1);
			vE_ptr = get_adr(I, pitch_I, HEIGHT-1, 0);
			vS_ptr = get_adr(I, pitch_I, 0, WIDTH-1);
			vW_ptr = get_adr(I, pitch_I, HEIGHT-1, WIDTH-2);
			*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4.);

			break;
		default:
			// Handles the rows excluding the first and the last
			unsigned int row_id = threadIdx.x;
			// The first element
			l_ptr  = get_adr(laplacian, pitch_L, row_id, 0);
			I_ptr  = get_adr(I, pitch_I, row_id, 0) ;
			vN_ptr = get_adr(I, pitch_I, row_id-1, 0);
			vE_ptr = get_adr(I, pitch_I, row_id, 1);
			vS_ptr = get_adr(I, pitch_I, row_id+1, 0);
			vW_ptr = get_adr(I, pitch_I, row_id, WIDTH-1);
			*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4.);

			// The elements excluding the first and last
			l_ptr = get_adr(laplacian, pitch_L, row_id, 1);
			I_ptr = get_adr(I, pitch_I, row_id, 1);
			vN_ptr = get_adr(I, pitch_I, row_id-1, 1);
			vE_ptr = get_adr(I, pitch_I, row_id, 2);
			vS_ptr = get_adr(I, pitch_I, row_id+1, 1);
			vW_ptr = get_adr(I, pitch_I, row_id, 0);
			for(unsigned int i = 1 ; i < WIDTH - 1; ++i,  ++I_ptr, ++l_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr) 
				*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4);
			
			// The last element
			l_ptr = get_adr(laplacian, pitch_L, row_id, WIDTH-1);
			I_ptr = get_adr(I, pitch_I, row_id, WIDTH-1);
			vN_ptr = get_adr(I, pitch_I, row_id-1, WIDTH-1);
			vE_ptr = get_adr(I, pitch_I, row_id, 0);
			vS_ptr = get_adr(I, pitch_I, row_id+1, WIDTH-1);
			vW_ptr = get_adr(I, pitch_I, row_id, WIDTH-2);
			*l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*I_ptr) * 4.);

		break;
	}


}

__host__
void cpu_laplacian(unsigned int width, unsigned int height, float* values, float* laplacian) {

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
  laplacian[0] = -(4 * values[0]) + (values[(height-1)*width] + values[1] + values[width] + values[width-1]);// top left
  laplacian[width-1] = -(4 * values[width-1]) + (values[(height-1)*width + (width-1)] + values[0] + values[width + (width-1)] + values[width-2]); // top right
  laplacian[(height-1)*width] = -(4 * values[(height-1)*width]) + (values[(height-2)*width] + values[(height-1)*width + 1] + values[0] + values[(height-1)*width + (width-1)]); // bottom left
  laplacian[(height-1)*width + (width-1)] = -(4 * values[(height-1)*width + (width-1)]) + (values[(height-2)*width + (width-1)]+values[(height-1)*width] + values[width-1] + values[(height-1)*width + (width-2)]); // bottom right

  // We handle the borders
  // o########o
  // oooooooooo
  // oooooooooo
  // oooooooooo
  // oooooooooo
  v_ptr = values + 1;
  vN_ptr = values + (height-1)*width + 1;
  vE_ptr = values + 2;
  vS_ptr = values + width + 1;
  vW_ptr = values;
  l_ptr = laplacian + 1;
  for(unsigned int i = 1 ; i < width-1 ; ++i, ++v_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++l_ptr) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

  // oooooooooo
  // oooooooooo
  // oooooooooo
  // oooooooooo
  // o########o
  v_ptr = values + (height-1)*width + 1;
  vN_ptr = values + (height-2)*width + 1;
  vE_ptr = values + (height-1)*width + 2;
  vS_ptr = values + 1 ;
  vW_ptr = values + (height-1)*width;
  l_ptr = laplacian + (height-1)*width + 1;
  for(unsigned int i = 1 ; i < width-1 ; ++i, ++v_ptr, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++l_ptr) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

  // oooooooooo
  // #ooooooooo
  // #ooooooooo
  // #ooooooooo
  // #ooooooooo
  // oooooooooo
  v_ptr = values + width;
  vN_ptr = values ;
  vE_ptr = values + width + 1;
  vS_ptr = values + 2*width ;
  vW_ptr = values + width + (width-1);
  l_ptr = laplacian + width;
  for(unsigned int i = 1 ; i < height-1 ; ++i, v_ptr+=width, vN_ptr+=width, vE_ptr+=width, vS_ptr+=width, vW_ptr+=width, l_ptr+=width) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

 
  // oooooooooo
  // ooooooooo#
  // ooooooooo#
  // ooooooooo#
  // ooooooooo#
  // oooooooooo
  // 
  v_ptr = values + width + (width-1);
  vN_ptr = values + (width-1);
  vE_ptr = values + width;
  vS_ptr = values + 2*width + (width-1) ;
  vW_ptr = values + width + (width-2);
  l_ptr = laplacian + width + (width-1);
  for(unsigned int i = 1 ; i < height-1 ; ++i, v_ptr+=width, vN_ptr+=width, vE_ptr+=width, vS_ptr+=width, vW_ptr+=width, l_ptr+=width) 
    *l_ptr = (*vN_ptr + *vE_ptr + *vS_ptr + *vW_ptr) - ((*v_ptr) * 4);

  // We handle the region inside the array exlucding a border of size 1,
  // i.e. the pixels # below
  // oooooooooo
  // o########o
  // o########o
  // o########o
  // oooooooooo
  v_ptr = values + (1*width + 1);
  vN_ptr = values + 1;
  vE_ptr = values + (1*width + 2);
  vS_ptr = values + (2*width + 1);
  vW_ptr = values + (1*width + 0);

  l_ptr = laplacian + (1*width + 1);
  for(unsigned int i = 1; i < height-1; ++i) {
    for(unsigned int j = 1 ; j < width-1 ; ++j, ++vN_ptr, ++vE_ptr, ++vS_ptr, ++vW_ptr, ++v_ptr, ++l_ptr) 
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
  for(unsigned int i = 0 ; i < N; ++i, ++v1ptr, ++v2ptr) {
    d = (*v1ptr) - (*v2ptr);
    res += d*d;
  }
  return sqrt(res);
}

__host__
void printArray(float * v,  unsigned int i) {
	float* vptr = v + i * WIDTH;
	for(unsigned int j = 0 ; j < WIDTH; ++j,  ++vptr) 
		printf("%f ", (*vptr));
	printf("\n");
	
}

__host__
void printArray(float * v) {
	float* vptr = v;
	for(unsigned int i = 0 ; i < HEIGHT; ++i) {
		for(unsigned int j = 0 ; j < WIDTH; ++j,  ++vptr) 
			printf("%f ", (*vptr));
		printf("\n");
	}
}

__host__
int main(int argc, char * argv[]) {

  unsigned int width = WIDTH;
  unsigned int height = HEIGHT;
  unsigned int nbcalls = 500;

  float *I, *lcpu, *lgpu, *dI, *dlgpu;
  I = (float*) malloc(width*height*sizeof(float));
  lcpu = (float*) malloc(width*height*sizeof(float));
  lgpu = (float*) malloc(width*height*sizeof(float));

  // Initialize the input image
  float *Iptr = I;
  for(unsigned int i = 0 ; i < width*height; ++i, ++Iptr)
    (*Iptr) = std::rand() / ((float)RAND_MAX);

  //************* CPU *****************//
  std::chrono::time_point<std::chrono::system_clock> start_cpu, end_cpu;
  start_cpu = std::chrono::system_clock::now();

  for(unsigned int i = 0 ; i < nbcalls; ++i) 
    cpu_laplacian(width, height, I, lcpu);
  
  end_cpu = std::chrono::system_clock::now();
  int elapsed_cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu-start_cpu).count();
  printf("CPU elapsed : %f ms per call \n", ((float)elapsed_cpu_ms)/nbcalls);

  //************* GPU *****************//
  
  size_t pitch_dI, pitch_dlgpu;
  cudaMallocPitch(&dI, &pitch_dI, width*sizeof(float), height);
  cudaMallocPitch(&dlgpu, &pitch_dlgpu, width*sizeof(float), height);
  
  // Copy the input to the GPU
  //cudaMemcpy(dI, I, N*N*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy2D(dI, pitch_dI, I, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);
  
  // Call the kernel
  int blocksPerGrid = 1;
  //dim3 threadsPerBlock(HEIGHT, 1, 1);
	int threadsPerBlock = HEIGHT;
	std::chrono::time_point<std::chrono::system_clock> start_gpu, end_gpu;
  start_gpu = std::chrono::system_clock::now();
   
	cudaMemcpy2D(dI, pitch_dI, I, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);
  for(unsigned int i = 0 ; i < nbcalls; ++i) {
		gpu_laplacian<<<blocksPerGrid, threadsPerBlock>>>(dI, pitch_dI, dlgpu, pitch_dlgpu);
	}
  
	cudaMemcpy2D(lgpu, width*sizeof(float), dlgpu, pitch_dlgpu, width*sizeof(float), height, cudaMemcpyDeviceToHost);
	end_gpu = std::chrono::system_clock::now();
	int elapsed_gpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu-start_gpu).count();
  printf("GPU elapsed : %f ms per call \n", ((float)elapsed_gpu_ms)/nbcalls);

	// Get the result
  //cudaMemcpy(lgpu, dlgpu, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy2D(lgpu, width*sizeof(float), dlgpu, pitch_dlgpu, width*sizeof(float), height, cudaMemcpyDeviceToHost);
  
  //********** Comparison *************//
  printf("Difference : %f \n", diffNorm(lcpu, lgpu, width*height));

	// For debug,  print some elements of the array
	//printArray(I);
	//printf("\n");
	//printArray(lcpu);
	//printf("\n");
	//printArray(lgpu);

  //***********************************//
  // Free the device memory
  cudaFree(dI);
  cudaFree(dlgpu);

  // Free the host memory
  delete[] I;
  delete[] lcpu;
  delete[] lgpu;
}
