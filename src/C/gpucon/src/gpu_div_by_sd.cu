
#include <stdio.h>

// divide correlation matrix elements by sd(x)*sd(y)
__global__ void divide_by_value_indexed_kernel(double * d_cors, double * d_x_sds, double * d_y_sds, double factor)
{
  d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= (d_x_sds[blockIdx.x] * d_y_sds[blockIdx.y] * factor);	
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" void gpu_tut_nix() 
{
	printf("tut nix...\n");
  int ngpu, devID;
  cudaDeviceProp deviceProps;
  
  cudaGetDeviceCount(&ngpu);
  printf("CUDA-capable device count: %i\n", ngpu);

  for(devID = 0; devID < ngpu; devID++) {
    cudaGetDeviceProperties(&deviceProps, devID);
    printf("ID %i CUDA device [%s]\n", devID, deviceProps.name);
	printf("Total Global Memory: %.2f MB\n", deviceProps.totalGlobalMem / (1024.0  * 1024.0));
  }
}

extern "C" 
__host__ void gpu_div_by_sd(double * h_cors, double * h_sds, int n, size_t p, int gpuID)
{
	double * d_sds = NULL;
	double * d_cors = NULL;
  double factor = (double) (n - 1.0);
	
	// printf("\nStart of CUDA part...\n");

  // printf("debug info: n=%d p=%d C[1, 1]=%.2f gpuID=%d\n", n, p, *h_cors, gpuID);

  cudaSetDevice(gpuID);
	
	// allocate device memory
 
  cudaMalloc((void**) &d_sds, p * sizeof(double));
  cudaMalloc((void**) &d_cors, p * p * sizeof(double)); 

  cudaMemcpy(d_sds, h_sds, p * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cors, h_cors, p * p * sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimGrid(p, p);	
  dim3 dimBlock(1, 1);	
  divide_by_value_indexed_kernel<<<dimGrid,dimBlock>>>(d_cors, d_sds, d_sds, factor); 

  cudaMemcpy(h_cors, d_cors, p * p * sizeof(double), cudaMemcpyDeviceToHost);

  // printf("C[1, 1]=%.2f after operation...\n", *h_cors);

  cudaFree(d_sds);
  cudaFree(d_cors);

  // printf(" done.\n");
} 

