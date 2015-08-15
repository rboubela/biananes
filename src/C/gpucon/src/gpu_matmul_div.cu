
#include <cublas.h>
#include <stdio.h>

// divide correlation matrix elements by sd(x)*sd(y)
__global__ void divide_by_value_kernel(double * d_cors, double * d_x_sds, double * d_y_sds, double factor)
{
  d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= (d_x_sds[blockIdx.x] * d_y_sds[blockIdx.y] * factor);	
}

extern "C"
__host__ void gpu_matmul_div(double * h_x, double * h_y, double * h_res, double * h_x_sds, double * h_y_sds, size_t n, size_t px, size_t py, int gpuID) 
{
  double * d_x = NULL;  
  double * d_y = NULL;  
  double * d_res = NULL;  
	double * d_x_sds = NULL;
	double * d_y_sds = NULL;
  double factor = (double) (n - 1.0);

  cudaSetDevice(gpuID);
  // printf("set device\n");

	cublasInit();
  // printf("init\n");
	cublasAlloc(n * px, sizeof(double), (void**) &d_x);
	cublasAlloc(px, sizeof(double), (void**) &d_x_sds);
	cublasAlloc(n * py, sizeof(double), (void**) &d_y);
	cublasAlloc(py, sizeof(double), (void**) &d_y_sds);
	cublasAlloc(px * py, sizeof(double), (void**) &d_res);
  // printf("alloc\n");
	
	// copy input data to gpu
	cublasSetVector(n * px, sizeof(double), h_x, 1, d_x, 1);
	cublasSetVector(px, sizeof(double), h_x_sds, 1, d_x_sds, 1);
	cublasSetVector(n * py, sizeof(double), h_y, 1, d_y, 1);
	cublasSetVector(py, sizeof(double), h_y_sds, 1, d_y_sds, 1);
  // printf("set res\n");

  // printf("dgemm... ");
	cublasDgemm('T', 'N', px, py, n, 1.0, d_x, n, d_y, n, 0.0, d_res, px);
  // printf("done.\n");

  // not optimal: too many blocks... just one thread per block
  // TODO optimize
  dim3 dimGrid(px, py);	
  dim3 dimBlock(1, 1);	
  divide_by_value_kernel<<<dimGrid,dimBlock>>>(d_res, d_x_sds, d_y_sds, factor);  

  cublasGetVector(px * py, sizeof(double), d_res, 1, h_res, 1);
  // printf("get res\n");
  
  cublasFree(d_x);
  cublasFree(d_x_sds);
  cublasFree(d_y);
  cublasFree(d_y_sds);
  cublasFree(d_res);
  cublasShutdown();
}

