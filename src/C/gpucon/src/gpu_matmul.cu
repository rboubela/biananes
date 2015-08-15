
#include <cublas.h>
#include <stdio.h>

extern "C"
__host__ void gpu_matmul(double * h_x, double * h_res, size_t n, size_t p, int gpuID) 
{
  double * d_x = NULL;  
  double * d_res = NULL;  

  cudaSetDevice(gpuID);
  // printf("set device\n");

	cublasInit();
  // printf("init\n");
	cublasAlloc(n*p, sizeof(double), (void**) &d_x);
	cublasAlloc(p*p, sizeof(double), (void**) &d_res);
  // printf("alloc\n");
	
	// copy input data to gpu
	cublasSetVector( n*p, sizeof(double), h_x, 1, d_x, 1);
  // printf("set res\n");

  // printf("dgemm... ");
	cublasDgemm('T', 'N', p, p, n, 1.0, d_x, n, d_x, n, 0.0, d_res, p);
  // printf("done.\n");

  cublasGetVector(p*p, sizeof(double), d_res, 1, h_res, 1);
  // printf("get res\n");
  
  cublasFree(d_x);
  cublasFree(d_res);
  cublasShutdown();
}

