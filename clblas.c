#include <sys/types.h>
#include <stdio.h>
#include <time.h>
#include <CL/cl.h>

/* Include the clBLAS header. It includes the appropriate OpenCL headers */
// #include <clBLAS.h>
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */

#define M 1000
#define N 1000
#define K 1000

static const cl_float alpha = 10;

static cl_float A[M*K];
static const size_t lda = K;        /* i.e. lda = K */

static cl_float B[K*N];

static const size_t ldb = N;        /* i.e. ldb = N */

static const cl_float beta = 20;

static cl_float C[M*N];

static const size_t ldc = N;        /* i.e. ldc = N */

static cl_float result[M*N];

int main( void )
{
  cl_int err;
  cl_platform_id platform = 0;
  cl_device_id device = 0;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx = 0;
  cl_command_queue queue = 0;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  int ret = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      if (i == j)
	A[i * M + j] = 1;
      else
	A[i * M + j] = 0;
    }
  }

  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j)
	B[i * K + j] = 1;
      else
	B[i * K + j] = 0;
    }
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j)
	C[i * M + j] = 1;
      else
	C[i * M + j] = 0;
    }
  }
  
  /* Setup OpenCL environment. */
  err = clGetPlatformIDs( 1, &platform, NULL );
  err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
  queue = clCreateCommandQueue( ctx, device, 0, &err );

  /* Setup clBLAS */
  err = clblasSetup( );

  /* Prepare OpenCL memory objects and place matrices inside them. */
  bufA = clCreateBuffer( ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
			 NULL, &err );
  bufB = clCreateBuffer( ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
			 NULL, &err );
  bufC = clCreateBuffer( ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
			 NULL, &err );

  err = clEnqueueWriteBuffer( queue, bufA, CL_TRUE, 0,
			      M * K * sizeof( *A ), A, 0, NULL, NULL );
  err = clEnqueueWriteBuffer( queue, bufB, CL_TRUE, 0,
			      K * N * sizeof( *B ), B, 0, NULL, NULL );
  err = clEnqueueWriteBuffer( queue, bufC, CL_TRUE, 0,
			      M * N * sizeof( *C ), C, 0, NULL, NULL );


  clock_t begin = clock();
  /* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
  err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
		     M, N, K,
		     alpha, bufA, 0, lda,
		     bufB, 0, ldb, beta,
		     bufC, 0, ldc,
		     1, &queue, 0, NULL, &event );

  /* Wait for calculations to be finished. */
  err = clWaitForEvents( 1, &event );
  clock_t end = clock();

  double runtime = (double)(end - begin) / CLOCKS_PER_SEC;
  printf ("%lf\n", runtime);
  
  /* Fetch results of calculations from GPU memory. */
  err = clEnqueueReadBuffer( queue, bufC, CL_TRUE, 0,
			     M * N * sizeof(*result),
			     result, 0, NULL, NULL );

  /* Release OpenCL memory objects. */
  clReleaseMemObject( bufC );
  clReleaseMemObject( bufB );
  clReleaseMemObject( bufA );

  /* Finalize work with clBLAS */
  clblasTeardown( );

  /* Release OpenCL working objects. */
  clReleaseCommandQueue( queue );
  clReleaseContext( ctx );

  return ret;


}
