#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <cuda.h>
#define N 10000000
#define N_THREADS_PER_BLOCK 1024

#define PI  3.14159265358979323846
double calculate_pi(double eps) {

    double prev_pi;
    double pi=2;
    double d=1.0;
    int i=2;
    do{
        prev_pi=pi;
        
        d=(double)(i+1);
        pi *=(i/d)*((i+2)/(d));
        i+=2;
        //printf("%f,%f,%f,%f,%f\n",ni,d,ni,d+1,pi);
    }while(abs(pi*2-prev_pi*2)>eps);
    return pi*2;

}

__device__ double atomicMul(double* address, double val) 
{ 
  unsigned long long int* address_as_ull = (unsigned long long int*)address; 
  unsigned long long int old = *address_as_ull, assumed; 
  do { 
 assumed = old; 
 old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed))); 
 } while (assumed != old); return __longlong_as_double(old);
} 

__global__ void calpia(double *pi,double eps) {
    //__shared__ double sharedData[N_THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //int localIdx = threadIdx.x;
    //printf("XDDD%d\n",blockIdx.x);
    double localSum=1.0 ;
    //tid=tid+1;
    //localSum=1.0;
    //double d;
    while (tid < N) {
        if(tid % 2 == 0){
        double numerator = (double)(tid*1.0);
        //double denominator = (double)((tid ) + 1);
        localSum *= ((numerator / (numerator-1.0)) * (numerator / (numerator+1.0 )));
        
        }
        tid += (blockDim.x * gridDim.x);
    }
    
    __syncthreads();
    pi[blockIdx.x*blockDim.x+threadIdx.x] = localSum;
    
    
   
}

// funkcja pomocnicza (dla miłośników programowania defensywnego)
void checkCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess == err) {
 fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
// exit(EXIT_FAILURE);
 }
}

int main() {
    //int num_samples = 1000000; // 
    double eps = 1e-20;         // Wartość akceptowalnej różnicy
    double pi;
 
    //double *a_d; // wskaźnik do zamapowanej pamięci urządzenia
    double *sumDev,*sumHost;
    size_t size = N * sizeof(double);
 
    clock_t start, end;
    sumHost = (double*)malloc(size); 
    
    start = clock();
    pi = calculate_pi( eps);
    end = clock();

    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Przybliżona wartość liczby pi: %.15f(error %.15f)\n", pi,PI-(pi));
    printf("Czas obliczeń: %f sekundy\n", cpu_time_used);

    cudaDeviceProp deviceProp;
    #if CUDART_VERSION < 2020
    #error "To urzadzenie nie wspiera mapowania pamieci ;(\n"
    #endif
     // Pobierz własności i sprawdź, czy urządzenie #0 wspiera mapowanie
     cudaGetDeviceProperties(&deviceProp, 0);
     //checkCUDAError("cudaGetDeviceProperties");
     if(!deviceProp.canMapHostMemory) {
     fprintf(stderr, "Urzadzenie %d nie wspiera mapowania pamieci ;(\n", 0);
     exit(EXIT_FAILURE);
     }
     
     double pi2;
start = clock();

    

    cudaMalloc((void **) &sumDev, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
    int blockSize = N_THREADS_PER_BLOCK;
    int nBlocks = N / blockSize + (N % blockSize > 0 ? 1 : 0);
    
    // Call the CUDA kernel
    calpia<<<nBlocks, blockSize>>>(sumDev, eps);
    cudaDeviceSynchronize();
    // Copy the result back to the host
   cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
   pi2=1.0;
    for(int i=0; i<N; i++){
		
		if(sumHost[i]==0){
    		//printf("%f\n",sumHost[i]);
		}
		else
		{
		pi2 *= sumHost[i];
		}}
end = clock();

  double gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Przybliżona wartość liczby pi2:%.15f(error %.15f)\n", pi2*2,PI-(pi2*2));
    printf("Czas obliczeń2:%f sekundy\n", gpu_time_used);
    free(sumHost); 
	cudaFree(sumDev);
    return 0;
}

