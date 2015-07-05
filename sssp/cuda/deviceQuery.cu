#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Global Memory(GB): %lu\n", prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Shared Memory per Block: %d\n", prop.sharedMemPerBlock);
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels: %d\n", prop.concurrentKernels);
    printf("  L2 Cache Size: %d\n", prop.l2CacheSize);
  }
  /*cudaError_t err;
  int *edgeArray, numEdges = 4;
  err = cudaMalloc((void **)&edgeArray, (numEdges + 1) * sizeof(int));
    if( err != cudaSuccess) {
        printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        return EXIT_FAILURE;
    }*/
}
