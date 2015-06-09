#include <iostream>
#include "CudaGraph.h"
#define cprint(...) printMutex.lock(); printf(__VA_ARGS__); printMutex.unlock();
#define p(x) printMutex.lock(); cout << x << endl; printMutex.unlock();

#define CUDA_ERR_CHECK  \
if( err != cudaSuccess) { \
    printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__); \
    return EXIT_FAILURE; \
} \

using namespace std;

__device__ int graph[2], *worklist, d_tail;
__device__ bool d_terminate;
__global__ void Cuda_BFS(CudaGraphClass *graphData) {
    
    printf("BlockId = %d, Thread ID : %d\n", blockIdx.x, threadIdx.x);
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
}

void CudaGraphClass::callBFS() {

    print("Hello inside cuda code\n");
    copyGraphToDevice();
    
    int terminate = false;
    while (terminate == false) {
        terminate = true;
        outs("Queue: Head: %d, Tail: %d\n", currentQueueHead, currentQueueTail);
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
//        cudaMemcpyToSymbol(d_tail, &tail, sizeof(bool), 0, cudaMemcpyHostToDevice);
        Cuda_BFS<<<2, 5>>>(this);
        cudaThreadSynchronize();
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    }
}
int CudaGraphClass::copyGraphToDevice() {

    cudaError_t err;
    err = cudaMalloc((void **)&graph[0], (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&graph[1], (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(graph[0], row[0], (numVertices + 2) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(graph[1], row[1], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    return 0;
}
