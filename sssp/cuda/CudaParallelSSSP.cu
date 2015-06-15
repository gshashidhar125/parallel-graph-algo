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

__device__ int *graph[3], d_numVertices, d_numEdges, *worklist, d_tail;
__device__ int *dist;
__device__ bool d_terminate;

__global__ void CudaInitialize(int *edgeArray1, int *edgeArray2, int *weightArray, int *distanceArray, int numVertices, int numEdges) {

    d_numVertices = numVertices;
    d_numEdges = numEdges;
    graph[0] = edgeArray1;
    graph[1] = edgeArray2;
    graph[2] = weightArray;
    dist = distanceArray;
}
__global__ void CudaPrintGraph() {

    print("\nEdge Array:\n");
    for (int i = 0; i < d_numEdges + 1; i++)
        print(" %d-%d[%d]", graph[0][i], graph[1][i], graph[2][i]);
    print("\n");
}
__global__ void Cuda_SSSP() {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > d_numEdges)
        return;

    // Edge (u, v)
    int u = graph[0][tid];
    int v = graph[1][tid];
    for(int i = 0; i < d_numVertices; i++) {
        if (dist[v] < dist[u] + graph[2][tid])
            dist[v] = dist[u] + graph[2][tid];
    }
}
void CudaGraphClass::callSSSP() {

    print("Hello inside cuda code\n");
    int terminate = false;
    while (terminate == false) {
        terminate = true;
//        outs("Queue: Head: %d, Tail: %d\n", currentQueueHead, currentQueueTail);
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
//        cudaMemcpyToSymbol(d_tail, &tail, sizeof(bool), 0, cudaMemcpyHostToDevice);
        Cuda_SSSP<<<2, 5>>>();
        cudaThreadSynchronize();
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    }
}

int CudaGraphClass::copyGraphToDevice() {

    int *edgeArray1, *edgeArray2, *weightArray, *distanceArray;
    cudaError_t err;
    err = cudaMalloc((void **)&edgeArray1, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&edgeArray2, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&weightArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&distanceArray, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray1, row[0], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray2, row[1], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(weightArray, row[2], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    CudaInitialize<<<1, 1>>>(edgeArray1, edgeArray2, weightArray, distanceArray, numVertices, numEdges);
    cudaThreadSynchronize();
    return 0;
}

void CudaGraphClass::populate(char *fileName) {

    inputFile.open(fileName);
    if (!inputFile.is_open()){
        cout << "invalid file";
        return;
    }

    srand(time(NULL));

    cout << numVertices << "--" << numEdges << endl;
    int i = 0, j, k;
    inputFile >> j >> k;
    while(i != numEdges) {

        //scanf("%d %d", &j, &k);
        inputFile >> j >> k;
        cout << "Read: " << j << "-- " << k;
        row[0][i] = j;
        row[1][i] = k;
        row[2][i] = (rand() % 2) ? rand() % 10 - 10 : rand() % 10;
        i++;
    }
}
void CudaGraphClass::printGraph() {
    CudaPrintGraph<<<1, 1>>>();
    cudaThreadSynchronize();
}
