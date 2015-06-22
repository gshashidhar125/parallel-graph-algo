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

NullBuffer Cuda_null_buffer;
std::ostream Cuda_null_stream(&Cuda_null_buffer);

__device__ int *graph[3], d_numVertices, d_numEdges, *worklist, d_tail;
__device__ bool d_terminate;

__global__ void CudaInitialize(int *edgeArray1, int *edgeArray2, int *weightArray, int numVertices, int numEdges) {

    d_numVertices = numVertices;
    d_numEdges = numEdges;
    graph[0] = edgeArray1;
    graph[1] = edgeArray2;
    graph[2] = weightArray;
}
__global__ void CudaPrintGraph() {

    outs("\nEdge Array:\n");
    for (int i = 0; i < d_numEdges + 1; i++)
        outs(" %d-%d[%d]", graph[0][i], graph[1][i], graph[2][i]);
    outs("\n");
}
__global__ void Cuda_SSSP(int *distance) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > d_numEdges)
        return;

    // Edge (u, v)
    int u = graph[0][tid];
    int v = graph[1][tid];
    for(int i = 0; i < d_numVertices; i++) {
        if (distance[v] < distance[u] + graph[2][tid])
            distance[v] = distance[u] + graph[2][tid];
    }
}
int CudaGraphClass::callSSSP() {

    int terminate = false, *distance, *d_distance;
    cudaError_t err;

    distance = new int[(numVertices + 1)];
    err = cudaMalloc((void **)&d_distance, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(d_distance, 0xff, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;

    while (terminate == false) {
        terminate = true;
//        outs("Queue: Head: %d, Tail: %d\n", currentQueueHead, currentQueueTail);
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
//        cudaMemcpyToSymbol(d_tail, &tail, sizeof(bool), 0, cudaMemcpyHostToDevice);
        //Cuda_SSSP<<<2, 5>>>(d_distance);
        //cudaThreadSynchronize();
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
    }
    err = cudaMemcpy(distance, d_distance, (numVertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    out << "Distances:\n";
    for (int i = 0; i <= numVertices + 1; i++)
        out << "["<< i << "] = " << distance[i] << endl;

    return 0;
}

int CudaGraphClass::copyGraphToDevice() {

    int *edgeArray1, *edgeArray2, *weightArray;
    cudaError_t err;
    err = cudaMalloc((void **)&edgeArray1, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&edgeArray2, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&weightArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray1, row[0], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray2, row[1], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(weightArray, row[2], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    CudaInitialize<<<1, 1>>>(edgeArray1, edgeArray2, weightArray, numVertices, numEdges);
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

    int i = 0, j, k;
    inputFile >> j >> k;
    while(i != numEdges) {

        //scanf("%d %d", &j, &k);
        inputFile >> j >> k;
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
