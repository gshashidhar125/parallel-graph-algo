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

__device__ int *graph[3], d_numVertices, d_numEdges, *d_worklist, d_tail;
__device__ bool d_terminate;

__global__ void CudaInitialize(int *vertexArray, int *edgeArray, int *weightArray, int *worklist, int numVertices, int numEdges) {

    d_numVertices = numVertices;
    d_numEdges = numEdges;
    graph[0] = vertexArray;
    graph[1] = edgeArray;
    graph[2] = weightArray;
    d_worklist = worklist;
    d_worklist[1] = 1;
}
__global__ void CudaPrintGraph() {

    print("Vertex Array:\n");
    for (int i = 0; i < d_numVertices + 2; i++)
        print(" %d", graph[0][i]);
    print("\n");
    print("Edge Array:\n");
    for (int i = 0; i < d_numEdges + 1; i++)
        print(" %d[%d]", graph[1][i], graph[2][i]);
    print("\n");
}
__global__ void Cuda_SSSP() {
    
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId > d_numEdges)
        return;

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
        Cuda_SSSP<<<2, 5>>>();
        cudaThreadSynchronize();
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

    int *vertexArray, *edgeArray, *weightArray, *worklist;
    cudaError_t err;
    err = cudaMalloc((void **)&vertexArray, (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&edgeArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&weightArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&worklist, (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(vertexArray, row[0], (numVertices + 2) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray, row[1], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(weightArray, row[2], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    CudaInitialize<<<1, 1>>>(vertexArray, edgeArray, weightArray, worklist, numVertices, numEdges);
    cudaThreadSynchronize();
    return 0;
}

void CudaGraphClass::populate(char *fileName) {

    inputFile.open(fileName);
    if (!inputFile.is_open()){
        cout << "invalid file";
        return;
    }

    cout << numVertices << "--" << numEdges << endl;
    int **AdjMatrix, i, j, k;
    AdjMatrix = new int* [numVertices + 1]();
    for (i = 0; i <= numVertices; i++) {

        AdjMatrix[i] = new int [numVertices + 1]();
    }
    i = numEdges;
    int lastj = 1, currentIndex = 1;
    inputFile >> j >> k;
    srand(time(NULL));
    while(i) {

        //scanf("%d %d", &j, &k);
        inputFile >> j >> k;
        cout << "Read: " << j << "-- " << k;
        AdjMatrix[j][k] = 1;
        while (lastj <= j || lastj == 1) {
            if (lastj == 1) {
                row[0][0] = currentIndex;
                row[0][1] = currentIndex;
            }else {
                row[0][lastj] = currentIndex;
            }
            lastj++;
        }
//        if (AdjMatrix[k][j] != 1)
        row[1][currentIndex] = k;
        row[2][currentIndex] = (rand() % 2) ? rand() % 10 - 10 : rand() % 10;
        currentIndex ++;
        i--;
    }
    row[1][0] = 0;
    // Sentinel node just points to the end of the last node in the graph
    while (lastj <= numVertices + 1) {
        row[0][lastj] = currentIndex;
        lastj++;
    }
    //row[0][lastj+1] = currentIndex;
    for (i = 1; i <= numVertices + 1; i++)
        print("Vertex: %d = %d\n", i, row[0][i]);

    print("Second Array:\n");
    for (i = 1; i <= numEdges; i++)
        print("Edges: Index: %d, Value = %d\n", i, row[1][i]);

    j = 1;
    for (i = 1; i <= numVertices; i++) {

        currentIndex = row[0][i];
        while (currentIndex < row[0][i+1]) {
//            print("%d %d\n", i, row[1][currentIndex]);
            if (AdjMatrix[i][row[1][currentIndex]] != 1 /*&&
                AdjMatrix[row[1][currentIndex]][i] != 1*/) {
                outs("\n\nGraph Do not Match\n\n");
                break;
            }
            j++;
            currentIndex ++;
        }
    }
}
void CudaGraphClass::printGraph() {
    CudaPrintGraph<<<1, 1>>>();
    cudaThreadSynchronize();
}
