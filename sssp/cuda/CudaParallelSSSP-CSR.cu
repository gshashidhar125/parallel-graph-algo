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

__device__ int *graph[3], d_numVertices, d_numEdges, *d_worklist, d_worklistLength;
__device__ int *d_inputPrefixSum, *d_prefixSum;
__device__ int d_prefixLevel;
__device__ bool d_terminate;

__global__ void CudaInitialize(int *vertexArray, int *edgeArray, int *weightArray, int *worklist, int *inputPrefixSum, int *prefixSum, int numVertices, int numEdges) {

    d_numVertices = numVertices;
    d_numEdges = numEdges;
    graph[0] = vertexArray;
    graph[1] = edgeArray;
    graph[2] = weightArray;
    d_worklist = worklist;
    d_inputPrefixSum = inputPrefixSum;
    d_prefixSum = prefixSum;
    d_worklist[0] = 1;
    d_worklist[1] = 3;
    d_worklist[2] = 4;
    d_worklist[3] = 5;
    d_worklist[4] = 6;
    d_worklist[5] = 9;
    d_worklist[6] = 8;
    d_worklist[7] = 7;
    d_worklist[8] = 10;
    d_worklist[9] = 11;
    d_worklistLength = 10;
    d_inputPrefixSum[numVertices + 1] = 0;
    d_inputPrefixSum[numVertices + 2] = 0;
    d_prefixSum[numVertices + 1] = 0;
    d_prefixSum[numVertices + 2] = 0;
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
    
    int vertex, numNeighbours, tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < d_worklistLength) {
        vertex = d_worklist[tId];
        numNeighbours = graph[0][vertex + 1] - graph[0][vertex];
        d_inputPrefixSum[tId] = numNeighbours;
        // PrefixScan()
        int index = tId, add = 1;
        while (2 * index + add < d_worklistLength) {
            d_inputPrefixSum[2 * index] += d_inputPrefixSum[2 * index + add];
            index = index << 1;
            add = add << 1;
        }
        if (tId == 0) {
            d_prefixLevel = add;
        }
        __syncthreads();
        int level;
        level = d_prefixLevel;
        index = tId * d_prefixLevel;
        print("Cuda: Thread = %d, index = %d, add = %d, level = %d\n", tId, index, add, level);
        while (level != 0) {
            if (index < d_worklistLength) {
                d_inputPrefixSum[index] -= d_inputPrefixSum[index + level / 2];
                d_prefixSum[index + level / 2] = d_inputPrefixSum[index] + d_prefixSum[index];
            }
            __syncthreads();
            index = index >> 1;
            level = level >> 1;
        }
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
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        Cuda_SSSP<<<2, 5>>>();
        CUDA_ERR_CHECK;
        cudaThreadSynchronize();
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
    }

    int *inputPrefixSum;
    cudaMemcpyFromSymbol(&inputPrefixSum, d_inputPrefixSum, sizeof(int *), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(distance, inputPrefixSum, (numVertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
/*    out << "Distances:\n";
    for (int i = 0; i <= numVertices + 1; i++)
        out << "["<< i << "] = " << distance[i] << endl;
*/
    out << "Prefix Sums\n";
    cudaMemcpyFromSymbol(&inputPrefixSum, d_prefixSum, sizeof(int *), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(distance, inputPrefixSum, (numVertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    for (int i = 0; i <= numVertices + 1; i++)
        out << "["<< i << "] = " << distance[i] << endl;
    return 0;
}
int CudaGraphClass::copyGraphToDevice() {

    int *vertexArray, *edgeArray, *weightArray, *worklist;
    int *inputPrefixSum, *prefixSum;
    cudaError_t err;
    err = cudaMalloc((void **)&vertexArray, (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&edgeArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&weightArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&worklist, (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&inputPrefixSum, (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&prefixSum, (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(inputPrefixSum, 0x0, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(prefixSum, 0x0, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(vertexArray, row[0], (numVertices + 2) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray, row[1], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(weightArray, row[2], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    CudaInitialize<<<1, 1>>>(vertexArray, edgeArray, weightArray, worklist, inputPrefixSum, prefixSum, numVertices, numEdges);
    cudaThreadSynchronize();
    return 0;
}

void CudaGraphClass::populate(char *fileName) {

    inputFile.open(fileName);
    if (!inputFile.is_open()){
        cout << "invalid file";
        return;
    }

    int **AdjMatrix, i, j, k;
    AdjMatrix = new int* [numVertices + 1]();
    for (i = 0; i <= numVertices; i++) {

        AdjMatrix[i] = new int [numVertices + 1]();
    }
    i = numEdges;
    int lastj = 0, currentIndex = 0;
    inputFile >> j >> k;
    srand(time(NULL));
    while(i > 0) {

        //scanf("%d %d", &j, &k);
        inputFile >> j >> k;
        AdjMatrix[j][k] = 1;
        while (lastj <= j || lastj == 0) {
            if (lastj == 0) {
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
    //row[1][0] = 0;
    // Sentinel node just points to the end of the last node in the graph
    while (lastj <= numVertices + 1) {
        row[0][lastj] = currentIndex;
        lastj++;
    }
    //row[0][lastj+1] = currentIndex;
/*    for (i = 0; i <= numVertices + 1; i++)
        print("Vertex: %d = %d\n", i, row[0][i]);

    print("Second Array:\n");
    for (i = 0; i <= numEdges; i++)
        print("Edges: Index: %d, Value = %d\n", i, row[1][i]);
*/
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
