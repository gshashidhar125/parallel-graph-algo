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
__device__ int *d_inputPrefixSum, *d_prefixSum, *d_blockPrefixSum;
__device__ int *d_prefixLevel;
__device__ bool d_terminate;

__global__ void CudaInitialize(int *vertexArray, int *edgeArray, int *weightArray, int *worklist, int *inputPrefixSum, int *prefixSum, int *blockPrefixSum, int numVertices, int numEdges) {

    d_numVertices = numVertices;
    d_numEdges = numEdges;
    graph[0] = vertexArray;
    graph[1] = edgeArray;
    graph[2] = weightArray;
    d_worklist = worklist;
    d_inputPrefixSum = inputPrefixSum;
    d_prefixSum = prefixSum;
    d_blockPrefixSum = blockPrefixSum;
    for (int i = 0; i < numVertices; i++) {
        d_worklist[i] = i;
    }
    d_worklistLength = d_numVertices;
    print("WLLenght = %d, numVertices = %d\n", d_worklistLength, d_numVertices);
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
    
    extern __shared__ int temp[];
    __shared__ int blockPrefixSum;
    int vertex, maxLength, numNeighbours, tId = threadIdx.x;
    if ((d_worklistLength - blockIdx.x * 1024) < 1024)
        maxLength = d_worklistLength - blockIdx.x * 1024;
    else
        maxLength = 1024;
    if (blockIdx.x * blockDim.x + tId < d_worklistLength) {
        vertex = d_worklist[blockIdx.x * blockDim.x + tId];
        numNeighbours = graph[0][vertex + 1] - graph[0][vertex];
        temp[tId] = numNeighbours;
    }
    //print("This is a thread : %d. Max = %d\n", threadIdx.x, maxLength);
    __syncthreads();
    // PrefixScan()
    int index =  2 * tId, add = 1;
    for (int depth = maxLength >> 1; depth > 0; depth = depth >> 1) {
        //while ((2 * index + add < maxLength) && (add <= 2)) {
        if (index + add < maxLength) {
            temp[index] += temp[index + add];
            index = index << 1;
            add = add << 1;
        }
        print("Level: %d Before\n", depth);
        __syncthreads();
        print("Level: %d After\n", depth);
    }
        /*if (tId == 0) {
            d_prefixLevel[blockIdx.x] = add;
            blockPrefixSum = temp[0];
        }*/
        //d_inputPrefixSum[blockIdx.x * blockDim.x + tId] = temp[tId];
    
    __syncthreads();
    if (tId < d_worklistLength)
        d_prefixSum[tId] = temp[tId];
/*    if (blockIdx.x * blockDim.x + tId < d_worklistLength) {
        int level, index;
        level = d_prefixLevel[blockIdx.x];
        index = tId * level;
        //print("Cuda: Thread = %d, index = %d, level = %d\n", tId, index, level);
        while (level != 0) {
            if (index + level / 2 < maxLength) {
                temp[index] -= temp[index + level / 2];
                d_prefixSum[blockIdx.x * blockDim.x + index + level / 2] = temp[index] + d_prefixSum[blockIdx.x * blockDim.x + index];
            }
            __syncthreads();
            index = index >> 1;
            level = level >> 1;
        }
        if (tId == 0) {
            d_prefixLevel[blockIdx.x] = blockPrefixSum;
            print("Block %d. PrefixSum = %d, Array Value = %d\n", blockIdx.x, blockPrefixSum, d_prefixLevel[blockIdx.x]);
        }
    }*/
}

__global__ void Cuda_BlockPrefixSum(int numBlocks) {
    
    int tId = threadIdx.x;
    extern __shared__ int temp[];
    if (tId < numBlocks) {
        int index = tId, add = 1;
        temp[tId] = d_prefixLevel[tId];
        //print("BEFORE Temp BlockPrefixSum[%d] = %d\n", tId, temp[tId]);
        while (2 * index + add < numBlocks) {
            temp[2 * index] += temp[2 * index + add];
            index = index << 1;
            add = add << 1;
        }
        if (tId == 0) {
            d_prefixLevel[blockIdx.x] = add;
        }
    }
    __syncthreads();
    if (tId < numBlocks) {
        //print("AFTER Temp BlockPrefixSum[%d] = %d\n", tId, temp[tId]);
        int level, index;
        level = d_prefixLevel[blockIdx.x];
        index = tId * level;
        while (level != 0) {
        //print("BLOCK PREFIX Cuda: Thread = %d, index = %d, level = %d\n", tId, index, level);
            if (index + level / 2 < numBlocks) {
                temp[index] -= temp[index + level / 2];
                d_blockPrefixSum[blockIdx.x * blockDim.x + index + level /2] = temp[index] + d_blockPrefixSum[blockIdx.x * blockDim.x + index];
            }
            __syncthreads();
            index = index >> 1;
            level = level >> 1;
        }
        //print("BlockPrefixSum[%d] = %d\n", tId, d_blockPrefixSum[tId]);
    }
}

__global__ void Cuda_AddBlockPrefix() {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < d_worklistLength) {
        d_prefixSum[tId] += d_blockPrefixSum[blockIdx.x];
    }
}
/*
__global__ void Cuda_PrefixSum() {
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < d_worklistLength) {
        int level, index;
        level = d_prefixLevel;
        index = tId * d_prefixLevel;
        print("Cuda: Thread = %d, index = %d, level = %d\n", tId, index, level);
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
}*/

int CudaGraphClass::callSSSP() {

    int terminate = false, *distance, *d_distance, *prefixLevel;
    cudaError_t err;

    distance = new int[(numVertices + 1)];
    err = cudaMalloc((void **)&d_distance, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(d_distance, 0xff, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    int numThreadsPerBlock = 1024;
    int numBlocksPerGrid = (numVertices + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    cout << numThreadsPerBlock << ", " << numBlocksPerGrid << "\n";

    err = cudaMalloc((void **)&prefixLevel, numBlocksPerGrid * sizeof(int));
    CUDA_ERR_CHECK;
    cudaMemcpyToSymbol(d_prefixLevel, &prefixLevel, sizeof(int *), 0, cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;

    while (terminate == false) {
        terminate = true;
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
       CUDA_ERR_CHECK;
       cudaError_t cudaerr = cudaDeviceSynchronize();
       if (cudaerr != cudaSuccess)
           printf("kernel launch failed with error \"%s\".\n",
                  cudaGetErrorString(cudaerr));
        print("____________________");
        Cuda_SSSP<<<numBlocksPerGrid, numThreadsPerBlock, numThreadsPerBlock * sizeof(int)>>>();
        CUDA_ERR_CHECK;
        cudaPeekAtLastError();
        CUDA_ERR_CHECK;
        //cudaThreadSynchronize();
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
        print("#####################");
/*        Cuda_BlockPrefixSum<<<(numBlocksPerGrid + numThreadsPerBlock) / numThreadsPerBlock, numThreadsPerBlock, numBlocksPerGrid * sizeof(int )>>>(numBlocksPerGrid);
        CUDA_ERR_CHECK;
        cudaPeekAtLastError();
        CUDA_ERR_CHECK;
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess)
            printf("kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
*/        //cudaThreadSynchronize();
/*        Cuda_AddBlockPrefix<<<numBlocksPerGrid, numThreadsPerBlock>>>();
        CUDA_ERR_CHECK;
        cudaThreadSynchronize();*/
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
    }

    int *inputPrefixSum;
/*    cudaMemcpyFromSymbol(&inputPrefixSum, d_inputPrefixSum, sizeof(int *), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(distance, inputPrefixSum, (numVertices + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    out << "Distances:\n";
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

    verifyPrefixSum(distance);
    return 0;
}

int CudaGraphClass::verifyPrefixSum(int *calculatedPrefix) {
    
    int *verifiedPrefix, prefix = 0;
    verifiedPrefix = new int[(numVertices + 1)];
    for (int vertex = 0; vertex < numVertices; vertex++) {
        verifiedPrefix[vertex] = prefix;
        int numNeighbours = row[0][vertex + 1] - row[0][vertex];
        prefix += numNeighbours;
    }
    /*for (int vertex = 0; vertex <= numVertices; vertex++) {
        print("Prefix[%d] = %d\n", vertex, verifiedPrefix[vertex]);
    }*/
    for (int vertex = 0; vertex < numVertices; vertex++) {
        if (verifiedPrefix[vertex] != calculatedPrefix[vertex]) {
            print("Verification failed at vertex %d.\n", vertex);
            print("Verified prefix = %d. Calculated prefix = %d\n", verifiedPrefix[vertex], calculatedPrefix[vertex]);
            return 1;
        }
    }
    return 0;
}

int CudaGraphClass::copyGraphToDevice() {

    int *vertexArray, *edgeArray, *weightArray, *worklist;
    int *inputPrefixSum, *prefixSum, *blockPrefixSum;
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
    int numThreadsPerBlock = 1024;
    int numBlocksPerGrid = (numVertices + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    err = cudaMalloc((void **)&blockPrefixSum, numBlocksPerGrid * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(inputPrefixSum, 0x0, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(prefixSum, 0x0, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(blockPrefixSum, 0x0, numBlocksPerGrid * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(vertexArray, row[0], (numVertices + 2) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray, row[1], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(weightArray, row[2], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    CudaInitialize<<<1, 1>>>(vertexArray, edgeArray, weightArray, worklist, inputPrefixSum, prefixSum, blockPrefixSum, numVertices, numEdges);
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
                outs("\n\nGraph Do not Match at [%d][%d]. CurrentIndex = %d\n\n", i, row[1][currentIndex], currentIndex);
                break;
            }
            j++;
            currentIndex ++;
        }
    }
    for (i = 0; i <= numVertices; i++) {

        delete[] AdjMatrix[i];
    }
    delete[] AdjMatrix;
}
void CudaGraphClass::printGraph() {
    CudaPrintGraph<<<1, 1>>>();
    cudaThreadSynchronize();
}
