#include <iostream>
#include "CudaGraph.h"
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>

#define cprint(...) printMutex.lock(); printf(__VA_ARGS__); printMutex.unlock();
#define p(x) printMutex.lock(); cout << x << endl; printMutex.unlock();

#define CUDA_ERR_CHECK  \
if( err != cudaSuccess) { \
    printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__); \
    return EXIT_FAILURE; \
}

#define CUDA_SET_DEVICE_ID \
cudaSetDevice(0);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

using namespace std;

__device__ int *graph[3], d_numVertices, d_numEdges, *d_worklist, *d_gatherWorklist, d_worklistLength;
__device__ int *d_distance, *d_parentInWorklist;
__device__ int *d_prefixSum, *d_blockPrefixSum;
__device__ int *d_prefixLevel;
__device__ bool d_terminate;

__global__ void CudaInitialize(int *vertexArray, int *edgeArray, int *weightArray, int *distance, int *worklist, int *worklist2, int *parentInWorklist, int *prefixSum, int *blockPrefixSum, int *prefixLevel, int numVertices, int numEdges) {

    d_numVertices = numVertices;
    d_numEdges = numEdges;
    graph[0] = vertexArray;
    graph[1] = edgeArray;
    graph[2] = weightArray;
    d_distance = distance;
    d_distance[1] = 0;
    d_worklist = worklist;
    d_gatherWorklist = worklist2;
    d_parentInWorklist = parentInWorklist;
    d_prefixSum = prefixSum;
    d_blockPrefixSum = blockPrefixSum;
    d_prefixLevel = prefixLevel;
    d_worklist[0] = 1;
    /*for (int i = 0; i < numVertices; i++) {
        d_worklist[i] = i;
        //d_distance[i] = 100;
    }*/
    d_worklistLength = 1;
    print("WLLenght = %d, numVertices = %d\n", d_worklistLength, d_numVertices);
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


// Prefix Sum calculation within a single block
__global__ void Cuda_IntraBlockPrefixSum() {
    
    extern __shared__ int temp[];
    __shared__ int blockPrefixSum;
    int vertex, maxLength, numNeighbours, tId = threadIdx.x;
    if ((d_worklistLength - blockIdx.x * 1024) < 1024) {
        maxLength = d_worklistLength - blockIdx.x * 1024 + 1;
        // To mark the boundary, maxLength is increased by 1 and the
        // numNeighbours of the last element is set to zero.
        temp[maxLength - 1] = 0;
    } else
        maxLength = 1024;
    if (blockIdx.x * blockDim.x + tId < d_worklistLength) {
        vertex = d_worklist[blockIdx.x * blockDim.x + tId];
        numNeighbours = graph[0][vertex + 1] - graph[0][vertex];
        temp[tId] = numNeighbours;
    }
    //print("This is a thread : %d. Max = %d\n", threadIdx.x, maxLength);
    __syncthreads();
    int index =  2 * tId, add = 1;
    for (int depth = maxLength; depth > 0; depth = depth >> 1) {
        if (index + add < maxLength) {
            temp[index] += temp[index + add];
            index = index << 1;
            add = add << 1;
        }
        __syncthreads();
    }
    if (tId == 0) {
        d_prefixLevel[blockIdx.x] = add;
        blockPrefixSum = temp[0];
        //print("Level = %d. MaxLength = %d\n", d_prefixLevel[blockIdx.x], maxLength);
    }
    
    /*if (tId < d_worklistLength)
        d_prefixSum[tId] = temp[tId];*/

    __syncthreads();
    int level;
    level = d_prefixLevel[blockIdx.x];
    index = tId * level;
    for (int depth = maxLength; depth > 0; depth = depth >> 1) {
        if (index + level / 2 < maxLength) {
            temp[index] -= temp[index + level / 2];
            d_prefixSum[blockIdx.x * blockDim.x + index + level / 2] = temp[index] + d_prefixSum[blockIdx.x * blockDim.x + index];
        }
        index = index >> 1;
        level = level >> 1;
        __syncthreads();
    }
    if (tId == 0) {
        d_prefixLevel[blockIdx.x] = blockPrefixSum;
        //print("Block %d. PrefixSum = %d, Array Value = %d\n", blockIdx.x, blockPrefixSum, d_prefixLevel[blockIdx.x]);
    }
}

// Prefix Sum on the whole block sum.
__global__ void Cuda_BlockOffsetPrefixSum(int numBlocks) {
    
    extern __shared__ int temp[];
    int tId = threadIdx.x;

    if (tId < numBlocks)
        temp[tId] = d_prefixLevel[tId];

    int index =  2 * tId, add = 1;
    __shared__ int sharedVar;
    for (int depth = numBlocks; depth > 0; depth = depth >> 1) {
        if (index + add < numBlocks) {
            temp[index] += temp[index + add];
            index = index << 1;
            add = add << 1;
        }
        __syncthreads();
    }
    if (tId == 0) {
        sharedVar = add;
        d_worklistLength = temp[0];
        print("New WorkList Length = %d\n", d_worklistLength);
    }
    __syncthreads();
    
    int level;
    level = sharedVar;
    index = tId * level;
    for (int depth = numBlocks; depth > 0; depth = depth >> 1) {
        if (index + level / 2 < numBlocks) {
            temp[index] -= temp[index + level / 2];
            d_blockPrefixSum[blockIdx.x * blockDim.x + index + level / 2] = temp[index] + d_blockPrefixSum[blockIdx.x * blockDim.x + index];
        }
        index = index >> 1;
        level = level >> 1;
        __syncthreads();
    }
}

__global__ void Cuda_AddBlockPrefix() {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < d_worklistLength) {
        d_prefixSum[tId] += d_blockPrefixSum[blockIdx.x];
    }
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
    delete[] verifiedPrefix;
    return 0;
}

int CudaGraphClass::verifyGatherWorklist(int *calculatedGatherWorklist, int newWorklistLength) {

    int vertex = 0, i = 0;
    while (i < newWorklistLength) {
        for (int j = row[0][vertex]; j < row[0][vertex + 1]; j++, i++) {
            if (row[1][j] != calculatedGatherWorklist[i]) {
                print("Verify Gather Worklist: Verification Failed at vertex %d\n", vertex);
                return 1;
            }
        }
        vertex++;
    }
    cout << "Gather Worklist Verified: " << i << "\n";
    return 0;
}


inline int reallocDeviceMemory(int *d_pointer, int newMemorySize) {

    int *devicePointer;
    cudaError_t err;

    err = cudaMemcpyFromSymbol(&devicePointer, d_pointer, sizeof(int *), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    err = cudaFree(devicePointer);
    CUDA_ERR_CHECK;
    
    err = cudaMalloc((void **)&devicePointer, newMemorySize * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpyToSymbol(d_pointer, &devicePointer, sizeof(int *), 0, cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;

    return 0;
}

int CudaGraphClass::PrefixSum(int worklistLength, int *newWorklistLength) {

    cudaError_t err;
    if (maxWorklistLength < worklistLength) {
        cout << "PrefixSum Realloc\n";
        int *devicePrefixSum, *newPrefixSum;
        maxWorklistLength = worklistLength;
        
        err = cudaMemcpyFromSymbol(&devicePrefixSum, d_prefixSum, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaFree(devicePrefixSum);
        CUDA_ERR_CHECK;
        
        err = cudaMalloc((void **)&newPrefixSum, maxWorklistLength * sizeof(int));
        CUDA_ERR_CHECK;
        err = cudaMemcpyToSymbol(d_prefixSum, &newPrefixSum, sizeof(int *), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;

        //reallocDeviceMemory(d_worklist, maxWorklistLength);
        /*int *deviceWorklist, *newWorklist;
        err = cudaMemcpyFromSymbol(&deviceWorklist, d_worklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaFree(deviceWorklist);
        CUDA_ERR_CHECK;
        
        err = cudaMalloc((void **)&newWorklist, maxWorklistLength * sizeof(int));
        CUDA_ERR_CHECK;
        err = cudaMemcpyToSymbol(d_worklist, &newWorklist, sizeof(int *), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;*/
    }
    int numBlocksPerGrid = (worklistLength + numThreadsPerBlock) / numThreadsPerBlock;
    if (maxNumBlocksPerGrid < numBlocksPerGrid) {
        maxNumBlocksPerGrid = numBlocksPerGrid;

        //reallocDeviceMemory(d_prefixLevel, maxNumBlocksPerGrid);
        int *devicePrefixLevel, *newPrefixLevel;
        err = cudaMemcpyFromSymbol(&devicePrefixLevel, d_prefixLevel, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaFree(devicePrefixLevel);
        CUDA_ERR_CHECK;
        
        err = cudaMalloc((void **)&newPrefixLevel, maxNumBlocksPerGrid * sizeof(int));
        CUDA_ERR_CHECK;
        err = cudaMemcpyToSymbol(d_prefixLevel, &newPrefixLevel, sizeof(int *), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;

        //reallocDeviceMemory(d_blockPrefixSum, maxNumBlocksPerGrid);
        int *deviceBlockPrefixSum, *newBlockPrefixSum;
        err = cudaMemcpyFromSymbol(&deviceBlockPrefixSum, d_blockPrefixSum, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaFree(deviceBlockPrefixSum);
        CUDA_ERR_CHECK;
        
        err = cudaMalloc((void **)&newBlockPrefixSum, maxNumBlocksPerGrid * sizeof(int));
        CUDA_ERR_CHECK;
        err = cudaMemcpyToSymbol(d_blockPrefixSum, &newBlockPrefixSum, sizeof(int *), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
    }

    Cuda_IntraBlockPrefixSum<<<numBlocksPerGrid, numThreadsPerBlock, numThreadsPerBlock * sizeof(int)>>>();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    Cuda_BlockOffsetPrefixSum<<<(numBlocksPerGrid + numThreadsPerBlock) / numThreadsPerBlock, numThreadsPerBlock, numBlocksPerGrid * sizeof(int )>>>(numBlocksPerGrid);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    Cuda_AddBlockPrefix<<<numBlocksPerGrid, numThreadsPerBlock>>>();
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int *devicePrefixSum, *hostPrefixSum;

    hostPrefixSum = new int[(worklistLength + 1)];
    err = cudaMemcpyFromSymbol(&devicePrefixSum, d_prefixSum, sizeof(int *), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(hostPrefixSum, devicePrefixSum, (worklistLength + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    out << "Prefix Sums\n";
    //for (int i = 0; i <= worklistLength; i++)
    //    out << "["<< i << "] = " << hostPrefixSum[i] << endl;

    //verifyPrefixSum(hostPrefixSum);
    err = cudaMemcpyFromSymbol(&worklistLength, d_worklistLength, sizeof(int), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    *newWorklistLength = worklistLength;
    delete[] hostPrefixSum;
    return 0;
}

// TODO: Build an optimized fine grained gathering algorithm
__global__ void populateNeighbours(int worklistLength) {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId < worklistLength) {
        int vertex = d_worklist[tId];
        printf("Thread %d: vertex = %d\n", tId, d_worklist[tId]);
        int edgeIndex = graph[0][vertex];
        int index = d_prefixSum[tId], lastIndex = d_prefixSum[tId + 1];
        //print("Thread: %d: vertex = %d, edgeIndex = %d, prefix = %d, lastIndex = %d\n", tId, vertex, edgeIndex, index, lastIndex);
        for (int i = 0; i < lastIndex - index; i++) {
            d_gatherWorklist[index + i] = graph[1][edgeIndex + i];
            //d_parentInWorklist[index + i] = vertex;
            d_parentInWorklist[index + i] = d_distance[vertex] + graph[2][edgeIndex + i];
        }
    }
}

int CudaGraphClass::gatherNeighbours(int worklistLength) {

    int numBlocksPerGrid = (worklistLength + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    cout << "Gather Neighbours: " << numThreadsPerBlock << ", " << numBlocksPerGrid << ", " << worklistLength << "\n";
    populateNeighbours<<<numBlocksPerGrid, numThreadsPerBlock>>>(worklistLength);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}

__global__ void removeDuplicatesInGather(int worklistLength) {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    if (tId != 0)
        return;

    int prevVertex = d_gatherWorklist[0], index = 1;
    for (int i = 1; i < worklistLength; i++) {
        if (prevVertex != d_gatherWorklist[i]) {
            d_gatherWorklist[index] = d_gatherWorklist[i];
            prevVertex = d_gatherWorklist[i];
            index++;
        }
    }
    d_worklistLength = index;
}

__global__ void processEdges(int worklistLength) {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ bool terminate;
    terminate = true;
    
    if (tId >= worklistLength)
        return;

    if (tId == 0 || d_gatherWorklist[tId] != d_gatherWorklist[tId - 1]) {
        int vertex = d_gatherWorklist[tId];
        int min = d_distance[vertex], i = 0;
        while((tId + i) < worklistLength && d_gatherWorklist[tId + i] == vertex) {
            if (min > d_parentInWorklist[tId + i]) {
                min = d_parentInWorklist[tId + i];
                terminate = false;
            }
            i++;
        }
        d_distance[vertex] = min;
    }

    if (terminate == false) {
        //printf("ThreadId : %d . Terminate = %d, d_terminate = %d\n", threadIdx.x, terminate, d_terminate);
        d_terminate = false;
    }
}

int CudaGraphClass::processNeighbours(int newWorklistLength) {

    int *deviceGatherWorklist, *deviceParentInWorklist;
    cudaError_t err;

    err = cudaMemcpyFromSymbol(&deviceGatherWorklist, d_gatherWorklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    err = cudaMemcpyFromSymbol(&deviceParentInWorklist, d_parentInWorklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;

    thrust::device_ptr<int> dev_gatherWorklist(deviceGatherWorklist);
    thrust::device_ptr<int> dev_parentInWorklist(deviceParentInWorklist);
    // wrap raw pointer with a device_ptr 
    //thrust::device_ptr<int> dev_gatherWorklist = thrust::device_pointer_cast(deviceGatherWorklist);

    thrust::sort_by_key(dev_gatherWorklist, dev_gatherWorklist + newWorklistLength, dev_parentInWorklist);
    int numBlocksPerGrid = (newWorklistLength + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    processEdges<<<numBlocksPerGrid, numThreadsPerBlock>>>(newWorklistLength);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    removeDuplicatesInGather<<<numBlocksPerGrid, numThreadsPerBlock>>>(newWorklistLength);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return 0;
}

int CudaGraphClass::callSSSP() {

    int terminate = false, *distance, *parentInWorklist, *gatherWorklist, worklistLength = 1;
    bool worklistReallocated;
    cudaError_t err;

    distance = new int[(numVertices + 1)];
    parentInWorklist = new int[(numEdges+ 1)];
    gatherWorklist = new int[(numEdges + 1)];
    //int numBlocksPerGrid = (numVertices + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    //cout << numThreadsPerBlock << ", " << numBlocksPerGrid << "\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (terminate == false) {
        terminate = true;
        worklistReallocated = false;
        err = cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        gpuErrchk(cudaDeviceSynchronize());

        int newWorklistLength;
        PrefixSum(worklistLength, &newWorklistLength);

        /*if (maxWorklistLength < newWorklistLength) {
            maxWorklistLength = newWorklistLength;
            worklistReallocated = true;
            
            //reallocDeviceMemory(d_worklist, maxWorklistLength);
            cout << "After Prefix Sum: Realloc\n";
            int *deviceGatherWorklist, *newGatherWorklist;
            int *deviceParentInWorklist, *newParentInWorklist;

            err = cudaMemcpyFromSymbol(&deviceGatherWorklist, d_gatherWorklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            cout << "Gather worklist = " << deviceGatherWorklist << "\n";
            err = cudaFree(deviceGatherWorklist);
            CUDA_ERR_CHECK;
            
            err = cudaMalloc((void **)&newGatherWorklist, maxWorklistLength * sizeof(int));
            CUDA_ERR_CHECK;
            err = cudaMemcpyToSymbol(d_gatherWorklist, &newGatherWorklist, sizeof(int *), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;

            err = cudaMemcpyFromSymbol(&deviceParentInWorklist, d_parentInWorklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            err = cudaFree(deviceParentInWorklist);
            CUDA_ERR_CHECK;
            
            err = cudaMalloc((void **)&newParentInWorklist, maxWorklistLength * sizeof(int));
            CUDA_ERR_CHECK;
            err = cudaMemcpyToSymbol(d_parentInWorklist, &newParentInWorklist, sizeof(int *), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;
            delete[] parentInWorklist;
            delete[] gatherWorklist;
            parentInWorklist = new int[maxWorklistLength];
            gatherWorklist = new int[maxWorklistLength];
            cout << "Size of gather and parent = " << maxWorklistLength << "\n";
        }*/
        cout << "New WorkList in Host = " << newWorklistLength << "\n";
        gatherNeighbours(worklistLength);
        int * tempdeviceGatherWorklist;
        err = cudaMemcpyFromSymbol(&tempdeviceGatherWorklist, d_gatherWorklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaMemcpy(gatherWorklist, tempdeviceGatherWorklist, newWorklistLength * sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        cout << "Before removing duplicates: \n";
        for (int i = 0; i < newWorklistLength; i++)
            out << "["<< i << "] = " << gatherWorklist[i] << " -- " << parentInWorklist[i] << " -> " << distance[gatherWorklist[i]] << endl;

        processNeighbours(newWorklistLength);

        int *deviceWorklist, *deviceGatherWorklist, *deviceParentInWorklist, *deviceDistance;
        err = cudaMemcpyFromSymbol(&deviceGatherWorklist, d_gatherWorklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaMemcpyFromSymbol(&deviceWorklist, d_worklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaMemcpyFromSymbol(&deviceParentInWorklist, d_parentInWorklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaMemcpyFromSymbol(&deviceDistance, d_distance, sizeof(int *), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;

        err = cudaMemcpy(gatherWorklist, deviceGatherWorklist, newWorklistLength * sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaMemcpy(parentInWorklist, deviceParentInWorklist, newWorklistLength * sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        err = cudaMemcpy(distance, deviceDistance, numVertices * sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        cout << "New Worklist: \n";
        //for (int i = 0; i < newWorklistLength; i++)
        //    out << "["<< i << "] = " << gatherWorklist[i] << " -- " << parentInWorklist[i] << " -> " << distance[gatherWorklist[i]] << endl;

        //verifyGatherWorklist(gatherWorklist, newWorklistLength);

        err = cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        cout << "Terminate: " << terminate << "\n";

        //worklistLength = newWorklistLength;
        err = cudaMemcpyFromSymbol(&worklistLength, d_worklistLength, sizeof(int), 0, cudaMemcpyDeviceToHost);
        CUDA_ERR_CHECK;
        cout << "After removing duplicated: " << worklistLength << "\n";
        //cout << "New Worklist: \n";
        for (int i = 0; i < worklistLength; i++)
            out << "["<< i << "] = " << gatherWorklist[i] << " -> " << distance[gatherWorklist[i]] << endl;

        /*if (worklistReallocated == true) {
            int *deviceWorklist, *newWorklist;
            
            err = cudaMemcpyFromSymbol(&deviceWorklist, d_worklist, sizeof(int *), 0, cudaMemcpyDeviceToHost);
            CUDA_ERR_CHECK;
            
            err = cudaFree(deviceWorklist);
            CUDA_ERR_CHECK;
            
            err = cudaMalloc((void **)&newWorklist, maxWorklistLength * sizeof(int));
            CUDA_ERR_CHECK;
            err = cudaMemcpyToSymbol(d_worklist, &newWorklist, sizeof(int *), 0, cudaMemcpyHostToDevice);
            CUDA_ERR_CHECK;

        }*/

        // Swap worklist and gatherWorklist
        err = cudaMemcpyToSymbol(d_worklist, &deviceGatherWorklist, sizeof(int *), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
        err = cudaMemcpyToSymbol(d_gatherWorklist, &deviceWorklist, sizeof(int *), 0, cudaMemcpyHostToDevice);
        CUDA_ERR_CHECK;
    }

    /*cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Elapsed time  = " << milliseconds << "\n";*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Elapsed time  = " << elapsedTime << " milliseconds \n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "Shortest Distances: \n";
    for (int i = 0; i <= numVertices; i++)
        out << "["<< i << "] = " << distance[i] << endl;
    
    delete[] distance;
    delete[] parentInWorklist;
    delete[] gatherWorklist;
    return 0;
}


int CudaGraphClass::copyGraphToDevice() {

    CUDA_SET_DEVICE_ID;
    gpuErrchk(cudaPeekAtLastError());
    int *vertexArray, *edgeArray, *weightArray, *distance, *parent, *worklist, *worklist2;
    int *prefixSum, *blockPrefixSum, *prefixLevel;
    cudaError_t err;
    err = cudaMalloc((void **)&vertexArray, (numVertices + 2) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&edgeArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&weightArray, (numEdges + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&distance, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    maxWorklistLength = numVertices + 2;
    err = cudaMalloc((void **)&worklist, numEdges/*maxWorklistLength*/ * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&worklist2, numEdges/*maxWorklistLength*/ * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&parent, numEdges/*maxWorklistLength*/ * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&prefixSum, maxWorklistLength * sizeof(int));
    CUDA_ERR_CHECK;
    maxNumBlocksPerGrid = 1024;
    err = cudaMalloc((void **)&blockPrefixSum, maxNumBlocksPerGrid * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMalloc((void **)&prefixLevel, maxNumBlocksPerGrid * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(prefixSum, 0x0, maxWorklistLength * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemset(blockPrefixSum, 0x0, maxNumBlocksPerGrid * sizeof(int));
    CUDA_ERR_CHECK;
    err = cudaMemcpy(vertexArray, row[0], (numVertices + 2) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(edgeArray, row[1], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(weightArray, row[2], (numEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemset(distance, 0x7f, (numVertices + 1) * sizeof(int));
    CUDA_ERR_CHECK;
    CudaInitialize<<<1, 1>>>(vertexArray, edgeArray, weightArray, distance, worklist, worklist2, parent, prefixSum, blockPrefixSum, prefixLevel, numVertices, numEdges);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
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
        row[2][currentIndex] = (rand() % 2) ? rand() % 10 + 1/* - 10 */: rand() % 10 + 1;
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
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
