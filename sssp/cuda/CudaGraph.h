//#ifndef CudaGraph_H
//#define CudaGraph_H
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#include <thrust/sort.h>

#define DEBUG

#ifdef DEBUG
#define print(...) printf(__VA_ARGS__)
#define dump cout
#else
#define print(...) ;
#define dump null_stream
#define Cuda_dump Cuda_null_stream
#endif

#define outs(...) printf(__VA_ARGS__)
#define out cout

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

using namespace std;

class NullBuffer : public std::streambuf
{
public:
  int overflow(int c) { return c; }
};

class CudaGraphClass {

public:
//private:
    int numVertices, numEdges;
    int numThreadsPerBlock; 
    int maxWorklistLength, maxNumBlocksPerGrid;
    int *row[3];
    ifstream inputFile;

    CudaGraphClass() {
        numVertices = 0;
        numEdges = 0;
    }
    CudaGraphClass(int vertices, int edges, int threadsPerBlock) {
        numVertices = vertices;
        numEdges = edges;
        numThreadsPerBlock = threadsPerBlock;
        //numBlocksPerGrid = (numVertices + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
        maxWorklistLength = numVertices;
        maxNumBlocksPerGrid = (numVertices + numThreadsPerBlock) / numThreadsPerBlock;

#ifdef CUDA_CSR
// NumVertices starting from 1 to NumVertices plus an addition Sentinel node
// which points to the last index of the row[1] array.
        row[0] = new int [numVertices + 2]();
        row[1] = new int [numEdges + 1]();
        row[2] = new int [numEdges + 1]();
#elif CUDA_EDGE_LIST
        row[0] = new int [numEdges + 1]();
        row[1] = new int [numEdges + 1]();
        row[2] = new int [numEdges + 1]();
#endif
    }

    void populate(char *fileName);
    void printGraph();
    int callSSSP();
    int copyGraphToDevice();
    int PrefixSum(int worklistLength, int *);
    int gatherNeighbours(int);
    int verifyPrefixSum(int *);
    int verifyGatherWorklist(int *, int);
    int processNeighbours(int worklistLength);
//    int reallocDeviceMemory(int *d_pointer, int newMemorySize);
};
//#endif
