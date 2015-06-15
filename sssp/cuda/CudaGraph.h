//#ifndef CudaGraph_H
//#define CudaGraph_H
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG

#ifdef DEBUG
#define print(...) printf(__VA_ARGS__)
#else
#define print(...) ;
#endif

#define outs(...) print(__VA_ARGS__)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher
#endif

using namespace std;

class CudaGraphClass {

public:
//private:
    int numVertices, numEdges;
    int *row[3];
    ifstream inputFile;

    CudaGraphClass() {
        numVertices = 0;
        numEdges = 0;
    }
    CudaGraphClass(int vertices, int edges) {
        numVertices = vertices;
        numEdges = edges;

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
    void callSSSP();
    int copyGraphToDevice();
};
//#endif
