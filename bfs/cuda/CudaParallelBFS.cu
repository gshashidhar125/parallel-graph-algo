#include <iostream>
#include "CudaGraph.h"
#define cprint(...) printMutex.lock(); printf(__VA_ARGS__); printMutex.unlock();
#define p(x) printMutex.lock(); cout << x << endl; printMutex.unlock();

using namespace std;

class CudaParallelBFS{

public:
    CudaGraphClass *graphData;
    CudaParallelBFS() {}
    void setGraph(CudaGraphClass *a) {
        graphData = a;
    }
};
