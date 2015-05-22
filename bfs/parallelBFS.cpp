#include <iostream>
#include <mutex>
#include <vector>
#include <pthread.h>
#include "Graph.h"
#include "Thread.h"

#define numThreads 4
#define print(...) printMutex.lock(); printf(__VA_ARGS__); printMutex.unlock();
#define p(x) printMutex.lock(); cout << x << endl; printMutex.unlock();
using namespace std;
std::mutex printMutex;

void parallelBFS:: InternalThreadEntry() {
    int currentThread = getThreadId();
    float fromEdge = currentThread * ((float)graphData->numEdges / numThreads);
    float toEdge = (currentThread + 1) * ((float)graphData->numEdges / numThreads) - 1;
    if (toEdge > graphData->numEdges)
        toEdge = graphData->numEdges - 1;
    p("Thread : " << currentThread << " = " << fromEdge << " - " << toEdge);
    for (int i = fromEdge; i < toEdge; i++) {
        print("Thread : %d\n", currentThread);
    }
}
