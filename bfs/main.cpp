#include <iostream>
#include <vector>
#include <fstream>
#include <pthread.h>
#include "Graph.h"
#include "Thread.h"

using namespace std;

class parallelBFS;

int main(int argc, char *argv[]) {

    if (argc != 2 || argv[1] == NULL) {
        printf("Wrong Number of Arguments\n");
        exit(1);
    }
    
    ifstream inputFile;

    inputFile.open(argv[1]);
    if (!inputFile.is_open()){
        cout << "invalid file";
        exit(1);
    }

    int numNodes, numEdges;
    inputFile >> numNodes >> numEdges;
    GraphClass graph(numNodes, numEdges);
    triple tempEdge;

    for (int i = 0; i < numEdges; i++) {
        inputFile >> tempEdge.first >> tempEdge.second.first >> tempEdge.second.second;
        graph.addEdge(tempEdge);
    }

    //cout << numNodes << ", " << numEdges << "\n";
    graph.print();

    int numThreads = 4;

    // Partition the graph based on the number of edges and the threads.
    vector<parallelBFS> threadList(numThreads);

    for (int i = 0; i < numThreads; i++) {
        threadList[i].setThreadId(i);
        threadList[i].setGraph(&graph);
        threadList[i].StartInternalThread();
    }

    for (int i = 0; i < numThreads; i++) {
        //cout << threadList[i].get() << "\n";
        threadList[i].WaitForInternalThreadToExit();
    }
    inputFile.close();
    return 0;
}
