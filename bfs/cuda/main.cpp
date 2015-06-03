#include <iostream>
#include <vector>
#include <fstream>
#include "Graph.h"
#include "Thread.h"

#include <cuda.h>
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

    graph.populate();

    triple tempEdge;

    for (int i = 0; i < numEdges; i++) {
        inputFile >> tempEdge.first >> tempEdge.second.first >> tempEdge.second.second;
        graph.addEdge(tempEdge);
    }

    //cout << numNodes << ", " << numEdges << "\n";
    graph.print();

    

    return 0;
}
