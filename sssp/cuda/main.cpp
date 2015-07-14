#include <iostream>
#include <vector>
#include <fstream>
#include "CudaGraph.h"

#include <cuda.h>
using namespace std;

class parallelSSSP;
NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);

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

    //dump << "Nothing will be printed";
    int numNodes, numEdges;
    inputFile >> numNodes >> numEdges;
    CudaGraphClass graph(numNodes, numEdges, 1024);

    cout << "Graph Population began\n";
    graph.populate(argv[1]);
    cout << "Graph Population end\n";
    graph.copyGraphToDevice();
    cout << "Copy graph end\n";

    //graph.printGraph();
    //std::cout << "Main function\n";

    graph.callSSSP();
    return 0;
}
