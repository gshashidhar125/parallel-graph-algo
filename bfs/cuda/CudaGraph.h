//#ifndef Graph_H
//#define Graph_H
#include <vector>
#include <iostream>

using namespace std;

typedef pair<int, pair<int, int> > triple;
typedef vector<triple> listTriple;

class CudaGraphClass {

public:
//private:
    int numNodes, numEdges;
    int *row[2];

    GraphClass() {
        numNodes = 0;
        numEdges = 0;
    }
    GraphClass(int nodes, int edges) {
        numNodes = nodes;
        numEdges = edges;
// NumVertices starting from 1 to NumVertices plus an addition Sentinel node
// which points to the last index of the Graph[1] array.
        row[0] = calloc(numNodes + 2, sizeof(int));
        row[1] = calloc(numEdges + 1, sizeof(int));
    }

    void populate() {


    }
    void print() {
    
        /*for (listTriple::iterator it = edgeList.begin(); it != edgeList.end(); it++) {
            cout << (*it).first << (*it).second.first << (*it).second.second << "\n";
        }*/
        for (int i = 0; i < numEdges; i++) {
            cout << edgeList[i].first << " - " << edgeList[i].second.first << "\t = " << edgeList[i].second.second << "\n";
        }
    }
    void change() {
        edgeList[0].first = 99;
        edgeList[0].second.second = 44;
    }
};
//#endif
