//#ifndef Graph_H
//#define Graph_H
#include <vector>
#include <iostream>

using namespace std;

typedef pair<int, pair<int, int> > triple;
typedef vector<triple> listTriple;

class GraphClass {

public:
//private:
    listTriple edgeList;
    int numNodes, numEdges;

    GraphClass() {
        numNodes = 0;
        numEdges = 0;
    }
    GraphClass(int a, int b) {
        numNodes = a;
        numEdges = b;
    }

    void addEdge(triple newEdge) {
        edgeList.push_back(newEdge);
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
