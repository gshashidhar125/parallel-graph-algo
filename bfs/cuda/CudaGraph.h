//#ifndef CudaGraph_H
//#define CudaGraph_H
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>

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
    int *row[2];
    ifstream inputFile;

    CudaGraphClass() {
        numVertices = 0;
        numEdges = 0;
    }
    CudaGraphClass(int vertices, int edges) {
        numVertices = vertices;
        numEdges = edges;
// NumVertices starting from 1 to NumVertices plus an addition Sentinel node
// which points to the last index of the row[1] array.
        row[0] = new int [numVertices + 2]();
        row[1] = new int [numEdges + 1]();
    }

    void populate(char *fileName) {
        
        inputFile.open(fileName);
        if (!inputFile.is_open()){
            cout << "invalid file";
            return;
        }

        cout << numVertices << "--" << numEdges << endl;
        int **AdjMatrix, i, j, k;
        AdjMatrix = new int* [numVertices + 1]();
        for (i = 0; i <= numVertices; i++) {
    
            AdjMatrix[i] = new int [numVertices + 1]();
        }
        i = numEdges;
        int lastj = 1, currentIndex = 1;
        inputFile >> j >> k;
        while(i) {
    
            //scanf("%d %d", &j, &k);
            inputFile >> j >> k;
            cout << "Read: " << j << "-- " << k;
            AdjMatrix[j][k] = 1;
            while (lastj <= j || lastj == 1) {
                if (lastj == 1) {
                    row[0][0] = currentIndex;
                    row[0][1] = currentIndex;
                }else {
                    row[0][lastj] = currentIndex;
                }
                lastj++;
            }
    //        if (AdjMatrix[k][j] != 1)
                row[1][currentIndex] = k;
            currentIndex ++;
            i--;
        }
        row[1][0] = 0;
        // Sentinel node just points to the end of the last node in the graph
        while (lastj <= numVertices + 1) {
            row[0][lastj] = currentIndex;
            lastj++;
        }
        //row[0][lastj+1] = currentIndex;
        for (i = 1; i <= numVertices + 1; i++)
            print("Vertex: %d = %d\n", i, row[0][i]);
    
        print("Second Array:\n");
        for (i = 1; i <= numEdges; i++)
            print("Edges: Index: %d, Value = %d\n", i, row[1][i]);
    
        j = 1;
        for (i = 1; i <= numVertices; i++) {
    
            currentIndex = row[0][i];
            while (currentIndex < row[0][i+1]) {
    //            print("%d %d\n", i, row[1][currentIndex]);
                if (AdjMatrix[i][row[1][currentIndex]] != 1 /*&&
                    AdjMatrix[row[1][currentIndex]][i] != 1*/) {
                    outs("\n\nGraph Do not Match\n\n");
                    break;
                }
                j++;
                currentIndex ++;
            }
        } 
    }
    
    void printGraph() {

    }

    void callBFS();
    int copyGraphToDevice();
};
//#endif
