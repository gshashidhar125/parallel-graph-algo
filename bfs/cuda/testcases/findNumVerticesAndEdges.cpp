

// first give input to the input_formatting algorithm and then with search function find weather there is zero in output, if it is not there which means
// that the input set considers vertices starting from 1. now give this output file as a input to this algo.

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>
#include <stdio.h>

using namespace std;


# define LINE_COUNT 10

int main() {

    std::vector<std::pair<int, int> > v ;
    std::vector<std::pair<int, int> > :: iterator it;
    int vertices, edges;

    // Input no of vertices and edges from the io redirected file.
/*    cin >> vertices;
    cin >> edges;
*/    
    int p, q, res;
   
    // Inputing whole edge set from input file into the vector.
    while ((res = scanf("%d %d", &p, &q)) != 0) {
        if (res < 0)
            break;
/*    for (int i=0; i<edges; i++)
    {
      cin >> p;
      cin >> q;
*/      if (p > vertices) vertices = p;
        if (q > vertices) vertices = q;
        v.push_back(std::make_pair<int, int>(p,q));     
    }
    
   /* 
  //checks in whole vector weather there is any zero vertex
  for(it = v.begin(); it != v.end(); it++) 
    {
        if( it->first ==0 || it->second ==0 )
	  flag = 1;
    }
   */
    std::sort(v.begin(), v.end());
   /*
    edges = v.size();
    
    for(it = v.begin(); it != v.end(); it++)
        vertices = it->first;    
    
    std::cout << vertices << "  " << edges << std::endl;
*/
   edges = v.size();
   std::cout << vertices << "  " << edges << std::endl;
//   std::cout << "0" << "  " << "0" << std::endl;
/*   for(it = v.begin(); it != v.end(); it++) 
    {
        std::cout << it->first << "  " << it->second << std::endl;
    }*/
}
