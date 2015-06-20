#include <iostream>
#include <list>

class Node_t {

public:    
    int key;
    std::list<Node_t *> child;
    int rank;   // Number of Children

    Node_t(int k) {
        key = k;
        rank = 0;
    }

    void insert(Node_t *newNode) {
        //child.add(newNode);
    }
};
typedef std::list<Node_t *> NodeList_t;
class FibHeap {

    std::list<Node_t *>root;
    Node_t *min;

public:
    FibHeap() {
        min = NULL;
        //for (int i = 0; i < 10; i++)
        //    root[0] = NULL;
    }

    void insert(int k) {

        Node_t *newNode = new Node_t(k);
        root.push_back(newNode);
        if (!min || min->key > k)
            min = newNode;
    }
    int delete_min() {
        
        Node_t *minNode = NULL;
        int minKey = 0;
        for (NodeList_t::iterator it = root.begin(); it != root.end(); ++it ) {
            if ((**it).key < minKey) {
                minKey = (**it).key;
                minNode = *it;
            }
        }
        if (!minNode) {
            std::cout << "No members in the heap. No Min element found\n";
            return 1;
        }
        
        for (NodeList_t::iterator it = minNode->child.begin(); it != minNode->child.end(); ++it ) {
            root.push_back(*it);
        }
        root.remove(minNode);
        return 1;
    }

    void printFBHeap() {

        for (NodeList_t::iterator it = root.begin(); it != root.end(); ++it ) {
            std::cout << (*it)->key << "\n";
        }
    }
};

int main() {

    FibHeap FH;
    FH.insert(3);
    FH.insert(4);
    FH.insert(6);
    FH.insert(8);
    FH.delete_min();
    FH.insert(9);
    
    FH.printFBHeap();
    return 0;
}
