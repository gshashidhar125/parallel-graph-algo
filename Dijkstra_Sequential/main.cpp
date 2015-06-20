#include <iostream>
#include <list>

class Node_t;
typedef std::list<Node_t *> NodeList_t;

class Node_t {
public:    
    int key;
    NodeList_t child;

    Node_t(int k) {
        key = k;
    }

    int getRank() {
        return child.size();
    }

    void insert(Node_t *newNode) {
        // Min-Heap insert
        //child.push_back(newNode);
    }
};

class FibHeap {

    NodeList_t root;
    Node_t *min;

public:
    FibHeap() {
        min = NULL;
    }

    void insert(int k) {

        Node_t *newNode = new Node_t(k);
        root.push_back(newNode);
        if (!min || min->key > k)
            min = newNode;
    }

    void updateMin() {
        
        int minKey = 9999;
        Node_t *minNode = NULL;
        for (NodeList_t::iterator it = root.begin(); it != root.end(); ++it ) {
            if((*it)->key < minKey) {
                minNode = *it;
            }
        }
        if (!minNode)
            min = NULL;
        else
            min = minNode;
    }

    Node_t * merge(Node_t *node1, Node_t *node2) {
        if (node1->key < node2->key) {
            node1->child.push_back(node2);
            return node1;
        }
        else {
            node2->child.push_back(node1);
            return node2;
        }
    }

    void meld() {

        Node_t *rank[100] = {};
        Node_t *current, *mergedNode;
        int i = 0, currentRank;

        for (NodeList_t::iterator it = root.begin(); it != root.end(); ++it ) {
            current = *it;
            currentRank = current->getRank();
            if (rank[currentRank] == NULL) {
                rank[currentRank] = current;
            } else {
                mergedNode = merge(current, rank[currentRank]);
                if (mergedNode == current)
                    root.remove(current);
                else
                    root.remove(rank[currentRank]);
            }
        }
    }

    int delete_min() {
        
        if (!min) {
            std::cout << "No members in the heap. No Min element found\n";
            return 1;
        }
        
        for (NodeList_t::iterator it = min->child.begin(); it != min->child.end(); ++it ) {
            root.push_back(*it);
        }
        root.remove(min);
        updateMin();

        meld();
        return 0;
    }

    void printFBHeap() {

        for (NodeList_t::iterator it = root.begin(); it != root.end(); ++it ) {
            std::cout << (*it)->key << "\n";
        }
    }
};

int main() {

    FibHeap FH;
    FH.insert(8);
    FH.delete_min();
    FH.insert(4);
    FH.insert(6);
    FH.delete_min();
    FH.insert(3);
    FH.insert(9);
    
    FH.printFBHeap();
    return 0;
}
