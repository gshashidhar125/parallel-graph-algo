#include <iostream>
#include <string>
#include <time.h>
#include <stdlib.h>
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

    void printNode(std::string spaces) {

        std::cout << spaces << key << "\n";
        spaces = spaces + "  ";
        for (NodeList_t::iterator it = child.begin(); it != child.end(); ++it ) {
            (*it)->printNode(spaces);
        }
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
                minKey = minNode->key;
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

        Node_t *rank[1000] = {};
        Node_t *current, *mergedNode;
        int i = 0, currentRank;
        bool done, alreadyMoved;

        for (NodeList_t::iterator it = root.begin(); it != root.end(); ) {
            current = *it;
            done = false;
            alreadyMoved = false;
            while (!done) {
                currentRank = current->getRank();
                if (rank[currentRank] == NULL) {
                    rank[currentRank] = current;
                    done = true;
                } else {
                    mergedNode = merge(current, rank[currentRank]);
                    if (mergedNode == current) {
                        root.remove(rank[currentRank]);
                    }
                    else {
                        if (*it == current) {
                            alreadyMoved = true;
                            it++;
                        }
                        root.remove(current);
                    }
                    current = mergedNode;
                    rank[currentRank] = NULL;
                }
            }
            if (!alreadyMoved)
                it++;
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
            (*it)->printNode("");
            //std::cout << (*it)->key << "\n";
            //if ((*it)->key == 3)
            //    it = root.erase(it);
            //std::advance(it, 1);
            //Node_t *temp = *it;
            //root.remove(temp);
        }
    }
};

int main() {

    FibHeap FH;

#ifdef TEST
    FH.insert(8);
    FH.delete_min();
    FH.insert(4);
    FH.insert(6);
    FH.delete_min();
    FH.insert(3);
    FH.insert(4);
    FH.insert(9);
    FH.insert(1);
    FH.insert(5);
    FH.delete_min();
#elif RANDOM

    srand(time(NULL));
    int count1 = 0, count2 = 0;
    for (int i = 0; i < 1000; i++) {

        //std::cout << rand() % 10 << "\n";
        if (rand() % 10 < 6) {
            FH.insert(rand() % 1000);
            count1++;
//            std::cout << "Insert\n";
        }else {
            FH.delete_min();
            count2++;
//            std::cout << "  Delete\n";
        }
    }
#endif
    FH.printFBHeap();
    std::cout << "Count: " << count1 << " - " << count2 << "\n";
    return 0;
}
