#include <stdio.h>
int main() {

    int length = 1000;
    int tid, index = 1, add = 1, level = 1;
    int array[1000], i;
    for (i = 0; i < length; i++)
        array[i] = 1000;
    while (add < length && add <= 2) {
        printf("Level %d. Add = %d\n", level, add);
        for (tid = 0; tid < length; tid++) {
            int temp = 2 * tid * index;
            if (temp + add < length) {
                array[temp] += array[temp + add];
                //printf("\tTid(%d). %d + %d\n", tid, temp, temp + add);
            }
        }
       index = index << 1;
       add = add << 1;
       level++;
    }
    //printf("Result : %d\n", array[0]);
    for (tid = 0; tid < length; tid++) {
        printf("[%d] = %d\n", tid, array[tid]);
    }
    return 0;
}
