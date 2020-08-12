/**
 * Miscellaneous Helpers for CNet.
 */

#include <stdlib.h>
#include "helpers.h"


/// Array Helpers


/**
 * Array Sum */
double cnet_sum(double *arr, int size) {
    double sum = 0;
    for(int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}


/**
 * Array Mean */
double cnet_mean(double *arr, int size) {
    return cnet_sum(arr, size) / size;
}


/**
 * Array random suffle
 * https://stackoverflow.com/questions/6127503/shuffle-array-in-c */
void cnet_shuffle(int *arr, int size) {
    if (size < 1) return;
    for (int i = 0; i < size - 1; i++) {
      int j = i + rand() / (RAND_MAX / (size - i) + 1);
      int t = arr[j];
      arr[j] = arr[i];
      arr[i] = t;
    }
}


/**
 * Idx Array */
int *cnet_idx(int size) {
    int *ret = malloc(sizeof(int)*size);
    for(int i = 0; i < size; i++) {
        ret[i] = i;
    }
    return ret;
}
