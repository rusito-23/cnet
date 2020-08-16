/**
 * Miscellaneous Helpers for CNet.
 *
 * Contains several helper functions, to keep the cnet code cleaner.
 */

#ifndef CNET_HELPERS_H
#define CNET_HELPERS_H

/// Array helpers


/**
 * Double Array Sum.
 * Performs the sum of an array.
 *
 * @param double *arr: The array
 * @param int size: Array size
 * @return double: sum
 */
double cnet_sum(double *arr, int size);


/**
 * Double Array Mean.
 *
 * @param double *arr: The array
 * @param int size: Array size
 * @return double: sum
 */
double cnet_mean(double *arr, int size);


/**
 * Random shuffle an array (in-place).
 *
 * @param double *arr: The array
 * @param int size: Array size
 */
void cnet_shuffle(int *arr, int size);


/**
 * Idx Array
 *
 * Allocates and initialize an array with indexes
 * starting from 0 to given size.
 *
 * @param int size;
 * @return int *: resulting array
 */
int *cnet_idx(int size);


/**
 * ArgMax
 *
 * Returns the index for the max element in a given array.
 *
 * @param double *arr;
 * @param int size: arr size
 * @return double: Index of max element (as double) 
 */
double cnet_argmax(double const *arr, int size);


#endif /* CNET_HELPERS_H */
