/*****************************************************************************
 *                               ACTIVATION
 * Implementation of available activation functions and some helpers.
 ****************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../include/activation.h"
#include "../include/helpers.h"


/// ReLU


/**
 * ReLU
 * Rectified Linear Units.
 *
 * @param double *: Sum of weights * inputs + Bias
 * @param int: Size
 */
void ReLU(
    double *a,
    int size
){
    for(int i = 0; i < size; i++)
        a[i] = a[i] > 0 ? a[i] : 0;
}


/**
 * ReLU Derivative.
 *
 * @param double *: ReLU Output
 * @param double *: destination array
 * @param int: source and destination size
 */
void ReLU_Dx(
    double *s,
    double *a,
    int size
){
    for(int i = 0; i < size; i++)
        a[i] = s[i] >= 0 ? 1 : 0;
}


/**
 * ReLU Delta Computation
 *
 * @param double *: Output
 * @param double *: Previous delta
 * @param int: Size
 */
void ReLU_Delta(
    double *output,
    double *delta,
    int size
){
    // activation derivative
    double *activation_dx = malloc(sizeof(double)*size);
    ReLU_Dx(
        output,
        activation_dx,
        size
    );

    // update delta
    for(int i = 0; i < size; i++)
        delta[i] *= activation_dx[i];

    free(activation_dx);
}


/// Sigmoid


/**
 * Sigmoid
 *
 * @param double *: Sum of weights * inputs + Bias
 * @param int: Size
 */
void Sigmoid(
    double *a,
    int size
){
    for(int i = 0; i < size; i++)
        a[i] = 1/(1 + expf(-a[i]));
}


/**
 * Sigmoid Derivative
 *
 * @param double *s: Sigmoid output
 * @param double *a: Destination array
 * @param int size: Source and Destination size
 */
void Sigmoid_Dx(
    double *s,
    double *a,
    int size
){
    for(int i = 0; i < size; i++)
        a[i] = s[i] * (1 - s[i]);
}


/**
 * Sigmoid Delta Computation
 *
 * @param double *: Output
 * @param double *: Previous delta
 * @param int: Size
 */
void Sigmoid_Delta(
    double *output,
    double *delta,
    int size
){
    // activation derivative
    double *activation_dx = malloc(sizeof(double)*size);
    Sigmoid_Dx(
        output,
        activation_dx,
        size
    );

    // update delta
    for(int i = 0; i < size; i++)
        delta[i] *= activation_dx[i];

    free(activation_dx);
}


/// SoftMax


/**
 * SoftMax
 *
 * @param double *: Sum of weights * inputs + Bias
 * @param int: Size
 */
void SoftMax(
    double *a,
    int size
){
    // get max z
    double max = a[0];
    for(int i = 0; i < size; i++)
        max = max < a[i] ? a[i] : max;

    // sum of exp(z - max)
    double sum = 0;
    for(int i = 0; i < size; i++)
        sum += expf(a[i] - max);

    // populate the destination array
    for(int i = 0; i < size; i++)
        a[i] = expf(a[i] - max) / sum;
}


/**
 * SoftMax Derivative
 *
 * This is particular derivative, since it fills a matrix instead
 * of a vector. This matrix will be nxn, n being the activation output length.
 * Don't panic about the Jacobian Matrix, since it will be dot-multiplied 
 * by the cost derivative (vector of size n), resulting in an n-sized vector.
 *
 * @param double *s: The output vector of the softmax function
 * @param double **d: Destination Matrix
 * @param int size: Input Z size (Jacobian Matrix size being: size x size)
 */
void SoftMax_Dx(
    double *s,
    double **d,
    int size
){
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            if (i == j) {
                d[i][j] = s[i] * (1 - s[i]);
            } else {
                d[i][j] = -s[i] * s[i];
            }
        }
    }
}


/**
 * SoftMax Delta Computation.
 *
 * This activation function results in a particular delta computation,
 * since we need to use a Jacobian matrix and perform a dot product 
 * with the previous delta value, which is a vector.
 *
 * @param double *: SoftMax Output 
 * @param double *: Previous Delta
 * @param int: Size
 */
#pragma clang diagnostic ignored "-Wunused-parameter"
void SoftMax_Delta(
    double *output,
    double *delta,
    int size
){
    // we don't do anything here, as the derivative is
    // already handled in the derivative of the cross entropy loss
}



/// Helpers


cnet_act_func *cnet_get_act(enum cnet_act_type type) {
    switch(type) {
        case relu_act: return ReLU;
        case sigmoid_act: return Sigmoid;
        case softmax_act: return SoftMax;
    }
}


cnet_act_func_delta *cnet_get_act_delta(enum cnet_act_type type) {
    switch(type) {
        case relu_act: return ReLU_Delta;
        case sigmoid_act: return Sigmoid_Delta;
        case softmax_act: return SoftMax_Delta;
    }
}
