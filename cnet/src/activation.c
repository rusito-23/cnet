/**
 * Activation Functions for CNet.
 */

#include <math.h>
#include <float.h>
#include "activation.h"


/// ReLU


void relu(
    double const *src,
    double *dst,
    int size
){
    for(int i = 0; i < size; i++) {
        dst[i] = src[i] > 0 ? src[i] : 0;
    }
}


void relu_dx(
    double const *src,
    double *dst,
    int size
){
    for(int i = 0; i < size; i++) {
        dst[i] = src[i] >= 0 ? 1 : 0;
    }
}


/// Sigmoid


void sigmoid(
    double const *src,
    double *dst,
    int size
){
    for(int i = 0; i < size; i++) {
        dst[i] = 1/(1 + expf(-src[i]));
    }
}


void sigmoid_dx(
    double const *src,
    double *dst,
    int size
){
    for(int i = 0; i < size; i++) {
        double sig = 1/(1 + expf(-src[i]));
        dst[i] = sig * (1 - sig);
    }
}


/// SoftMax


void softmax(
    double const *src,
    double *dst,
    int size
){
    double denom = 0;
    for(int i = 0; i < size; i++) {
        denom += expf(src[i]);
    }

    for(int i = 0; i < size; i++) {
        dst[i] = expf(src[i]) / denom;
    }
}


void softmax_dx(
    double const *src,
    double *dst,
    int size
){
    double denom = 0;
    for(int i = 0; i < size; i++) {
        denom += expf(src[i]);
    }

    for(int i = 0; i < size; i++) {
        double comm = -expf(src[i])/pow(denom, 2);
        double factor = 0;

        for(int j = 0; i < size && i != j; j++) {
            factor += expf(src[j]);
        }

        dst[i] = comm*factor;
    }

}


/// Activations getters


cnet_act_fun cnet_get_act(enum cnet_act type) {
    switch(type) {
        case act_relu: return relu;
        case act_sigmoid: return sigmoid;
        case act_softmax: return softmax;
    }
}


cnet_act_fun cnet_get_act_dx(enum cnet_act type) {
    switch(type) {
        case act_relu: return relu_dx;
        case act_sigmoid: return sigmoid_dx;
        case act_softmax: return softmax_dx;
    }
}
