/**
 * Loss Functions for CNet.
 */


#include <math.h>
#include <stdlib.h>
#include "loss.h"
#include "helpers.h"


/// mse


void mse(
    double const *pred,
    double const *real,
    double *dst,
    int size
){
    for(int i = 0; i < size; i++) {
        dst[i] = pow(pred[i] - real[i], 2);
    }
}


void mse_dx(
    double const *pred,
    double const *real,
    double *dst,
    int size
){
    for(int i = 0; i < size; i++) {
        dst[i] = 2 * (pred[i] - real[i]);
    }
}


/// getters


cnet_loss_fun cnet_get_loss(enum cnet_loss type) {
    switch(type) {
        case loss_mse: return mse;
    }
}


cnet_loss_fun cnet_get_loss_dx(enum cnet_loss type) {
    switch(type) {
        case loss_mse: return mse_dx;
    }
}


double cnet_loss_mean(
    enum cnet_loss type,
    double const *pred,
    double const *real,
    int size
){
    double *loss_arr = malloc(sizeof(double) * size);
    cnet_get_loss(type)(
        pred, 
        real,
        loss_arr,
        size
    );
    double loss_mean = cnet_mean(loss_arr, size);
    free(loss_arr);
    return loss_mean;
}
