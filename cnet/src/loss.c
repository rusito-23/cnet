/**
 * Loss Functions for CNet.
 */


#include <math.h>
#include "loss.h"


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
        dst[i] = (pred[i] - real[i]);
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
