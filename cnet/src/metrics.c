/**
 * Metric Functions for CNet.
 */


#include <math.h>
#include "metrics.h"


double accuracy(
    double const *pred,
    double const *real,
    int size
){
    double res = 0;
    for(int i = 0; i < size; i++) {
        res += round(pred[i]) == round(real[i]);
    }
    return res / size;
}


/// getters


cnet_metric_fun cnet_get_metric(enum cnet_metric type) {
    switch(type) {
        case metric_accuracy: return accuracy;
    }
}


const char *cnet_get_metric_name(enum cnet_metric type) {
    switch(type) {
        case metric_accuracy: return "Accuracy";
    }
}
