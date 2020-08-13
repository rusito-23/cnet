/**
 * CNet Interface.
 */

#ifndef CNET_H
#define CNET_H

#include <stdio.h>
#include "activation.h" 
#include "loss.h" 
#include "metrics.h" 


/// cnet struct definition

struct cnet;
struct clayer;


typedef struct cnet {
    /* input/output dimensions */
    int in_size, out_size;
    /* layer helpful indices */
    int n_layers, last_layer;
    /* layers */
    struct clayer **layers;
} cnet;


typedef struct clayer {
    /* input/output dimensions */
    int in_size, out_size;
    /* activation type */
    enum cnet_act act_type; 
    /* trainable parameters */
    double **weights;
    double *bias;
    /* activation output - A = act(Z) */
    double *A;
    /* sum of inputs * weights - Z = sum(i*w) + bias */
    double *Z;
    /* cost derivative over the output */
    double *dC_dA;
    /* output derivative over Z */
    double *dA_dZ;
} clayer;


/// api definition


/**
 * Create CNet.
 *
 * Allocs the necessary memory for the network.
 *
 * @param int in_size: Input size
 * @param int out_size: Output size
 * @param int n_layers: Number of layers that should be allocated.
 * @return cnet *: Pointer to the allocated structure.
 */
cnet *nn_init(
    int in_size,
    int out_size,
    int n_layers
);


/**
 * Free CNet.
 *
 * @param cnet *nn: CNet
 */
void nn_free(
    cnet *nn
);


/**
 * Add a Layer to the CNet.
 *
 * Assumes that the nn_init method was called, and all cnet* attributes
 * are correctly initialized.
 * The weights for the layer will be initialized with uniform
 * random numbers between 0 and 1.
 *
 * @param cnet *nn: CNet
 * @param int in_size: Input size for the new layer
 * @param int out_size: Output size for the new layer
 * @param cnet_act_type act_type: Activation type for the new layer.
 */
void nn_add(
    cnet *nn,
    int in_size,
    int out_size,
    enum cnet_act act_type
);


/**
 * CNet Prediction. 
 *
 * @param const cnet *nn: CNet
 * @param const double *X: Input (sized nn->in_size)
 * @return const double *: Pointer to results (sized nn->out_size)
 */
const double *nn_predict(
    cnet const *nn,
    double const *X
);


/**
 * Train the network.
 *
 * Performs backward passes through the net using SGD (single training sample)
 * It shuffles both the X and Y in every epoch to achieve better results.
 *
 * @param const cnet *nn: CNet
 * @param double const** X: Inputs (size data_len x nn->in_size)
 * @param double const** Y: Expected output (size data_len x nn->out_size)
 * @param int data_len: Number of training samples
 * @param cnet_loss loss_type: Loss type to use
 * @param cnet_metric metric_type: Metric type to use
 * @param double learning_rate: Learning rate
 */
void nn_train(
    cnet const *nn,
    double **X,
    double **Y,
    int data_len,
    enum cnet_loss loss_type,
    enum cnet_metric metric_type,
    double learning_rate,
    int epochs
);


/**
 * Load the network from FILE.
 *
 * Initializes and reads the network weights from the given FILE.
 * The FILE must follow the given structure, or else this function will return
 * a corrupt nn and maybe even crash when being fred.
 *
 * @param FILE: network saved file.
 * @return cnet *: cnet
 */
cnet *nn_load(
    FILE *in
);


/**
 * Save the network into FILE.
 *
 * Saves the given network into a file, following the following structure:
 * in_size out_size n_layers
 * layer_in_size layer_out_size layer_act_type
 * layer_bias ...
 * layer_weights ...
 * ...
 *
 * @param cnet *nn: cnet
 * @param FILE out: output file
 */
void nn_save(
    cnet const *nn,
    FILE *out
);


#endif /* CNET_H */
