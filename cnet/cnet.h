/**
 * CNet Interface.
 */

#ifndef CNET_H
#define CNET_H

#include "activation.h" 
#include "loss.h" 


/// cnet struct definition

struct cnet;
struct clayer;


typedef struct cnet {
    int in_size, out_size;
    int n_layers, last_layer;

    struct clayer **layers;
} cnet;


typedef struct clayer {
    int in_size, out_size;
    enum cnet_act act_type; 

    double bias;
    double **weights;

    double *output;
    double *delta;

    // TODO: these arrays should only be allocated in training mode
    double *dC_dA;
    double *dA_dZ;
} clayer;


/// api definition


/**
 * Create Network.
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
 * Free Network.
 *
 * @param cnet *nn: Network
 */
void nn_free(
    cnet *nn
);


/**
 * Add a Layer to the Network.
 *
 * Assumes that the nn_init method was called, and all cnet* attributes
 * are correctly initialized.
 * The weights for the layer will be initialized with uniform
 * random numbers between 0 and 1.
 *
 * @param cnet *nn: Network
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
 * Network Forward Pass. 
 *
 * @param const cnet *nn: Network
 * @param const double *X: Input
 * @return const double *: Pointer to results
 */
const double *nn_predict(
    cnet const *nn,
    double const *X
);


/**
 * Train the network.
 *
 * @param const cnet *nn: Network
 * @param double const** X: Inputs (size data_len x nn->in_size)
 * @param double const** Y: Expected output (size data_len x nn->out_size)
 * @param int data_len: Number of training samples
 * @param cnet_loss_type loss: Loss type to use
 * @param double learning_rate: Learning rate
 */
void nn_train(
    cnet const *nn,
    double **X,
    double **Y,
    int data_len,
    enum cnet_loss loss_type,
    double learning_rate,
    int epochs
);


#endif /* CNET_H */
