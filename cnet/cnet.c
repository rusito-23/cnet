/**
 * CNet Implementation
 *
 * Implements an Artificial Neural Network and several methods to work with.
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "cnet.h"

#define sfree(x) free(x); x = NULL

/**
 * Create Network. */
cnet *nn_init(
    int in_size,
    int out_size,
    int n_layers
){
    cnet *nn = malloc(sizeof(cnet));
    nn->in_size = in_size;
    nn->out_size = out_size;
    nn->n_layers = n_layers;
    nn->layers = malloc(sizeof(clayer*) * n_layers);
    nn->last_layer = 0;
    return nn;
}


/**
 * Free Network. */
void nn_free(
    cnet *nn
){
    for(int i = 0; i < nn->n_layers; i++) {
        struct clayer *layer = nn->layers[i];
        for(int j = 0; j < layer->out_size; j++) {
            free(layer->weights[j]);
        }

        free(layer->weights);
        free(layer->bias);
        free(layer->A);
        free(layer->Z);
        free(layer->dC_dA);
        free(layer->dA_dZ);
        free(layer);
    }
    free(nn->layers);
    free(nn);
}


/**
 * Add a Layer to the Network. */
void nn_add(
    cnet *nn,
    int in_size,
    int out_size,
    enum cnet_act act_type
){
    // check the input/output size
    assert(nn->last_layer == 0 ||
           in_size == nn->layers[nn->last_layer - 1]->out_size);
    assert(nn->last_layer < nn->n_layers - 1 || out_size == nn->out_size);

    // alloc the layer
    struct clayer* layer = malloc(sizeof(clayer));
    layer->in_size = in_size;
    layer->out_size = out_size;
    layer->act_type = act_type;

    layer->weights = malloc(sizeof(double*)*layer->out_size);
    layer->bias = malloc(sizeof(double)*layer->out_size);
    layer->A = malloc(sizeof(double)*layer->out_size);
    layer->Z = malloc(sizeof(double)*layer->out_size);
    layer->dC_dA = malloc(sizeof(double)*layer->out_size);
    layer->dA_dZ = malloc(sizeof(double)*layer->out_size);

    // randomize weights and biases between 0 and 1
    for(int i = 0; i < layer->out_size; i++) {
        layer->bias[i] = ((double)rand())/((double)RAND_MAX);
        layer->weights[i] = malloc(sizeof(double)*layer->in_size);
        for(int j = 0; j < layer->in_size; j++) {
            layer->weights[i][j] = ((double)rand())/((double)RAND_MAX);
        }
    }

    // add layer to the net
    assert(nn->last_layer < nn->n_layers);
    nn->layers[nn->last_layer++] = layer;
}


/**
 * Network Forward Pass
 *
 * Simply passes a given input (with expected size) through the net.
 * Does not return the result pointer, this should be accessed through
 * the nn->layers[last_layer - 1]->result;
 *
 * @param cnet const *nn: Network
 * @param double const *X: Input (sized nn->in_size)
 */
void nn_forward(
    cnet const *nn,
    double const *X
){
    double const *in = X;

    // pass through every layer in the net
    for(int i = 0; i < nn->n_layers; i++) {
        struct clayer *layer = nn->layers[i];

        // pass through every neuron in the net
        for(int k = 0; k < layer->out_size; k++) {
            // compute z for neuron
            double z = 0;
            for(int j = 0; j < layer->in_size; j++) {
                z += layer->weights[k][j] * in[j];
            }
            
            z += layer->bias[i];
            layer->Z[k] = z;
        }


        // activate the layer output
        cnet_act_fun act = cnet_get_act(layer->act_type);
        act(layer->Z, layer->A, layer->out_size);

        // set input for next layer
        in = layer->A;
    }
}


/**
 *
 * Network Backward Pass
 *
 * Performs a single backpropagation step, using SGD, hence
 * it only takes one train sample.
 *
 * @param cnet const *nn: Network
 * @param double *X: Input (sized nn->in_size)
 * @param double *Y: Expected output (sized nn->out_size)
 * @param cnet_loss_type: Loss type to use
 * @param double learning_rate: Learning Rate
 */
void nn_backward(
    cnet const *nn,
    double *X,
    double *Y,
    enum cnet_loss loss_type,
    double learning_rate
){
    // backpropagation

    for(int l = nn->n_layers; l-->0;) {

        struct clayer* layer = nn->layers[l];
        struct clayer* next = l < (nn->n_layers - 1) ? nn->layers[l + 1] : NULL;
        struct clayer* previous = l > 0 ? nn->layers[l - 1] : NULL;

        // activation derivative over the layer's delta
        cnet_get_act_dx(layer->act_type)(
            layer->Z,
            layer->dA_dZ,
            layer->out_size
        );

        // delta derivative over the weights
        // this corresponds with the previous layer's output
        double *dZ_dW = previous == NULL ? X : previous->A;

        // cost derivative over the current output
        if (next == NULL) {
            cnet_loss_fun loss_dx = cnet_get_loss_dx(loss_type);
            loss_dx(
                layer->A,
                Y,
                layer->dC_dA,
                nn->out_size
            );
        } else {
            // as this is an intermediate layer,
            // we need to compute the derivative of the cost over the current
            // activation output using the previously computed dC_dA, along
            // with the dependencies of these values for the current layer
            // activation output and weights.
            for(int j = 0; j < layer->out_size; j++) {
                double dC_dA_k = 0;
                for(int k = 0; k < next->out_size; k++) {
                    dC_dA_k += next->dC_dA[k] * next->dA_dZ[j] * next->weights[k][j];
                }
                layer->dC_dA[j] = dC_dA_k;
            }
        }

        // update the weights and biases
        for(int i = 0; i < layer->out_size; i++) {
            double dC_dB = layer->dA_dZ[i] * layer->dC_dA[i];
            layer->bias[i] -= learning_rate * dC_dB;

            for(int j = 0; j < layer->in_size; j++) {
                double dC_dW = dZ_dW[j] * dC_dB;
                layer->weights[i][j] -= learning_rate * dC_dW;
            }
        }
    }
}


/**
 * Network Prediction. */
const double *nn_predict(
    cnet const *nn,
    double const *X
){
    // pass the input through the net
    nn_forward(nn, X);

    // return the output for the last layer
    return nn->layers[nn->n_layers - 1]->A;
}


/**
 * Network Train Algorithm */
void nn_train(
    cnet const *nn,
    double **X,
    double **Y,
    int data_len,
    enum cnet_loss loss_type,
    double learning_rate,
    int epochs
){

    // init temporary helpers
    double *lossarr = malloc(sizeof(double) * nn->out_size);

    for(int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0;

        // TODO: shuffle the training set

        // for each sample in the dataset (SGD - batch size 1)
        for(int sample = 0; sample < data_len; sample++) {

            // pass the sample through the net

            double const *predicted = nn_predict(nn, X[sample]);

            // compute the loss

            double loss = 0;
            cnet_get_loss(loss_type)(
                predicted, 
                Y[sample],
                lossarr,
                nn->out_size
            );

            for (int i = 0; i < nn->out_size; i++) {
                loss += lossarr[i];
            }
            epoch_loss += loss / nn->out_size;

            // backprop step

            nn_backward(
                nn,
                X[sample],
                Y[sample],
                loss_type,
                learning_rate
            );
        }

        // log loss
        printf("[EPOCH %d/%d] - Loss: %lf \n", epoch, epochs, epoch_loss / data_len);
    }
    free(lossarr);
}
