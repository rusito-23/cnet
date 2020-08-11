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
        free(layer->output);
        free(layer->delta);
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
    layer->output = malloc(sizeof(double)*layer->out_size);
    layer->delta = malloc(sizeof(double)*layer->out_size);
    layer->dC_dA = malloc(sizeof(double)*layer->out_size);
    layer->dA_dZ = malloc(sizeof(double)*layer->out_size);

    // initialize weights
    for(int i = 0; i < layer->out_size; i++) {
        layer->weights[i] = malloc(sizeof(double)*layer->in_size);

        // randomize weights between 0 and 1
        for(int j = 0; j < layer->in_size; j++) {
            layer->weights[i][j] = (double)(rand()/RAND_MAX);
        }
    }

    // add layer to the net
    assert(nn->last_layer < nn->n_layers);
    nn->layers[nn->last_layer++] = layer;
}


/**
 * Network Forward Pass. */
const double *nn_predict(
    cnet const *nn,
    double const *X
){
    double const *in = X;

    // pass through every layer in the net
    for(int i = 0; i < nn->n_layers; i++) {
        struct clayer *layer = nn->layers[i];

        // pass through every neuron in the net
        for(int k = 0; k < layer->out_size; k++) {

            // compute delta for neuron
            double delta = 0;
            for(int j = 0; j < layer->in_size; j++) {
                delta += layer->weights[k][j] * in[j];
            }
            
            layer->delta[k] = delta;
        }


        // activate the layer output
        cnet_act_fun act = cnet_get_act(layer->act_type);
        act(layer->delta, layer->output, layer->out_size);

        // set input for next layer
        in = layer->output;
    }

    // return the output for the last layer
    return nn->layers[nn->n_layers - 1]->output;
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
            epoch_loss = loss / nn->out_size;


            // backpropagation

            for(int l = nn->n_layers; l-->0;) {
                struct clayer* layer = nn->layers[l];
                struct clayer* next = l < nn->n_layers ? nn->layers[l + 1] : NULL;
                struct clayer* previous = l > 0 ? nn->layers[l - 1] : NULL;

                // activation derivative over the layer's delta
                cnet_get_act_dx(layer->act_type)(
                    layer->delta,
                    layer->dA_dZ,
                    layer->out_size
                );

                // delta derivative over the weights
                // this corresponds with the previous layer's output
                double *dZ_dW = previous == NULL ? X[sample] : previous->output;

                // cost derivative over the current output
                if (next == NULL) {
                    cnet_loss_fun loss_dx = cnet_get_loss_dx(loss_type);
                    loss_dx(
                        layer->output,
                        X[sample],
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

                // update the weights
                for(int i = 0; i < layer->out_size; i++) {
                    for(int j = 0; j < layer->in_size; j++) {
                        double dC_dW = dZ_dW[j] * layer->dA_dZ[i] * layer->dC_dA[i];
                        layer->weights[i][j] -= learning_rate * dC_dW;
                    }
                }
            }
        }

        // log loss
        printf("[EPOCH %d/%d] - Loss: %lf \n", epoch, epochs, epoch_loss / data_len);
    }
    free(lossarr);
}
