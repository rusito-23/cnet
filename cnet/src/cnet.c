/**
 * CNet Implementation
 *
 * Implements an Artificial Neural Network and several methods to work with.
 */

#include <assert.h>
#include <stdlib.h>
#include "cnet.h"
#include "helpers.h"


/**
 * Create CNet. */
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
 * Free CNet. */
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
 * Add a Layer to the CNet. */
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
 * CNet Forward Pass
 *
 * Simply passes a given input (with expected size) through the net.
 * Does not return the result pointer, this should be accessed through
 * the nn->layers[last_layer - 1]->result;
 *
 * @param cnet const *nn: CNet
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
 * CNet Backward Pass
 *
 * Performs a single backpropagation step, using SGD, hence
 * it only takes one train sample.
 *
 * @param cnet const *nn: CNet
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
            layer->bias[i] += learning_rate * dC_dB;

            for(int j = 0; j < layer->in_size; j++) {
                double dC_dW = dZ_dW[j] * dC_dB;
                layer->weights[i][j] += learning_rate * dC_dW;
            }
        }
    }
}


/**
 * CNet Prediction. */
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
 * CNet Train Algorithm */
void nn_train(
    cnet const *nn,
    double **X_train,
    double **Y_train,
    double **X_val,
    double **Y_val,
    int train_size,
    int val_size,
    enum cnet_loss loss_type,
    enum cnet_metric metric_type,
    double learning_rate,
    int epochs,
    FILE *history_file
){
    // check nn initialization
    assert(nn->last_layer == nn->n_layers);
    assert(nn->layers[nn->last_layer - 1]->out_size == nn->out_size);
    assert(nn->layers[0]->in_size == nn->in_size);

    // init history file
    fprintf(history_file, "train_loss val_loss train_metric val_metric\n");

    // init temporary helper arrays
    int *idx_arr = cnet_idx(train_size);

    for(int epoch = 0; epoch < epochs; epoch++) {
        double train_loss = 0, val_loss = 0; 
        double train_metric = 0, val_metric = 0;

        // shuffle the training set
        cnet_shuffle(idx_arr, train_size);

        // epoch training
        // SGD - batch size 1
        for(int s = 0; s < train_size; s++) {
            int sample = idx_arr[s];

            // pass the training sample through the net
            double const *train_pred = nn_predict(nn, X_train[sample]);

            // compute training loss and metric
            train_loss += cnet_loss_mean(
                loss_type,
                train_pred,
                Y_train[sample],
                nn->out_size
            );
            train_metric += cnet_get_metric(metric_type)(
                train_pred, 
                Y_train[sample],
                nn->out_size
            );

            // backprop step
            nn_backward(
                nn,
                X_train[sample],
                Y_train[sample],
                loss_type,
                learning_rate
            );
        }

        // epoch validation
        for(int s = 0; s < val_size; s++) {
            // pass the training sample through the net
            double const *val_pred = nn_predict(nn, X_val[s]);

            val_loss += cnet_loss_mean(
                loss_type,
                val_pred,
                Y_val[s],
                nn->out_size
            );
            val_metric += cnet_get_metric(metric_type)(
                val_pred, 
                Y_val[s],
                nn->out_size
            );
        } 

        // log metrics
        printf(
            "[EPOCH %d/%d] "
            "- Train Loss: %lf "
            "- Train %s: %lf "
            "- Val Loss: %lf "
            "- Val %s: %lf \n",
            epoch,
            epochs,
            train_loss / train_size,
            cnet_get_metric_name(metric_type),
            train_metric / train_size,
            val_loss / val_size,
            cnet_get_metric_name(metric_type),
            val_metric / val_size
        );

        // save history
        fprintf(
            history_file,
            "%.20e %.20e %.20e %.20e\n",
            train_loss / train_size,
            val_loss / val_size,
            train_metric / train_size,
            val_metric / val_size
        );
    }
    free(idx_arr);
}
