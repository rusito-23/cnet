/**
 *
 * Test the Neural Network Behavior.
 *
 * Initialize random inputs / expected outputs, and check the loss results.
 */

#include <stdlib.h>
#include <stdio.h>
#include "../cnet/cnet.h"


#define print(x) printf("%s\n", x); fflush(NULL);


// set network properties
int input_size = 2;
int output_size = 1;
int hidden_size = 3;
int n_layers = 3;
int n_samples = 500;
int epochs = 10000;
double lr = 0.1;

int main() {
    // set random seed
    srand((unsigned int)23);

    /// initialize neural network with 3 layers
    /// input size 2 and output 1
    cnet *nn = nn_init(
        input_size,
        output_size,
        n_layers
    );
        
    /// add layers
    nn_add(nn, input_size, hidden_size, act_sigmoid);
    nn_add(nn, hidden_size, hidden_size, act_sigmoid);
    nn_add(nn, hidden_size, output_size, act_sigmoid);

    /// generate random samples
    double** X = (double **)malloc(sizeof(double *)*n_samples);
    double** Y = (double **)malloc(sizeof(double *)*n_samples);

    for (int i = 0; i < n_samples; i++) {

        X[i] = (double*)malloc(sizeof(double)*input_size);
        Y[i] = (double*)malloc(sizeof(double)*output_size);

        for (int j = 0; j < input_size; j++) {
            X[i][j] = (double)rand()/RAND_MAX*2.0-1.0;
        }

        for (int j = 0; j < output_size; j++) {
            Y[i][j] = (double)rand()/RAND_MAX*2.0-1.0;
        }
    }
    
    // train
    nn_train(
        nn,
        X,
        Y,
        n_samples,
        loss_mse,
        lr,
        epochs
    );

    // free all objects
    for (int i = 0; i < n_samples; i++) {
        free(X[i]);
        free(Y[i]);
    }
    free(X);
    free(Y);
    nn_free(nn);

    return 0;
}
