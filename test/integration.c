/**
 * Integration Tests for CNet.
 * */

#include <stdlib.h>
#include <stdio.h>
#include "cnet.h"


#define print(x) printf("%s\n", x); fflush(NULL);


/**
 * Passes random inputs through the net,
 * useful to run with valgrind and to check that any of the changes
 * makes the app crash.
 * */
void test_random_inputs() {
    // set network properties
    int input_size = 6;
    int output_size = 10;
    int hidden_size = 9;
    int n_layers = 4;
    int n_samples = 100;
    int epochs = 1000;
    double lr = 0.001;

    /// initialize neural network with 3 layers
    /// input size 2 and output 1
    cnet *nn = nn_init(
        input_size,
        output_size,
        n_layers
    );
        
    /// add layers
    nn_add(nn, input_size,  hidden_size, act_sigmoid);
    nn_add(nn, hidden_size, hidden_size, act_sigmoid);
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

    // create a file to save output
    FILE *history_file = fopen("test_random_inputs.dat", "w");
    
    // train
    nn_train(
        nn,
        X,
        Y,
        n_samples,
        loss_mse,
        metric_accuracy,
        lr,
        epochs,
        history_file
    );

    // free all objects
    for (int i = 0; i < n_samples; i++) {
        free(X[i]);
        free(Y[i]);
    }
    free(X);
    free(Y);
    nn_free(nn);
}


/**
 * Run all tests. */
int main() {

    // set random seed
    srand((unsigned int)23);

    // random inputs
    printf(
        "*************************************************************\n"
        "                   RUNNING WITH RANDOM INPUT                 \n"
        "*************************************************************\n"
    );

    test_random_inputs();

    printf(
        "*************************************************************\n"
        "                           PASSED                            \n"
        "*************************************************************\n"
    );

}
