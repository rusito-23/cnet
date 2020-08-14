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
    // network properties
    int input_size = 2;
    int output_size = 1;
    int hidden_size = 2;
    int n_layers = 2;
    int n_samples = 4;
    int epochs = 10000;
    double lr = 0.1;

    // create training samples

    double **X = malloc(sizeof(double*)*n_samples);
    double **Y = malloc(sizeof(double*)*n_samples);
    for (int i = 0; i < n_samples; i++) {
        X[i] = malloc(sizeof(double)*input_size);
        Y[i] = malloc(sizeof(double)*output_size);
    } 

    X[0][0] = 0.0f;
    X[0][1] = 0.0f;
    X[1][0] = 1.0f;
    X[1][1] = 0.0f;
    X[2][0] = 0.0f;
    X[2][1] = 1.0f;
    X[3][0] = 1.0f;
    X[3][1] = 1.0f;

    Y[0][0] = 0.0f;
    Y[1][0] = 1.0f;
    Y[2][0] = 1.0f;
    Y[0][0] = 0.0f;

    /// initialize neural network
    cnet *nn = nn_init(
        input_size,
        output_size,
        n_layers
    );
        
    /// add layers
    nn_add(nn, input_size,  hidden_size, act_sigmoid);
    nn_add(nn, hidden_size, output_size, act_sigmoid);

    // create a file to save output
    FILE *history_file = fopen("test/test_random_inputs.dat", "w");
    
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
