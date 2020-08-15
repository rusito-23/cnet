/**
 * Integration Tests for CNet.
 * */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cnet.h"


#define print(x) printf("%s\n", x); fflush(NULL);


/**
 * Passes random inputs through the net,
 * useful to run with valgrind and to check that any of the changes
 * makes the app crash.
 * */
void test_random_inputs() {
    // sizes
    int input_size = 4;
    int output_size = 1;
    int hidden_size = 8;

    // layers
    int n_layers = 4;

    // samples
    int train_size = 100;
    int val_size = 30;

    // hyperparameters
    int epochs = 500;
    double lr = 0.001;

    // create training samples

    double **X_train = malloc(sizeof(double*)*train_size);
    double **Y_train = malloc(sizeof(double*)*train_size);
    for (int i = 0; i < train_size; i++) {
        X_train[i] = malloc(sizeof(double)*input_size);
        for(int j = 0; j < input_size; j++) {
            X_train[i][j] = ((double)rand())/((double)RAND_MAX);
        }

        Y_train[i] = malloc(sizeof(double)*output_size);
        for(int j = 0; j < output_size; j++) {
            Y_train[i][j] = round((double)rand())/((double)RAND_MAX);
        }
    } 

    // create validation samples

    double **X_val = malloc(sizeof(double*)*val_size);
    double **Y_val = malloc(sizeof(double*)*val_size);
    for (int i = 0; i < val_size; i++) {
        X_val[i] = malloc(sizeof(double)*input_size);
        for(int j = 0; j < input_size; j++) {
            X_val[i][j] = ((double)rand())/((double)RAND_MAX);
        }

        Y_val[i] = malloc(sizeof(double)*output_size);
        for(int j = 0; j < output_size; j++) {
            Y_val[i][j] = round((double)rand())/((double)RAND_MAX);
        }
    } 

    /// initialize neural network
    cnet *nn = nn_init(
        input_size,
        output_size,
        n_layers
    );
        
    /// add layers
    nn_add(nn, input_size,  hidden_size, act_sigmoid);
    nn_add(nn, hidden_size,  hidden_size, act_sigmoid);
    nn_add(nn, hidden_size,  hidden_size, act_sigmoid);
    nn_add(nn, hidden_size, output_size, act_sigmoid);

    // create a file to save output
    FILE *history_file = fopen("test/test_random_inputs.dat", "w");
    
    // train
    nn_train(
        nn,
        X_train,
        Y_train,
        X_val,
        Y_val,
        train_size,
        val_size,
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
