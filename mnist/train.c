/**
 * Train CNet on the MNist Dataset.
 */

#include <stdio.h>
#include "cnet.h"
#include "dataset.h"
#include "config.h"


int main() {
    // hyperparameters
    double lr = 0.01;
    double epochs = 100;

    // define dataset variables
    int train_size = 600;
    int val_size = 60;

    // define input / output
    int output_size = OUTPUT_SIZE;
    int input_size = INPUT_SIZE;

    // load mnist dataset
    mnist_dataset *ds = mnist_init(train_size, val_size);

    // init model
    int n_layers = 5;
    cnet *nn = nn_init(
        input_size,
        output_size,
        n_layers
    );
        
    /// add layers
    nn_add(nn, input_size, 128,         act_sigmoid);
    nn_add(nn, 128,        128,         act_sigmoid);
    nn_add(nn, 128,        64,          act_sigmoid);
    nn_add(nn, 64,         32,          act_sigmoid);
    nn_add(nn, 32,         output_size, act_softmax);

    // create a file to save output
    FILE *history_file = fopen(HISTORY_FILE_PATH, "w");

    // train
    nn_train(
        nn,
        ds->X_train,
        ds->Y_train,
        ds->X_val,
        ds->Y_val,
        ds->train_size,
        ds->val_size,
        loss_mse,
        metric_accuracy_argmax,
        lr,
        epochs,
        history_file
    );

    // free all objects
    nn_free(nn);
    mnist_free(ds);

    return 0;
}
