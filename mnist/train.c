/**
 * Train CNet on the MNist Dataset.
 */

#include <stdio.h>
#include "cnet.h"
#include "dataset.h"
#include "config.h"


int main() {
    // hyperparameters
    double lr = 0.001;
    double epochs = 50;

    // define dataset variables
    int train_size = TRAIN_SIZE;
    int val_size = VAL_SIZE;

    // define input / output
    int output_size = OUTPUT_SIZE;
    int input_size = INPUT_SIZE;

    // load mnist dataset
    mnist_dataset *train_set = mnist_train_set(train_size);
    mnist_dataset *val_set = mnist_val_set(val_size);

    // init model
    int n_layers = 3;
    cnet *nn = nn_init(
        input_size,
        output_size,
        n_layers
    );

    /// add layers
    nn_add(nn,  input_size,     256,            relu_act);
    nn_add(nn,  256,            128,            relu_act);
    nn_add(nn,  128,            output_size,    softmax_act);

    // create a file to save output
    FILE *history_file = fopen(HISTORY_FILE_PATH, "w");

    // train
    nn_train(
        nn,
        train_set->images,
        train_set->labels,
        val_set->images,
        val_set->labels,
        train_set->size,
        val_set->size,
        cross_entropy_loss,
        metric_accuracy_argmax,
        lr,
        epochs,
        history_file
    );

    // save model
    FILE *model_file = fopen(MODEL_FILE_PATH, "w");
    nn_save(
        nn,
        model_file
    );

    // free all objects
    nn_free(nn);
    mnist_free(train_set);
    mnist_free(val_set);

    return 0;
}
