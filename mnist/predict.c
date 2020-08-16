/**
 * Use the saved CNet model to predict over the MNIST dataset. 
 **/

#include <stdio.h>
#include "cnet.h"
#include "helpers.h"
#include "dataset.h"
#include "config.h"


int main() {

    // load model from file
    FILE *model_file = fopen(MODEL_FILE_PATH, "r");
    cnet *nn = nn_load(model_file);

    // load test set
    int val_size = 50;
    mnist_dataset *val_set = mnist_val_set(val_size);

    // predict over all samples
    for(int i = 0; i < val_size; i++) {
        double *image = val_set->images[i];
        double *target = val_set->labels[i];

        double const *out = nn_predict(
            nn,
            image
        );

        double real = cnet_argmax(target, nn->out_size);
        double pred = cnet_argmax(out, nn->out_size);

        printf("Sample [%d] - expected: %lf - predicted: %lf \n", i, real, pred);
    }

    // free all objects
    nn_free(nn);

    return 0;
}
