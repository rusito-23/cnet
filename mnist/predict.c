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
    int val_size = VAL_SIZE;
    mnist_dataset *val_set = mnist_val_set(val_size);

    // initialize the confusion matrix
    int confusion_matrix[OUTPUT_SIZE][OUTPUT_SIZE] = {0};
    int real_count[OUTPUT_SIZE] = {0};

    // init score
    double accuracy = 0;

    // predict over all samples
    for(int i = 0; i < val_size; i++) {
        double *image = val_set->images[i];
        double *target = val_set->labels[i];

        double const *out = nn_predict(
            nn,
            image
        );

        // take the argmax for each sample
        int real = (int)cnet_argmax(target, nn->out_size);
        int pred = (int)cnet_argmax(out, nn->out_size);

        // count
        real_count[real]++;

        // confusion
        confusion_matrix[real][pred]++;
        accuracy += (real == pred);
    }

    // write confusion matrix
    FILE *conf_file = fopen(CONF_FILE_PATH, "w");
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            fprintf(
                conf_file,
                "%lf ",
                ((double)confusion_matrix[i][j] / real_count[i])
            );
        }
        fprintf(conf_file, "\n");
    }

    // log final results
    printf(
        "Final accuracy: %lf over %d samples. \n",
        accuracy / VAL_SIZE,
        VAL_SIZE
    );

    // free all objects
    nn_free(nn);

    return 0;
}
