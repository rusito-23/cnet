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

    // report utils
    int support[OUTPUT_SIZE] = {0};     // label count
    double accuracy = 0;                // percentage (TP + TN)/(ALL)
    int tp[OUTPUT_SIZE] = {0};          // label true positives
    int fp[OUTPUT_SIZE] = {0};          // label false positives
    int fn[OUTPUT_SIZE] = {0};          // label false negatives

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

        // update classification report 
        support[real]++;
        accuracy += (real == pred);
        tp[real] += (real == pred);
        fn[real] += !(real == pred);
        fp[pred] += !(real == pred);

        // update confusion
        confusion_matrix[real][pred]++;
    }

    // write confusion matrix
    FILE *conf_file = fopen(CONF_FILE_PATH, "w");
    for(int i = 0; i < OUTPUT_SIZE; i++) {
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            fprintf(
                conf_file,
                "%lf ",
                ((double)confusion_matrix[i][j] / support[i])
            );
        }
        fprintf(conf_file, "\n");
    }

    // log classification report
    FILE *report_file = fopen(REPORT_FILE_PATH, "w");
    fprintf(report_file,
        "CLASSIFICATION REPORT \n\n"
        "         precision  recall     f1-score   support \n"
    );

    for(int i = 0; i < OUTPUT_SIZE; i++) {
        double precision = (double)tp[i] / (tp[i] + fp[i]);
        double recall = (double)tp[i] / (tp[i] + fn[i]);
        double f1 = 2*(recall * precision) / (recall + precision);
        fprintf(
            report_file,
            "Digit %d  %lf  %lf  %lf  %d \n",
            i,
            precision,
            recall,
            f1,
            support[i]
        );
    }

    fprintf(
        report_file,
        "\n\nFinal Accuracy: %lf - Samples: %d",
        accuracy / val_size,
        val_size
    );

    // free all objects
    nn_free(nn);
    mnist_free(val_set);

    return 0;
}
