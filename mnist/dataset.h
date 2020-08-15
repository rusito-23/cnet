/**
 * Read the MNIST Dataset into a struct for CNet usage.
 * Taken from:
 *    Takafumi Hoiruchi. 2018.
 *    https://github.com/takafumihoriuchi/MNIST_for_C
 * Modified a little bit to fit my code style and requirements
 * (needed dynamically allocated arrays and stuff)
*/

#ifndef MNIST_DATASET_H
#define MNIST_DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>


// path for data
#define TRAIN_IM_PATH "./mnist/data/train-images.idx3-ubyte"
#define TRAIN_LABEL_PATH "./mnist/data/train-labels.idx1-ubyte"
#define VAL_IM_PATH "./mnist/data/t10k-images.idx3-ubyte"
#define VAL_LABEL_PATH "./mnist/data/t10k-labels.idx1-ubyte"

// helpful defines
#define INPUT_SIZE 784 // 28*28
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 60000
#define VAL_SIZE 10000
#define INFO_IM_LEN 4
#define INFO_LABEL_LEN 2


// mnist dataset structure

typedef struct mnist_dataset {
    int train_size, val_size;
    double **X_train;
    double **Y_train;
    double **X_val;
    double **Y_val;
} mnist_dataset;


// functions

void mnist_read_data_image_file(
    char *file_path,
    double **data,
    int data_len
){
    // open file
    int file;
    if ((file = open(file_path, O_RDONLY)) == -1) {
        printf("Failed to open file: %s", file_path);
        exit(-1);
    }

    // read info array (we don't need this just now)
    int info_arr[INFO_IM_LEN];
    read(file, info_arr, INFO_IM_LEN * sizeof(int));

    // read image data
    for(int i = 0; i < data_len; i++) {
        unsigned char image_data[INPUT_SIZE];
        read(file, &image_data, INPUT_SIZE * sizeof(unsigned char));
        for(int j = 0; j < INPUT_SIZE; j++) {
            data[i][j] = (double)image_data[j] / 255.0;
        }
    }
}


void mnist_read_data_label_file(
    char *file_path,
    double **data,
    int data_len
){
    // open file
    int file;
    if ((file = open(file_path, O_RDONLY)) == -1) {
        printf("Failed to open file: %s", file_path);
        exit(-1);
    }

    // read info array (we don't need this just now)
    int info_arr[INFO_LABEL_LEN];
    read(file, info_arr, INFO_LABEL_LEN * sizeof(int));

    // read data
    for(int i = 0; i < data_len; i++) {
        unsigned char d;
        read(file, &d, sizeof(unsigned char));
        for(int j = 0; j < OUTPUT_SIZE; j++) {
            data[i][j] = d == ((unsigned char)j) ? 1 : 0;
        }
    }
}


mnist_dataset *mnist_init(
    int train_size,
    int val_size
){
    // alloc dataset struct
    mnist_dataset *ds = malloc(sizeof(mnist_dataset));

    // init basic info
    assert(train_size <= TRAIN_SIZE);
    assert(val_size <= VAL_SIZE);
    ds->train_size = train_size;
    ds->val_size = val_size;

    // alloc data arrays
    ds->X_train = malloc(sizeof(double *)*train_size);
    ds->Y_train = malloc(sizeof(double *)*train_size);
    for(int i = 0; i < train_size; i++) {
        ds->X_train[i] = malloc(sizeof(double)*INPUT_SIZE);
        ds->Y_train[i] = malloc(sizeof(double)*OUTPUT_SIZE);
    }

    ds->X_val = malloc(sizeof(double *)*val_size);
    ds->Y_val = malloc(sizeof(double *)*val_size);
    for(int i = 0; i < val_size; i++) {
        ds->X_val[i] = malloc(sizeof(double)*INPUT_SIZE);
        ds->Y_val[i] = malloc(sizeof(double)*OUTPUT_SIZE);
    }

    // read train images
    mnist_read_data_image_file(
        TRAIN_IM_PATH,
        ds->X_train,
        ds->train_size
    );

    // read val images
    mnist_read_data_image_file(
        VAL_IM_PATH,
        ds->X_val,
        ds->val_size
    );

    // read train labels
    mnist_read_data_label_file(
        TRAIN_LABEL_PATH,
        ds->Y_train,
        ds->train_size
    );

    // read val labels
    mnist_read_data_label_file(
        VAL_LABEL_PATH,
        ds->Y_val,
        ds->val_size
    );

    return ds;
}


void mnist_free(
    mnist_dataset *ds
){
    // free data arrays
    for(int i = 0; i < ds->train_size; i++) {
        free(ds->X_train[i]);
        free(ds->Y_train[i]);
    }
    free(ds->X_train);
    free(ds->Y_train);

    for(int i = 0; i < ds->val_size; i++) {
        free(ds->X_val[i]);
        free(ds->Y_val[i]);
    }
    free(ds->X_val);
    free(ds->Y_val);

    // free struct
    free(ds);
}


#endif /* MNIST_DATASET_H */
