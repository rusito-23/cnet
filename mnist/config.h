/**
 * MNIST CNet Train Configuration
 * */

#ifndef MNIST_CFG_H
#define MNIST_CFG_H

/* OUTPUT PATHS */

#define HISTORY_FILE_PATH       "./mnist/out/history.dat"
#define CONF_FILE_PATH          "./mnist/out/conf_matrix.dat"
#define REPORT_FILE_PATH        "./mnist/out/report.txt"
#define MODEL_FILE_PATH         "./mnist/out/model.cnet"


/* DATASET PATHS */

#define TRAIN_IM_PATH           "./mnist/data/train-images.idx3-ubyte"
#define TRAIN_LABEL_PATH        "./mnist/data/train-labels.idx1-ubyte"
#define VAL_IM_PATH             "./mnist/data/t10k-images.idx3-ubyte"
#define VAL_LABEL_PATH          "./mnist/data/t10k-labels.idx1-ubyte"


/* DATA INFO */

#define INPUT_SIZE      784     // 28px*28px
#define OUTPUT_SIZE     10      // 10 digits
#define TRAIN_SIZE      60000   // training samples
#define VAL_SIZE        10000   // validation samples
#define INFO_IM_LEN     4       // number of information bytes in train file
#define INFO_LABEL_LEN  2       // number of information bytes in val file


#endif /* MNIST_CFG_H */
