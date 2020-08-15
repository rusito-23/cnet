/**
 * Loss Functions for CNet.
 */

#ifndef CNET_LOSS_H
#define CNET_LOSS_H


/* Available Types */

enum cnet_loss {
    loss_mse
};


/**
 * Loss function.
 *
 * Searchs the cost of a predicted output over the expected values.
 * Stores the result in the given destination array.
 * Both arrays should have the same size.
 *
 * @param const double *pred: Predictions array
 * @param const double *real: Expected array
 * @param double *dst: Destination array
 * @param int size: Predictions/Expected/Destination size.
 */
typedef void (*cnet_loss_fun)(
    double const *pred,
    double const *real,
    double *dst,
    int size
);


/**
 * Get loss function
 * Given a loss type, returns a pointer to the loss function.
 *
 * @param enum cnet_loss_type: Loss Type
 */
cnet_loss_fun cnet_get_loss(enum cnet_loss type);


/**
 * Get loss function derivative.
 * Given a loss type, returns a pointer to the loss function derivative.
 *
 * @param enum cnet_loss_type: Loss Type
 */
cnet_loss_fun cnet_get_loss_dx(enum cnet_loss type);


/**
 * Calculate loss mean for given arrays.
 * Uses the loss for each of the elements in the given arrays
 * (should have the same size), and returns the mean loss between these
 * arrays.
 *
 * @param enum cnet_loss_type type: Loss Type
 * @param double *pred: Predicted values
 * @param double *real: Expected values
 * @param int size: Size of pred and real
 */
double cnet_loss_mean(
    enum cnet_loss type,
    double const *pred,
    double const *real,
    int size
);


#endif /* CNET_LOSS_H */
