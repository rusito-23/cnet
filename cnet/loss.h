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
 * @param int size: Source/Destination size.
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


#endif /* CNET_LOSS_H */
