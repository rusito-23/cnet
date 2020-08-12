/**
 * Activation Functions for CNet.
 */

#ifndef CNET_ACTIVATION_H
#define CNET_ACTIVATION_H


/* Available Types */

enum cnet_act {
    act_relu,
    act_softmax,
    act_sigmoid
};


/**
 * Activation Function.
 *
 * Performs a non-linear activation over an array of doubles.
 * Stores the result in the given destination array.
 * Both arrays should have the same size.
 *
 * @param const double *src: Source array
 * @param double *dst: Destination array
 * @param int size: Source/Destination size.
 */
typedef void (*cnet_act_fun)(
    double const *src,
    double *dst, 
    int size
);


/**
 * Get activation function.
 * Given an activation type, returns a pointer to the activation function.
 *
 * @param enum cnet_act_type: Activation Type
 */
cnet_act_fun cnet_get_act(enum cnet_act type);


/**
 * Get activation function derivative.
 * Given an activation type,
 * returns a pointer to the activation function derivative.
 *
 * @param enum cnet_act_type: Activation Type
 */
cnet_act_fun cnet_get_act_dx(enum cnet_act type);


#endif /* CNET_ACTIVATION_H */
