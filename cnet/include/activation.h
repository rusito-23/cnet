/*****************************************************************************
 *                               ACTIVATION
 * Available activation functions and some helpers.
 ****************************************************************************/

#ifndef CNET_ACTIVATION_H
#define CNET_ACTIVATION_H


enum cnet_act_type {
    relu_act,                   // Rectified Linear Units
    sigmoid_act,                // Sigmoid
    softmax_act                 // Softmax
};


/**
 * Activation Function
 *
 * These functions will be in charge of performing in-place activation
 * for the sum of weights * inputs in a layer.
 *
 * @param double *: Sum of weights * inputs + Bias
 * @param int: Size
 */
typedef void cnet_act_func(double *, int); 


/**
 * Activation Function Delta Computation.
 *
 * This funcions will be in charge of computing the layer's delta.
 * It will take as parameters the pointer to the array containing 
 * the delta computed using the derivative of the cost function.
 *
 * @param double*: Output
 * @param double*: Delta
 * @param int: Size
 */
typedef void cnet_act_func_delta(
    double *,
    double *,
    int
);


/**
 * Get activation function by type 
 *
 * Returns a pointer to the activation function implementation,
 * for the given type.
 *
 * @param enum cnet_act_type: Activation type
 * @return cnet_act_func*
 */
cnet_act_func *cnet_get_act(enum cnet_act_type type);


/**
 * Get activation delta computation function by type
 *
 * Returns a pointer to the activation function
 * delta computation implementation.
 *
 * @param enum cnet_act_type: Activation type
 * @return cnet_act_func_delta*
 */
cnet_act_func_delta *cnet_get_act_delta(enum cnet_act_type type);


#endif /* CNET_ACTIVATION_H */
