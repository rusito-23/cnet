/**
 * CNet file management.
 *
 * Load & Save the CNet model from a given file.
 */

#include <stdlib.h>
#include "cnet.h"


/**
 * Save CNet into File. */
void nn_save(
    cnet const* nn,
    FILE *out
){
    // save basic network info
    fprintf(out, "%d %d %d \n", nn->in_size, nn->out_size, nn->n_layers);

    // save every layer info
    for(int i = 0; i < nn->n_layers; i++) {
        clayer *layer = nn->layers[i];
        fprintf(
            out,
            "%d %d %d \n",
            layer->in_size,
            layer->out_size,
            layer->act_type
        );

        // save every layer biases
        for(int j = 0; j < layer->out_size; j++) {
            fprintf(out, " %.20e", layer->bias[j]);
        }
        fprintf(out, "\n");

        // save every layer weights
        for(int j = 0; j < layer->out_size; j++) {
            for(int k = 0; k < layer->in_size; k++) {
                fprintf(out, " %.20e", layer->weights[j][k]);
            }
            fprintf(out, "\n");
        }
    }
}


/**
 * Load CNet from File. */
cnet *nn_load(
    FILE *in
){
    // load basic network info
    int in_size, out_size, n_layers;
    fscanf(in, "%d %d %d \n", &in_size, &out_size, &n_layers);

    // init cnet
    cnet *nn = nn_init(in_size, out_size, n_layers);

    for(int i = 0; i < nn->n_layers; i++) {
        // load layer info
        int in_size, out_size;
        enum cnet_act act_type;
        fscanf(
            in,
            "%d %d %d \n",
            &in_size,
            &out_size,
            &act_type
        );

        // create layer
        nn_add(
            nn,
            in_size,
            out_size,
            act_type
        );
        clayer *layer = nn->layers[i];

        // load biases
        for(int j = 0; j < layer->out_size; j++) {
            fscanf(in, " %le", &(layer->bias[j]));
        }
        fscanf(in, "\n");

        // load weights
        for(int j = 0; j < layer->out_size; j++) {
            for(int k = 0; k < layer->in_size; k++) {
                fscanf(in, " %le", &(layer->weights[j][k]));
            }
            fscanf(in, "\n");
        }
    }

    return nn;
}
