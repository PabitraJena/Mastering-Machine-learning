#pragma once

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensor.h"

struct TensorLayer{
    Tensor w;       // (output_size, input_size)
    Tensor b;       // (output_size, 1)
    Tensor z;       // Pre-activation
    Tensor a;       // Activation
    Tensor delta;   // Backpropagation error
    Tensor dw;      // Weight gradients
};
typedef struct TensorLayer TensorLayer;

struct TensorNetwork{
    size_t nlayers;
    size_t *sizes;
    TensorLayer *layers;
};
typedef struct TensorNetwork TensorNetwork;

static void randomWeights(Tensor *w, size_t fan_in){
    for (size_t i = 0; i < w->size; i++){
        w->data[i] = (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f) * sqrtf(2.0f / (float)fan_in);
    }
}

static TensorNetwork createTensorNetwork(size_t nlayers, const size_t *sizes){
    TensorNetwork nn = {0};

    if (nlayers == 0 || sizes == NULL || nlayers > SIZE_MAX / sizeof(TensorLayer) || nlayers > SIZE_MAX / sizeof(size_t)){
        return nn;
    }

    for (size_t i = 0; i < nlayers; i++){
        if (sizes[i] == 0){        
            return nn;
        }
    }
    nn.sizes = (size_t *)malloc((nlayers + 1) * sizeof(size_t));
    nn.layers = (TensorLayer *)calloc(nlayers, sizeof(TensorLayer));
    
    if (nn.sizes == NULL || nn.layers == NULL){
        free(nn.sizes);
        free(nn.layers);
        return (TensorNetwork){0};
    }

    nn.nlayers = nlayers;

    for (size_t i = 0; i <= nlayers; i++){
        nn.sizes[i] = sizes[i];
    }
    
    for (size_t i = 0; i < nlayers; i++){
        TensorLayer *layer = &nn.layers[i];

        size_t w_shape[] = {sizes[i + 1], sizes[i]};
        size_t v_shape[] = {sizes[i + 1], 1};

        layer->w = createTensor(2, w_shape);
        layer->b = createTensor(2, v_shape);
        layer->z = createTensor(2, v_shape);
        layer->a = createTensor(2, v_shape);
        layer->delta = createTensor(2, v_shape);
        layer->dw = createTensor(2, w_shape);

        if (!Tensor_isValid(&layer->w) || !Tensor_isValid(&layer->b) || !Tensor_isValid(&layer->z) || !Tensor_isValid(&layer->a) || !Tensor_isValid(&layer->delta) || !Tensor_isValid(&layer->dw)) {

            for (size_t j = 0; j <= i; j++) {
                TensorLayer *p = &nn.layers[j];
                freeTensor(&p->w);
                freeTensor(&p->b);
                freeTensor(&p->z);
                freeTensor(&p->a);
                freeTensor(&p->delta);
                freeTensor(&p->dw);
            }

            free(nn.layers);
            free(nn.sizes);
            return (TensorNetwork){0};
        }
        randomWeights(&layer->w, sizes[i]);
    }

    return nn;
}

static void FreeTensorNetwork(TensorNetwork *nn){
    if (nn == NULL){
        return;
    }

    for (size_t i = 0; i < nn->nlayers; i++){
        TensorLayer *layer = &nn->layers[i];

        freeTensor(&layer->w);
        freeTensor(&layer->b);
        freeTensor(&layer->z);
        freeTensor(&layer->a);
        freeTensor(&layer->delta);
        freeTensor(&layer->dw);
    }

    free(nn->layers);
    free(nn->sizes);
    *nn = (TensorNetwork){0};
}

static void TensorNetworkForward(TensorNetwork *nn, const Tensor *input){
    if (nn == NULL || !Tensor_isValid(input) || input->ndim != 2 || input->shape[1] != 1 || input->shape[0] != nn->sizes[0]){
        return;
    }

    const Tensor *prev = input;

    for (size_t l = 0; l < nn->nlayers; l++){
        
        TensorLayer *layer = &nn->layers[l];
        size_t in_size = nn->sizes[l];
        size_t out_size = nn->sizes[l + 1];
        
        for (size_t o = 0; o < out_size; o++){
            
            float sum = layer->b.data[o];

            for (size_t i = 0; i < in_size; i++){
                sum += layer->w.data[o * in_size + i] * prev->data[i];
            }
            
            layer->z.data[o] = sum;

            // activae here

            layer->a.data[o] = (l + 1 < nn->nlayers && sum < 0.0f) ? 0.0f : sum;
        }
        prev = &layer->a;
    }
}

static void TensorSoftmax(const Tensor *logits, Tensor * probability){
    if (!Tensor_isValid(logits) || !Tensor_isValid(probability) || logits->size != probability->size){
        return;
    }
    
    float max_value = logits->data[0];
    for (size_t i = 0; i < logits->size; i++){
        if (logits->data[i] > max_value){
            max_value = logits->data[i];
        }
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < logits->size; i++){
        probability->data[i] = expf(logits->data[i] - max_value);
        sum += probability->data[i];
    }
    
    for (size_t i = 0; i < probability->size; i++){
        probability->data[i] /= sum;
    }
}

static size_t TensorArgMax(const Tensor *t){
    size_t max = 0;
    for (size_t i = 0; i < t->size; i++){
        if (t->data[i] > t->data[max]){
            max = i;
        }
    }
    return max;
}

static void TensornetworkBackward(TensorNetwork *nn, const Tensor *intput, const Tensor *probability, size_t target){
    if (nn == NULL || !Tensor_isValid(intput) || !Tensor_isValid(probability) || target >= probability->size){
        return;
    }

    size_t last = nn->nlayers - 1;
    for (size_t i = 0; i < nn->sizes[last + 1]; i++){
        nn->layers[last].delta.data[i] = probability->data[i] - (i == target ? 1.0f : 0.0f);
    }
    
    for (size_t l = last; l-- > 0; ){
        TensorLayer *current = &nn->layers[l];
        TensorLayer *next = &nn->layers[l + 1];

        size_t current_size = nn->sizes[l + 1];
        size_t next_size = nn->sizes[l + 2];

        for (size_t i = 0; i < current_size; i++){
            float error = 0.0f;

            for (size_t j = 0; j < next_size; j++){
                error += next->w.data[j * current_size + i] * next->delta.data[j];
            }
            current->delta.data[i] = current->z.data[i] > 0.0f ? error : 0.0f;
        }
        
    }

    for (size_t l = 0; l < nn->nlayers; l++){
        TensorLayer *layer = &nn->layers[l];

        const Tensor *prev = l == 0? intput : &nn->layers[l - 1].a;

        size_t in_size = nn->sizes[l];
        size_t out_size = nn->sizes[l + 1];

        for (size_t o = 0; o < out_size; o++) {
            for (size_t i = 0; i < in_size; i++) {
                layer->dw.data[o * in_size + i] = layer->delta.data[o] * prev->data[i];
            }
        }
    }
}

static void TensorNetworkUpdate(TensorNetwork *nn, float learning_rate){
    for (size_t l = 0; l < nn->nlayers; l++){
        TensorLayer *layer = &nn->layers[l];

        for (size_t i = 0; i < layer->w.size; i++){
            layer->w.data[i] -= learning_rate * layer->dw.data[i];
        }
        
        for (size_t i = 0; i < layer->b.size; i++){
            layer->b.data[i] -= learning_rate * layer->delta.data[i];
        }
    }
    
}