#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

struct Tensor{
    float *data;
    size_t *shape;
    size_t ndim;
    size_t size;
};
typedef struct Tensor Tensor;

static Tensor Tensor_isInvalid(){
    return (Tensor){0};
}

static bool Tensor_isValid(const Tensor *t){
    if (t == NULL || t->data == NULL || t->shape == NULL || t->ndim == 0 || t->shape == 0){
        return false;
    }

    size_t expected_size = 1;

    for (size_t i = 0; i < t->ndim; i++){
        if (t->shape[i] == 0 || expected_size > SIZE_MAX / t->shape[i]){
            return false;
        }
        expected_size *= t->shape[i];
    }
    return (expected_size == t->size);
}

static bool Tensor_sameShape(const Tensor *a, const Tensor *b){
    if (!Tensor_isValid(a) || !Tensor_isValid(b) || a->ndim != b->ndim || a->size != b->size){
        return false;
    }

    for (size_t i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            return false;
        }
    }
    return true;
}

static size_t Tensor_Offset(const Tensor *t, const size_t *indices){
    if(!Tensor_isValid(t) || indices == NULL){
        return SIZE_MAX;
    }

    size_t offset = 0;
    for (size_t i = 0; i < t->ndim; i++){
        if (indices[i] >= t->shape[i]){
            return SIZE_MAX;
        }
        offset = offset * t->shape[i] + indices[i];
    }
    return offset;
}

void freeTensor(Tensor *t){
    if (t == NULL){
        return;
    }
    free(t->data);
    free(t->shape);
    *t = (Tensor){0};
}

Tensor createTensor(const size_t ndim, const size_t *shape){
    if (ndim == 0 || shape == NULL || ndim >= SIZE_MAX / sizeof(size_t)){
        return Tensor_isInvalid();
    }

    Tensor t = {0};

    t.shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (t.shape == NULL){
        return Tensor_isInvalid();
    }
    t.ndim = ndim;
    t.size = 1;

    for (size_t i = 0; i < ndim; i++){
        if (shape[i] == 0 || t.size > SIZE_MAX / shape[i]){
            freeTensor(&t);
            return Tensor_isInvalid();
        }

        t.shape[i] = shape[i];
        t.size *= shape[i];
    }
    
    if (t.size > SIZE_MAX / sizeof(float)){
        freeTensor(&t);
        return Tensor_isInvalid();
    }
    
    t.data = (float *)calloc(t.size, sizeof(float));

    if (t.data == NULL){
        freeTensor(&t);
        return Tensor_isInvalid();
    }

    return t;
}

Tensor cloneTensor(const Tensor *t){
    if (!Tensor_isValid(t)){
        return Tensor_isInvalid();
    }

    Tensor out = createTensor(t->ndim, t->shape);

    if (!Tensor_isValid(&out)){
        return Tensor_isInvalid();
    }

    for (size_t i = 0; i < t->size; i++){
        out.data[i] = t->data[i];
    }
    return out;
}

void fillTensor(const Tensor *t, float value){
    if (!Tensor_isValid(t)){
        return;
    }

    for (size_t i = 0; i < t->size; i++){
        t->data[i] = value;
    }
}

bool setTensor(const Tensor *t, const size_t *indices, float value){
    size_t offset = Tensor_Offset(t, indices);

    if (offset == SIZE_MAX){
        return false;
    }
    t->data[offset] = value;
    return true;
}

bool getTensor(const Tensor *t, const size_t *indices, float *out){
    if(out == NULL){
        return false;
    }

    size_t offset = Tensor_Offset(t, indices);

    if(offset == SIZE_MAX){
        return false;
    }

    *out = t->data[offset];
    return true;
}

void printTensor(const Tensor *t){
    if(!Tensor_isValid(t)){
        printf("Invalid tensor\n");
        return;
    }

    printf("Shape: (");

    for(size_t i = 0; i < t->ndim; i++){
        printf("%zu%s", t->shape[i], i + 1 < t->ndim ? ", " : "");
    }

    printf(")\nData: [");

    for(size_t i = 0; i < t->size; i++){
        printf("%.4f%s", t->data[i], i + 1 < t->size ? ", " : "");
    }

    printf("]\n");
}

Tensor addTensor(const Tensor *a, const Tensor *b){
    if (!Tensor_sameShape(a, b)){
        return Tensor_isInvalid();
    }

    Tensor out = createTensor(a->ndim, a->shape);
    if (!Tensor_isValid(&out)){
        return Tensor_isInvalid();
    }

    for (size_t i = 0; i < a->size; i++){
        out.data[i] = a->data[i] + b->data[i];
    }
    return out;
}

Tensor subTensor(const Tensor *a, const Tensor *b){
    if (!Tensor_sameShape(a, b)){
        return Tensor_isInvalid();
    }

    Tensor out = createTensor(a->ndim, a->shape);
    if (!Tensor_isValid(&out)){
        return Tensor_isInvalid();
    }

    for (size_t i = 0; i < a->size; i++){
        out.data[i] = a->data[i] - b->data[i];
    }
    return out;
}

Tensor mulTensor(const Tensor *a, const Tensor *b){
    if (!Tensor_sameShape(a, b)){
        return Tensor_isInvalid();
    }

    Tensor out = createTensor(a->ndim, a->shape);
    if (!Tensor_isValid(&out)){
        return Tensor_isInvalid();
    }

    for (size_t i = 0; i < a->size; i++){
        out.data[i] = a->data[i] * b->data[i];
    }
    return out;
}

Tensor scaleTensor(const Tensor *t, float scaler){
    if (!Tensor_isValid(t)){
        return Tensor_isInvalid();
    }

    Tensor out = createTensor(t->ndim, t->shape);
    if (!Tensor_isValid(&out)){
        return Tensor_isInvalid();
    }

    for (size_t i = 0; i < t->size; i++){
        out.data[i] = t->data[i] * scaler;
    }
    return out;
}

Tensor transposeTensor(const Tensor *t){
    if (!Tensor_isValid(t) || t->ndim != 2){
        return Tensor_isInvalid();
    }

    size_t rows = t->shape[0];
    size_t cols = t->shape[1];

    size_t shape[] = {rows, cols};
    Tensor out = createTensor(2, shape);
    if (!Tensor_isValid(&out)){
        return Tensor_isInvalid();
    }
    
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            out.data[j * rows + i] = t->data[i * cols + j];
        }
    }
    return out;
}

Tensor matmulTensor(const Tensor *a, const Tensor *b){
    if (!Tensor_isValid(a) || !Tensor_isValid(b) || a->ndim != 2 || b->ndim != 2){
        return Tensor_isInvalid();
    }

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    size_t n = b->shape[1];

    if (k != b->shape[0]){
        return Tensor_isInvalid();
    }
    
    size_t shape[] = {m, n};
    Tensor out = createTensor(2, shape);
    if(!Tensor_isValid(&out)){
        return Tensor_isInvalid();
    }

    for (size_t i = 0; i < m; i++){
        for (size_t j = 0; j < k; j++){
            float av = a->data[i * k + j];
            for (size_t l = 0;l < n; l++){
                out.data[i * n + l] += av * b->data[j * n + l];
            }
            
        }
    }
    return out;
}

float sumTensor(const Tensor *t){
    if(!Tensor_isValid(t)){
        return NAN;
    }

    float sum = 0.0f;
    for(size_t i = 0; i < t->size; i++){
        sum += t->data[i];
    }

    return sum;
}