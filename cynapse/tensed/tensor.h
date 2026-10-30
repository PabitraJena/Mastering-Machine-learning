#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#include "../ansi_color.h"

typedef float (*TensorOp)(float, float);

float Tadd(float a, float b){
    return a + b;
}

float Tsubtract(float a, float b){
    return a - b;
}

float Tmultiply(float a, float b){
    return a * b;
}

struct Tensor{
    float *data;
    size_t *shape;
    size_t ndim;
    size_t size;
};
typedef struct Tensor Tensor;

size_t TensorNumEl(size_t ndim, const size_t *shape){
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++){
        if (shape[i] == 0 || size > SIZE_MAX / shape[i]){
            return 0;
        }
        size *= shape[i];
    }
    return size;
}

Tensor *TensorCreate(size_t ndim, const size_t *shape){
    if (!shape || ndim == 0){
        return NULL;
    }
    size_t size = TensorNumEl(ndim, shape);
    if (size == 0 || size > SIZE_MAX / sizeof(float)){
        return NULL;
    }

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t){
        return NULL;
    }
    t->data = (float *)calloc(size, sizeof(float));
    t->shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!t->data || !t->shape){
        free(t->data);
        free(t->shape);
        free(t);
        return NULL;
    }

    memcpy(t->shape, shape, ndim * sizeof(size_t));
    t->ndim = ndim;
    t->size = size;

    return t;
}

void TensorFree(Tensor *t){
    if (!t){
        return;
    }
    free(t->data);
    free(t->shape);
    free(t);
}

void TensorFill(Tensor *t, float value){
    if (!t){
        return;
    }
    for (size_t i = 0; i < t->size; i++){
        t->data[i] = value;
    }
}

Tensor *TensorZeros(size_t ndim, const size_t *shape){
    return (TensorCreate(ndim, shape));
}

Tensor *TensorOnes(size_t ndim, const size_t *shape){
    Tensor *t = TensorCreate(ndim, shape);
    if (!t){
        return NULL;
    }
    TensorFill(t, 1.0f);
    return t;
}

Tensor *TensorRand(size_t ndim, const size_t * shape){
    Tensor *t = TensorCreate(ndim, shape);
    if (!t){
        return NULL;
    }
    for (size_t i = 0; i < t->size; i++){
        t->data[i] = (float)rand() / (float)RAND_MAX;
    }
    return t;
}

size_t TensorOffset(Tensor *t, const size_t *indices){
    size_t offset = 0;
    for (size_t i = 0; i < t->ndim; i++){
        if (indices[i] >= t->shape[i]){
            return SIZE_MAX;
        }
        
        offset = offset * t->shape[i] + indices[i];
    }
    return offset;
}

float TensorGet(Tensor *t, const size_t *indices) {
    if (!t || !indices){
        return 0.0f;
    }

    size_t offset = TensorOffset(t, indices);
    if (offset == SIZE_MAX){
        return 0.0f;
    }

    return t->data[offset];
}

void TensorSet(Tensor *t, const size_t *indices, float value) {
    if (!t || !indices){
        return;
    }

    size_t offset = TensorOffset(t, indices);

    if (offset != SIZE_MAX){
        t->data[offset] = value;
    }
}

bool TensorCmpShape(Tensor *a, Tensor *b) {
    if (!a || !b || a->ndim != b->ndim){
        return false;
    }

    for (size_t i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            return false;
        }
    }

    return true;
}

Tensor *calculate(Tensor *a, Tensor *b, TensorOp op){
    if (!TensorCmpShape(a, b)){
        return NULL;
    }

    Tensor *out = TensorCreate(a->ndim, a->shape);
    if (!out){
        return NULL;
    }

    for (size_t i = 0; i < a->size; i++){
        out->data[i] = op(a->data[i], b->data[i]);
    }

    return out;
}

Tensor *TensorAdd(Tensor *a, Tensor *b){
    return (calculate(a, b, Tadd));
}

Tensor *TensorsSub(Tensor *a, Tensor *b){
    return (calculate(a, b, Tsubtract));
}

Tensor *TensorsMul(Tensor *a, Tensor *b){
    return (calculate(a, b, Tmultiply));
}

// Scaler mul
Tensor *TensorScale(Tensor *a, float scalar){
    if (!a){
        return NULL;
    }

    Tensor *out = TensorCreate(a->ndim, a->shape);
    if (!out){
        return NULL;
    }

    for (size_t i = 0; i < a->size; i++){
        out->data[i] = a->data[i] * scalar;
    }

    return out;
}

Tensor *TensorMatMul(Tensor *a, Tensor *b){
    if (!a || !b || a->ndim != 2 || b->ndim != 2){
        return NULL;
    }

    size_t m = a->shape[0];
    size_t n = a->shape[1];
    size_t p = b->shape[1];

    if (n != b->shape[0]){
        return NULL;
    }

    size_t shape[] = {m, p};

    Tensor *out = TensorCreate(2, shape);
    if (!out) return NULL;

    for (size_t i = 0; i < m; i++){
        for (size_t j = 0; j < p; j++){
            float sum = 0.0f;

            for (size_t k = 0; k < n; k++){
                sum += a->data[i * n + k] *
                       b->data[k * p + j];
            }

            out->data[i * p + j] = sum;
        }
    }
    return out;
}

Tensor *TensorTranspose(Tensor *a){
    if (!a || a->ndim != 2){
        return NULL;
    }

    size_t rows = a->shape[0];
    size_t cols = a->shape[1];

    size_t shape[] = {cols, rows};

    Tensor *out = TensorCreate(2, shape);
    if (!out){
        return NULL;
    }

    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            out->data[j * rows + i] = a->data[i * cols + j];
        }
    }

    return out;
}

Tensor *tensor_clone(Tensor *a) {
    if (!a){
        return NULL;
    }
    Tensor *out = TensorCreate(a->ndim, a->shape);
    if (!out){
        return NULL;
    }

    memcpy(out->data, a->data, a->size * sizeof(float));

    return out;
}

void TensorPrint(Tensor *t){
    if (!t){
        printf(BOLD_RED "Tensor(NULL)\n" RESET);
        return;
    }

    printf(BOLD_YELLOW "Tensor" RESET "(\n");
    printf("    " BOLD_MAGENTA "ndim" RESET "=" BOLD_YELLOW "%zu" RESET ",\n", t->ndim);
    printf("    " BOLD_MAGENTA "shape" RESET "=[");

    for (size_t i = 0; i < t->ndim; i++){
        printf(BOLD_YELLOW "%zu" RESET, t->shape[i]);

        if (i + 1 < t->ndim){
            printf(WHITE ", " RESET);
        }
    }

    printf(WHITE "],\n" RESET);

    printf("    " BOLD_MAGENTA "size" RESET "=" BOLD_YELLOW "%zu" RESET ",\n", t->size);
    printf("    " BOLD_MAGENTA "data" RESET "=[\n");

    for (size_t i = 0; i < t->size; i++){
        if (t->data[i] < 0){
            printf("        " BOLD_RED "%.4f" RESET, t->data[i]);
        }
        else{
            printf("        " BOLD_BLUE "%.4f" RESET, t->data[i]);
        }
        if (i + 1 < t->size){
            printf(WHITE ", " RESET);

            if (t->ndim > 1 && (i + 1) % t->shape[t->ndim - 1] == 0){
                printf("\n");
            }
        }
    }

    printf("\n    " WHITE "]\n" RESET);
    printf(")\n");
}