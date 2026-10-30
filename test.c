
#include "Cynapse/tensed/tensor.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    size_t shape_test[] = {2, 4};

    Tensor *a = TensorRand(2, shape_a);
    Tensor *b = TensorRand(2, shape_b);
    Tensor *test = TensorZeros(2, shape_test);

    if (!a || !b || !test) {
        TensorFree(a);
        TensorFree(b);
        TensorFree(test);
        return 1;
    }

    printf("Tensor A:\n");
    TensorPrint(a);

    printf("\nTensor B:\n");
    TensorPrint(b);

    Tensor *c = TensorMatMul(a, b);

    printf("\nA @ B:\n");
    TensorPrint(c);

    float values[] = {
        1.5f, -2.3f,  3.7f, -4.1f,
        -5.6f,  6.2f, -7.8f,  8.9f
    };

    for (size_t i = 0; i < test->size; i++) {
        test->data[i] = values[i];
    }

    TensorPrint(test);

    TensorFree(a);
    TensorFree(b);
    TensorFree(c);
    TensorFree(test);

    return 0;
}
