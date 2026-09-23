#pragma once

#include <math.h>

enum ActivationType{
    ACTIVATION_NONE,
    ACTIVATION_SIGMOID,
};
typedef enum ActivationType ActivationType;

float ActivationForward(ActivationType type, float x){
    switch (type) {
        case ACTIVATION_SIGMOID:
            return 1.0f / (1.0f + expf(-x));

        case ACTIVATION_NONE:
        default:
            return x;
    }
}

float ActivationDerivative(ActivationType type, float x){
    switch (type) {
        case ACTIVATION_SIGMOID: {
            float y = ActivationForward(type, x);
            return y * (1.0f - y);
        }

        case ACTIVATION_NONE:
        default:
            return 1.0f;
    }
}