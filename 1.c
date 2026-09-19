// y = 2x + 1

#include <stdio.h>

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof((arr)[0]))

#define APPROACH 1

int X[] = {1, 2, 3, 4, 5, 6};
int Y[] = {3, 5, 7, 9, 11, 13};

// The Tsoding's finite approximation approach
float cost(float w, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < ARRAY_LENGTH(X); ++i) {
        float x = X[i];
        float y = Y[i];
        // Prediction formula: y = wx + b
        float d = x * w + b - y;
        result += d * d;
    }
    result /= ARRAY_LENGTH(X);
    return result;
}

int main(){
    float learning_rate = 0.01f;
    float eps = 0.001f;
    // y = wx + b
    float w = 0.0f;
    float b = 0.0f;
    for (int ep = 0; ep < 1000; ep++){
        float dw = 0.0f;
        float db = 0.0f;
        #if APPROACH
            float c = cost(w, b);

            dw = (cost(w + eps, b) - c) / eps;
            db = (cost(w, b + eps) - c) / eps;
        #else
            for (size_t i = 0; i < ARRAY_LENGTH(X); ++i) {
                float y = w * X[i] + b;
                float err = y - Y[i];

                dw += err * X[i];
                db += err;
            }

            dw /= ARRAY_LENGTH(X);
            db /= ARRAY_LENGTH(X);
        #endif

        w -= learning_rate * dw;
        b -= learning_rate * db;
        printf("w: %f, b: %f\n", w, b);
    }
    return 0;
}