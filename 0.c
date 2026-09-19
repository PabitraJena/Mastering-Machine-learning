// predicting y = 2x

#include <stdio.h>

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof((arr)[0]))

int main(){
    int X[] = {1, 2, 3, 4, 5, 6};
    int Y[] = {2, 4, 6, 8, 10, 12};

    float learning_rate = 0.01f;

    // y = wx
    float w = 0.0f;
    for (int ep = 0; ep < 100; ep++){
        float dw = 0.0f;
        for (int i = 0; i < ARRAY_LENGTH(X); i++){
            float y = w * X[i];
            float err = y - Y[i];

            dw += err * X[i];
        }

        dw /= ARRAY_LENGTH(X);
        w -= learning_rate * dw;

        printf("w: %f\n", w);
    }
    return 0;
}