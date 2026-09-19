// OR gate

#include <stdio.h>

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof((arr)[0]))

int main(){
    int X[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    int Y[] = {0, 1, 1, 1};

    float learning_rate = 0.01f;

    // y = w1x1 + w2x2 + b 
    float w1 = 0.0f;
    float w2 = 0.0f;
    float b = 0.0f;

    for (int ep = 0; ep < 1000; ep++){
        float dw1 = 0.0f;
        float dw2 = 0.0f;
        float db = 0.0f;

        for (int i = 0; i < ARRAY_LENGTH(X); i++){
            int x1 = X[i][0];
            int x2 = X[i][1];

            float y = w1 * x1 + w2 * x2 + b;
            float err = y - Y[i];

            dw1 += err * x1;
            dw2 += err * x2;
            db += err;
        }

        dw1 /= ARRAY_LENGTH(X);
        dw2 /= ARRAY_LENGTH(X);
        db /= ARRAY_LENGTH(X);

        w1 -= learning_rate * dw1;
        w2 -= learning_rate * dw2;
        b -= learning_rate * db;
        printf("w1=%f, w2=%f, b=%f\n", w1, w2, b);
    }

    printf("\n-------------------------\n");
    printf("0 | 0   :   %f\n", (0 * w1) +(0 * w2) +b);
    printf("0 | 1   :   %f\n", (0 * w1) +(1 * w2) +b);
    printf("1 | 0   :   %f\n", (1 * w1) +(0 * w2) +b);
    printf("1 | 1   :   %f\n", (1 * w1) +(1 * w2) +b);
    return 0;
}
