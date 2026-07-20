/*
Model to predict the function y = 2x + 1
*/

#include <stdio.h>

int main(){
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {3, 5, 7, 9, 11};
    int n = 5;

    double w = 0.0f;
    double b = 0.0f;

    double learning_rate = 0.01f;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++){
        double dw = 0.0f; // w gradient
        double db = 0.0f; // b gradient

        double loss = 0.0f;

        for (int i = 0; i < n; i++){
            double prediction = w * x[i] + b;
            double error = prediction - y[i];

            loss += error * error; // square error

            dw += error * x[i];
            db += error;
        }
        // Average gradient
        dw = (2.0 / n) * dw;
        db = (2.0 / n) * db;
        loss /= n;

        w -= learning_rate * dw;
        b -= learning_rate * db;

        if (epoch % 100 == 0) {
            printf("Epoch %4d | Loss = %.6f | w = %.6f | b = %.6f\n", epoch, loss, w, b);
        }
    }

    printf("\nFinal model:\n");
    printf("y = %.4fx + %.4f\n", w, b);

    double test_x = 10;
    double prediction = w * test_x + b;
    printf("\nPrediction for x = %.2f: %.4f\n", test_x, prediction);

    return 0;
}