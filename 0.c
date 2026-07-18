#include <stdio.h>

int main() {
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {2, 4, 6, 8, 10};
    int n = 5;

    double w = 0.0;

    double learning_rate = 0.01;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double gradient = 0.0;

        for (int i = 0; i < n; i++) {
            double prediction = w * x[i];
            double error = prediction - y[i];
            gradient += error * x[i];
        }

        gradient /= n;
        w -= learning_rate * gradient;
    }

    printf("Learned weight: %.4f\n", w);

    double test_x = 7;
    double predicted_y = w * test_x;

    printf("Prediction for x = %.1f: %.2f\n", test_x, predicted_y);

    return 0;
}