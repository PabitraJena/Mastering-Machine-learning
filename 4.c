// XOR gate
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof((arr)[0]))

struct Perceptron{
    int num_inputs;
    float *weights;
    double bias;
};
typedef struct Perceptron Perceptron;

float float_rand(){
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x){
    return (x * (1.0f - x));
}

Perceptron CreatePerceptron(int num_inputs){
    Perceptron p;
    p.num_inputs = num_inputs;
    p.bias = float_rand() * 0.5f;

    p.weights = malloc(num_inputs * sizeof(float));
    for (int i = 0; i < num_inputs; i++){
        p.weights[i] = float_rand() * 0.5f;
    }
    return p;
}

float perceive(Perceptron *p, float inputs[]){
    float sum = p->bias;
    for (int i = 0; i < p->num_inputs; i++){
        sum += inputs[i] * p->weights[i];
    }
    return sum;
}

void FreePerceptron(Perceptron *p) {
    free(p->weights);
}

int main(){
    srand(67);

    float X[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    float Y[] = {0.0f, 1.0f, 1.0f, 0.0f};

    float learning_rate = 0.5f;

    Perceptron p1 = CreatePerceptron(2);
    Perceptron p2 = CreatePerceptron(2);
    Perceptron out = CreatePerceptron(2);

    for (int ep = 0; ep < 100000; ep++){
        for (int samples = 0; samples < 4; samples++){
            float *inputs = X[samples];
            float target = Y[samples];

            float y1 = perceive(&p1, inputs);
            float y2 = perceive(&p2, inputs);

            float a1 = sigmoid(y1);
            float a2 = sigmoid(y2);

            float out_input[] = {a1, a2};

            float y3 = perceive(&out, out_input);
            float pred = sigmoid(y3);

            float err = pred - target;
            float d3 = err * sigmoid_derivative(pred);

            float old_w1 = out.weights[0];
            float old_w2 = out.weights[1];

            out.weights[0] -= learning_rate * d3 * a1;
            out.weights[1] -= learning_rate * d3 * a2;
            out.bias -= learning_rate * d3;

            float d1 = d3 * old_w1 * sigmoid_derivative(a1);
            float d2 = d3 * old_w2 * sigmoid_derivative(a2);

            for (int i = 0; i < 2; i++){
                p1.weights[i] -= learning_rate * d1 * inputs[i];
                p2.weights[i] -= learning_rate * d2 * inputs[i];
            }

            p1.bias -= learning_rate * d1;
            p2.bias -= learning_rate * d2;
        }
    }
    
    
    printf("\n-------------------------\n");

    for (int samples = 0; samples < 4; samples++){
        float *inputs = X[samples];

        float out_input[] = {
            sigmoid(perceive(&p1, inputs)),
            sigmoid(perceive(&p2, inputs))
        };

        float y3 = perceive(&out, out_input);
        float pred = sigmoid(y3);

        printf("%f XOR %f = %.4f (target: %f)\n", X[samples][0], X[samples][1], pred, Y[samples]);
    }

    FreePerceptron(&p1);
    FreePerceptron(&p2);
    FreePerceptron(&out);
    return 0;
}
