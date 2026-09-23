#include "cynapse/nn.h"
#include "cynapse/matrix.h"

int main(void){
    srand(67);

    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    float targets[4] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    int layer_sizes[] = { 2, 2, 1 };

    ActivationType activations[] = {
        ACTIVATION_SIGMOID,
        ACTIVATION_SIGMOID
    };

    NeuralNetwork nn = CreateNetwork( 2, layer_sizes, activations);

    if (!nn.layers){
        log_message(INFO, "Failed to create neural network.");
        return 1;
    }


    const int epochs = 100000;
    const float learning_rate = 0.1f;
    for (int epoch = 0; epoch < epochs; epoch++){
        float total_loss = 0.0f;
        for (int sample = 0; sample < 4; sample++){

            Matrix input = CreateMatrix(2, 1);

            input.data[0] = inputs[sample][0];
            input.data[1] = inputs[sample][1];

            NetworkForward( &nn, &input);

            float prediction = nn.layers[nn.num_layers - 1] .output.data[0];

            float error = prediction - targets[sample];
            total_loss +=  (error * error) / 2;
            Matrix gradient = CreateMatrix(1, 1);
            gradient.data[0] = error;

            NetworkBackward( &nn, &gradient );

            NetworkUpdate( &nn, learning_rate );
            FreeMatrix(&gradient);
            FreeMatrix(&input);
        }

        if (epoch % 10000 == 0){
            printf( "Epoch: %d | Loss: %.6f\n", epoch, total_loss / 4.0f);
        }
    }

    printf("\n-------------------------\n");

    for (int sample = 0; sample < 4; sample++){

        Matrix input = CreateMatrix(2, 1);

        input.data[0] = inputs[sample][0];
        input.data[1] = inputs[sample][1];

        NetworkForward( &nn, &input);

        float prediction = nn.layers[nn.num_layers - 1].output.data[0];

        int result = prediction >= 0.5f ? 1 : 0;
        printf( "%.0f XOR %.0f = %.6f -> %d\n", inputs[sample][0], inputs[sample][1], prediction, result);

        FreeMatrix(&input);
    }
    FreeNetwork(&nn);
    return 0;
}
