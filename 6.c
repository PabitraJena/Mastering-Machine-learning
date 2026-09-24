// Mnist classifier

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cynapse/nn.h"
#include "cynapse/logger.h"

struct MNISTImages{
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
    unsigned char *data;
};
typedef struct MNISTImages MNISTImages;

struct MNISTLabels{
    uint32_t count;
    unsigned char *data;
};
typedef struct MNISTLabels MNISTLabels; 

uint32_t read_uint32_big_enndian(FILE *f){
    unsigned char bytes[4];
    if (fread(bytes, 1, 4, f) != 4){
        log_message(ERROR, "Failed to read 4 bytes");
        exit(EXIT_FAILURE);
    }
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | ((uint32_t)bytes[3]);
}

MNISTImages LoadImages(const char *filename){
    FILE *f = fopen(filename, "rb");
    if (!f){
        log_message(ERROR, "File could not be opened");
        exit(EXIT_FAILURE);
    }

    uint32_t magic = read_uint32_big_enndian(f);
    uint32_t count = read_uint32_big_enndian(f);
    uint32_t rows  = read_uint32_big_enndian(f);
    uint32_t cols  = read_uint32_big_enndian(f);
    
    if (magic != 2051){
        log_message(ERROR, "Invalid image file: magic = %u\n", magic);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    size_t size = (size_t)count * rows * cols;
    unsigned char *data = malloc(size);
    
    if (!data){
        log_message(ERROR, "Memory allocation failed");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (fread(data, 1, size, f) != size){
        log_message(ERROR, "Failed to readimage data");
        free(data);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);

    MNISTImages images = {
        .count = count,
        .rows = rows,
        .cols = cols,
        .data = data
    };

    log_message(SUCCESS, filename);
    return images;
}

MNISTLabels LoadLabels(const char *filename){
    FILE *f = fopen(filename, "rb");
    if (!f){
        log_message(ERROR, "File could not be opened");
        exit(EXIT_FAILURE);
    }
    uint32_t magic = read_uint32_big_enndian(f);
    uint32_t count = read_uint32_big_enndian(f);
    if (magic != 2049){
        log_message(ERROR, "Invalid image file: magic = %u\n", magic);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    unsigned char *data = malloc(count);
    if (!data) {
        log_message(ERROR, "Memory allocation failed\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (fread(data, 1, count, f) != count) {
        log_message(ERROR, "Failed to read labels\n");
        free(data);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);

    MNISTLabels labels = {
        .count = count,
        .data = data
    };

    log_message(SUCCESS, filename);
    return labels;
}

unsigned char GetPixel(MNISTImages *images, int index, uint32_t row, uint32_t col){
    return (images->data[index * images->rows * images->cols + row * images->cols + col]);
}

void PrintImage(MNISTImages *images, int index){
    for (uint32_t r = 0; r < images->rows; r++){
        for (uint32_t c = 0; c < images->cols; c++){
            unsigned char pixel = GetPixel(images, index, r, c);

            if (pixel > 200){
                printf("##");
            }
            else if (pixel > 100){
                printf("++");
            }
            else if (pixel > 30){
                printf("..");
            }
            else{
                printf("  ");
            }
        }
        printf("\n");
    }
}

int main(){
    srand(67);

    MNISTImages images = LoadImages("data/train-images.idx3-ubyte");
    log_message(INFO, "Images: %u", images.count);
    log_message(INFO, "Size: %ux%u", images.rows, images.cols);

    MNISTLabels labels = LoadLabels("data/train-labels.idx1-ubyte");
    log_message(INFO, "Labels: %u", labels.count);

    int index = rand() % 784 + 1;

    PrintImage(&images, index);
    log_message(INFO, "Label: %d", labels.data[index]);

    size_t image_size = (size_t)images.rows * images.cols;

    int layer_sizes[] = {784, 128, 64, 10};
    ActivationType activations[] = {
        ACTIVATION_RELU,
        ACTIVATION_RELU,
        ACTIVATION_NONE
    };
    
    NeuralNetwork nn = CreateNetwork( 3, layer_sizes, activations);

    int epochs = 3;
    float learning_rate = 0.001f;
    Matrix input = CreateMatrix((int)image_size, 1);
    Matrix probabilities = CreateMatrix(10, 1);
    Matrix gradient = CreateMatrix(10,1);

    int correct;

    for (int ep = 0; ep < epochs; ep++){
        float total_loss = 0.0f;
        correct = 0;
        for (uint32_t index = 0; index < images.count; index++){
            for (size_t i = 0; i < image_size; i++){
                input.data[i] = images.data[index * image_size + i] / 255.0f;
            }

            if (index % 1000 == 0){
                log_message(SUCCESS, "Epoch %d: image %u / %u", ep + 1, index, images.count);
            }


            NetworkForward(&nn, &input);

            Matrix *last_layer = &nn.layers[nn.num_layers - 1].output;
            Softmax(last_layer, &probabilities);

            int prediction = ArgMax(&probabilities);
            if (prediction == labels.data[index]){
                correct++;
            }
            
            int target = labels.data[index];

            float loss = -logf(probabilities.data[target] + 1e-8f);
            total_loss += loss;

            for (int i = 0; i < 10; i++){
                gradient.data[i] = probabilities.data[i];
            }
            gradient.data[target] -= 1.0f;

            NetworkBackward(&nn, &gradient);
            NetworkUpdate(&nn, learning_rate);
        }

        log_message(INFO, "Epoch %d | Loss: %f | Accuracy: %.2f%%",ep + 1,total_loss / images.count, 100.0f * (float)correct / images.count);
        
    }

    int test_index = rand() % 784 + 1;
    int test_pred = ArgMax(&probabilities);
    PrintImage(&images,test_index);
    printf("Predicted Label: %d\n", test_pred);
    printf("Actual Label: %d\n", labels.data[test_index]);

    FreeMatrix(&probabilities);
    FreeNetwork(&nn);
    free(images.data);
    free(labels.data);
    return 0;
}