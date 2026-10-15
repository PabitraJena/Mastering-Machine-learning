#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include "cynapse/tensed/nn_tensor.h"
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
    srand((unsigned int)time(NULL));

    MNISTImages images = LoadImages("data/train-images.idx3-ubyte");

    MNISTLabels labels = LoadLabels("data/train-labels.idx1-ubyte");

    if (images.count != labels.count){
        log_message(ERROR, "Image/label count mismatch");
        free(images.data);
        free(labels.data);
        return EXIT_FAILURE;
    }

    size_t image_size = (size_t)images.rows * images.cols;

    if (image_size != 784){
        log_message(ERROR, "Expected 28x28 MNIST images");
        free(images.data);
        free(labels.data);
        return EXIT_FAILURE;
    }

    size_t layer_sizes[] = {784, 128, 64, 10};
    TensorNetwork nn = createTensorNetwork(3, layer_sizes);
    
    if (nn.layers == NULL){
        log_message(ERROR, "Failed to create neural network");
        free(images.data);
        free(labels.data);
        return EXIT_FAILURE;
    }
    
    size_t input_shape[] = {784, 1};
    size_t output_shape[] = {10, 1};
    
    Tensor input = createTensor(2, input_shape);
    Tensor probabilities = createTensor(2, output_shape);
    
    if (!Tensor_isValid(&input) || !Tensor_isValid(&probabilities)){
        log_message(ERROR, "Failed to allocate tensors");
        
        freeTensor(&input);
        freeTensor(&probabilities);
        FreeTensorNetwork(&nn);
        free(images.data);
        free(labels.data);
        return EXIT_FAILURE;
    }
    
    const int epochs = 10;
    const float learning_rate = 0.001f;
    
    for (int e = 0; e < epochs; e++){
        double total_loss = 0.0;
        size_t correct = 0;
        
        for (uint32_t index = 0; index < images.count; index++){
            // Image value normalised to 0.0f - 1.0f
            for (size_t i = 0; i < image_size; i++) {
                input.data[i] = images.data[(size_t)index * image_size + i] / 255.0f;
            }
            TensorNetworkForward(&nn, &input);

            Tensor *logits = &nn.layers[nn.nlayers - 1].a;
            TensorSoftmax(logits, &probabilities);

            size_t prediction = TensorArgMax(&probabilities);
            size_t target = labels.data[index];

            if (target >= 10){
                log_message(ERROR, "Invalid MNIST label: %zu", target);
                freeTensor(&input);
                freeTensor(&probabilities);
                FreeTensorNetwork(&nn);
                free(images.data);
                free(labels.data);
                return EXIT_FAILURE;
            }

            if (prediction == target){
                correct++;
            }
            total_loss -= logf(probabilities.data[target] + 1e-8f);

            TensornetworkBackward(&nn, &input, &probabilities, target);
            TensorNetworkUpdate(&nn, learning_rate);

            if (index % 1000 == 0) {
                log_message(INFO, "Epoch %d: image %u / %u", e + 1, index, images.count);
            }
        }
        log_message( INFO, "Epoch %d | Loss: %.4f | Accuracy: %.2f%%", e + 1, (float)(total_loss / images.count), 100.0f * (float) correct / images.count);
    }
    
    freeTensor(&input);
    freeTensor(&probabilities);
    FreeTensorNetwork(&nn);

    free(images.data);
    free(labels.data);
    return EXIT_SUCCESS;
}