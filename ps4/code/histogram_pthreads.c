#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;
int n_threads;

unsigned char* image;
unsigned char* output_image;
int* histogram;
float *transfer_function;

// Shared barrier variables
int counter = 0;
pthread_mutex_t barrier_mutex;
pthread_cond_t cond_var;

pthread_mutex_t histogram_mutex;

void barrier() {
    pthread_mutex_lock(&barrier_mutex);
    counter++;
    if (counter == n_threads){
        counter = 0;
        pthread_cond_broadcast(&cond_var);
    } else {
        while (pthread_cond_wait(&cond_var, &barrier_mutex)!=0);
    }
    pthread_mutex_unlock(&barrier_mutex);
}

void *thread_loops(void *arg) {
    int rank = (long)arg;
    printf("Thread %d\n", rank);

    int *local_histogram = (int *)calloc(sizeof(int), color_depth);
    for (int i = rank; i < image_size; i+= n_threads) {
        local_histogram[image[i]]++;
    }

    pthread_mutex_lock(&histogram_mutex);

    for (int i = 0; i < color_depth; i++) {
        histogram[i] += local_histogram[i];
    }

    pthread_mutex_unlock(&histogram_mutex);

    barrier();

    for (int i = rank; i < color_depth; i += n_threads) {
        for (int j = 0; j < i+1; j++) {
            transfer_function[i] += color_depth*((float)histogram[j])/(image_size);
        }
    }

    barrier();


    for (int i = rank; i < image_size; i += n_threads) {
        output_image[i] = transfer_function[image[i]];
    }

    printf("Finished thread %d\n", rank);

    return NULL;
}

int main(int argc, char** argv){

    if (argc != 3) {
        printf("Useage: %s image n_threads\n", argv[0]);
        exit(-1);
    }
    n_threads = atoi(argv[2]);

    image = read_bmp(argv[1]);
    output_image = malloc(sizeof(unsigned char) * image_size);

    histogram = (int *)calloc(sizeof(int), color_depth);
    transfer_function = (float *)calloc(sizeof(float), color_depth);

    pthread_t threads[n_threads-1];
    for (long i = 0; i < n_threads - 1; i++) {
        pthread_create(&threads[i], NULL, thread_loops, (void*)i+1);
    }
    thread_loops(NULL);

    for (int i = 0; i < n_threads - 1; i++) {
        pthread_join(threads[i], NULL);
    }

    write_bmp(output_image, image_width, image_height);
    printf("finished\n");
}
