#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "bmp.h"

typedef struct{
    int x;
    int y;
} pixel_t;

typedef struct{
    int size;
    int buffer_size;
    pixel_t* pixels;
} stack_t;


// Global variables
int rank,                       // MPI rank
    size,                       // Number of MPI processes
    dims[2],                    // Dimensions of MPI grid
    coords[2],                  // Coordinate of this rank in MPI grid
    periods[2] = {0,0},         // Periodicity of grid
    north,south,east,west,      // Four neighbouring MPI ranks
    image_size[2] = {512,512},  // Hard coded image size
    local_image_size[2],        // Height/width  of local part of image (not including border)
    lsize,                      // Size of local part of image (not including border)
    lsize_border;               // Size of local part of image (including border)


MPI_Comm cart_comm;             // Cartesian communicator


// MPI datatypes, you may have to add more.
MPI_Datatype border_row_t,
             border_col_t,
             border_col_receive_t,
             image_t,
             receive_image_t;


unsigned char *image,           // Entire image, only on rank 0
              *region,          // Region bitmap. 1 if in region, 0 elsewise
              *local_image,     // Local part of image
              *local_region;    // Local part of region bitmap


// Create new pixel stack
stack_t* new_stack(){
    stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
    stack->size = 0;
    stack->buffer_size = 1024;
    stack->pixels = (pixel_t*)malloc(sizeof(pixel_t*)*1024);
}


// Push on pixel stack
void push(stack_t* stack, pixel_t p){
    if(stack->size == stack->buffer_size){
        stack->buffer_size *= 2;
        stack->pixels = realloc(stack->pixels, sizeof(pixel_t)*stack->buffer_size);
    }
    stack->pixels[stack->size] = p;
    stack->size += 1;
}


// Pop from pixel stack
pixel_t pop(stack_t* stack){
    stack->size -= 1;
    return stack->pixels[stack->size];
}


// Check if two pixels are similar. The hardcoded threshold can be changed.
// More advanced similarity checks could have been used.
int similar(unsigned char* im, pixel_t p, pixel_t q){
    int a = im[p.x +  (p.y+1) * (local_image_size[1] + 2) + 1];
    int b = im[q.x +  (q.y+1) * (local_image_size[1] + 2) + 1];
    int diff = abs(a-b);
    return diff < 2;
}


// Create and commit MPI datatypes
void create_types(){

    MPI_Type_contiguous(
        lsize_border,
        MPI_UNSIGNED_CHAR,
        &receive_image_t
    );
    MPI_Type_commit(&receive_image_t);

    MPI_Type_vector(local_image_size[0], 1, local_image_size[1]+2, MPI_UNSIGNED_CHAR, &border_col_t);
    MPI_Type_commit(&border_col_t);

    MPI_Type_contiguous(local_image_size[0], MPI_UNSIGNED_CHAR, &border_col_receive_t);
    MPI_Type_commit(&border_col_receive_t);

    MPI_Type_contiguous(local_image_size[1], MPI_UNSIGNED_CHAR, &border_row_t);
    MPI_Type_commit(&border_row_t);
}


// Send image from rank 0 to all ranks, from image to local_image
void distribute_image(){
    
    MPI_Request request;
    MPI_Irecv(
            local_image,
            1,
            receive_image_t,
            0,
            1,
            cart_comm,
            &request
            );
    
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int receiver_coords[2];
            MPI_Cart_coords(cart_comm, i, 2, receiver_coords);

            unsigned char output_buffer[lsize_border];
            for (int row = 0; row < local_image_size[0] + 2; row++) {
                if (receiver_coords[0] == 0 && row == 0) {
                    continue;
                }
                if (receiver_coords[0] == dims[0] - 1 && row == local_image_size[0] + 1) {
                    continue;
                }

                int offset = 0;
                int length = local_image_size[1] + 2;
                if (receiver_coords[1] == 0) {
                    offset = 1;
                    length--;
                }
                if (receiver_coords[1] == dims[1] - 1) {
                    length--;
                }

                int read_index = image_size[1] * (row-1 + receiver_coords[0] * local_image_size[0]) + receiver_coords[1] * local_image_size[1];
                int write_index = (local_image_size[1]+2) * row + offset;
                memcpy(&output_buffer[write_index], &image[read_index], length);

            }
            MPI_Send(
                output_buffer,
                1,
                receive_image_t,
                i,
                1,
                cart_comm
                );
        }
    }

    MPI_Wait(&request, MPI_STATUS_IGNORE);
}


// Exchange borders with neighbour ranks
void exchange(stack_t* stack){

    MPI_Request request;

    if (south != -2) {
        int send_index = (local_image_size[1]+2) * local_image_size[0] + 1;
        MPI_Isend(&local_region[send_index], 1, border_row_t, south, 2, cart_comm, &request);

        unsigned char receive_buffer[local_image_size[1]];
        MPI_Recv(receive_buffer, 1, border_row_t, south, 2, cart_comm, MPI_STATUS_IGNORE);

        int receive_index = send_index + local_image_size[1] + 2;
        for (int i = 0; i < local_image_size[1]; i++) {
            if (receive_buffer[i] == 1 && local_region[receive_index + i] != 1) {
                local_region[receive_index + i] = 1;

                pixel_t seed;
                seed.x = i;
                seed.y = local_image_size[0];
                push(stack, seed);
            }
        }
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    if (north != -2) {
        int send_index = local_image_size[1] + 2 + 1;
        MPI_Isend(&local_region[send_index], 1, border_row_t, north, 2, cart_comm, &request);

        unsigned char receive_buffer[local_image_size[1]];
        MPI_Recv(receive_buffer, 1, border_row_t, north, 2, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < local_image_size[1]; i++) {
            if (receive_buffer[i] == 1 && local_region[1 + i] != 1) {
                local_region[1 + i] = 1;

                pixel_t seed;
                seed.x = i;
                seed.y = -1;
                push(stack, seed);
            }
        }
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    if (east != -2) {
        int send_index = local_image_size[1] * 2 + 2;
        MPI_Isend(&local_region[send_index], 1, border_col_t, east, 2, cart_comm, &request);

        unsigned char receive_buffer[local_image_size[0]];
        MPI_Recv(receive_buffer, 1, border_col_receive_t, east, 2, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < local_image_size[0]; i++) {
            if (receive_buffer[i] == 1 && local_region[send_index + 1 + (local_image_size[1] + 2) * i] != 1) {
                local_region[send_index + 1 + (local_image_size[1] + 2) * i] = 1;

                pixel_t seed;
                seed.x = local_image_size[1];
                seed.y = i;
                push(stack, seed);
            }
        }
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    if (west != -2) {
        int send_index = local_image_size[1] + 2 + 1;
        MPI_Isend(&local_region[send_index], 1, border_col_t, west, 2, cart_comm, &request);

        unsigned char receive_buffer[local_image_size[0]];
        MPI_Recv(receive_buffer, 1, border_col_receive_t, west, 2, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 0; i < local_image_size[0]; i++) {
            if (receive_buffer[i] == 1 && local_region[send_index - 1 + (local_image_size[1] + 2) * i] != 1) {
                local_region[send_index - 1 + (local_image_size[1] + 2) * i] = 1;

                pixel_t seed;
                seed.x = -1;
                seed.y = i;
                push(stack, seed);
            }
        }
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(cart_comm);
}


// Gather region bitmap from all ranks to rank 0, from local_region to region
void gather_region(){

    MPI_Request request;

    MPI_Isend(
        local_region,
        1,
        receive_image_t,
        0,
        1,
        cart_comm,
        &request
    );

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            unsigned char receive_buffer[lsize_border];
            MPI_Recv(
                receive_buffer,
                1,
                receive_image_t,
                i,
                1,
                cart_comm,
                MPI_STATUS_IGNORE
            );
            int sender_coords[2];
            MPI_Cart_coords(cart_comm, i, 2, sender_coords);


            for (int row = 1; row < local_image_size[0] + 1; row++) {

                int write_index = image_size[1] * (row + local_image_size[0] * sender_coords[0]) + local_image_size[1] * sender_coords[1];
                int read_index = (local_image_size[1] + 2) * row + 1;

                memcpy(&region[write_index], &receive_buffer[read_index], local_image_size[1]);
            }
        }
    }

    MPI_Barrier(cart_comm);
}

// Determine if all ranks are finished. You may have to add arguments.
// You dont have to have this check as a seperate function
int finished(stack_t *stack){
    int finished = (stack->size == 0);
    int global_finished = 0;
    if (rank == 0) {
        int num_finished = finished;
        for (int i = 1; i < size; i++) {
            int rank_finished;
            MPI_Recv(&rank_finished, 1, MPI_INT, i, 3, cart_comm, MPI_STATUS_IGNORE);
            if (rank_finished != 0) {
                num_finished++;
            }
        }
        if (num_finished == size) {
            global_finished = 1;
        }
        for (int i = 1; i < size; i++) {
            MPI_Send(&global_finished, 1, MPI_INT, i, 3, cart_comm);
        }
    } else {
        MPI_Request request;
        MPI_Isend(&finished, 1, MPI_INT, 0, 3, cart_comm, &request);

        MPI_Recv(&global_finished, 1, MPI_INT, 0, 3, cart_comm, MPI_STATUS_IGNORE);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    return global_finished;
}


// Check if pixel is inside local image
int inside(pixel_t p){
    return (p.x >= 0 && p.x < local_image_size[1] && p.y >= 0 && p.y < local_image_size[0]);
}


// Adding seeds in corners.
void add_seeds(stack_t* stack){
    int seeds [8];
    seeds[0] = 5;
    seeds[1] = 5;
    seeds[2] = image_size[1]-5;
    seeds[3] = 5;
    seeds[4] = image_size[1]-5;
    seeds[5] = image_size[0]-5;
    seeds[6] = 5;
    seeds[7] = image_size[0]-5;
    
    for(int i = 0; i < 4; i++){
        pixel_t seed;
        seed.x = seeds[i*2] - coords[1]*local_image_size[1];
        seed.y = seeds[i*2+1] -coords[0]*local_image_size[0];
        
        if(inside(seed)){
            push(stack, seed);
        }
    }
}

void draw_chessboard() {
    for (int i = 0; i < local_image_size[0] + 2; i++) {
        for (int j = 0; j < local_image_size[1] + 2; j++) {
            if (i % 2 == 0 && j % 2 == 0) {
                local_region[i * (local_image_size[1] + 2) + j] = 1;
            } else if (i % 2 == 1 && j % 2 == 1) {
                local_region[i * (local_image_size[1] + 2) + j] = 1;
            } else {
                local_region[i * (local_image_size[1] + 2) + j] = 0;
            }
        }
    }
}


// Region growing, serial implementation
void grow_region() {

    stack_t *stack = new_stack();
    add_seeds(stack);

    int local_region_width = local_image_size[1] + 2;

    while (finished(stack) == 0) {
        while(stack->size > 0) {
            pixel_t pixel = pop(stack);

            local_region[(pixel.y+1) * local_region_width + pixel.x + 1] = 1;


            int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
            for (int c = 0; c < 4; c++) {
                pixel_t candidate;
                candidate.x = pixel.x + dx[c];
                candidate.y = pixel.y + dy[c];

                if (!inside(candidate)) {
                    continue;
                }

                if (local_region[(candidate.y+1) * local_region_width + candidate.x + 1]) {
                    continue;
                }

                if (similar(local_image, pixel, candidate)) {
                    local_region[(candidate.y+1) * local_region_width + candidate.x + 1] = 1;
                    push(stack, candidate);
                }
            }
        }
        exchange(stack);
    }
}


// MPI initialization, setting up cartesian communicator
void init_mpi(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm );
    MPI_Cart_coords( cart_comm, rank, 2, coords );
    
    MPI_Cart_shift( cart_comm, 0, 1, &north, &south );
    MPI_Cart_shift( cart_comm, 1, 1, &west, &east );
}


void load_and_allocate_images(int argc, char** argv){

    if(argc != 2){
        printf("Useage: region file");
        exit(-1);
    }
    
    if(rank == 0){
        image = read_bmp(argv[1]);
        region = (unsigned char*)calloc(sizeof(unsigned char),image_size[0]*image_size[1]);
    }
    
    local_image_size[0] = image_size[0]/dims[0];
    local_image_size[1] = image_size[1]/dims[1];
    
    lsize = local_image_size[0]*local_image_size[1];
    lsize_border = (local_image_size[0] + 2)*(local_image_size[1] + 2);
    local_image = (unsigned char*)malloc(sizeof(unsigned char)*lsize_border);
    local_region = (unsigned char*)calloc(sizeof(unsigned char),lsize_border);
}


void write_image(){
    if(rank==0){
        for(int i = 0; i < image_size[0]*image_size[1]; i++){

            image[i] *= (region[i] == 0);
        }
        write_bmp(image, image_size[0], image_size[1]);
    }
}


int main(int argc, char** argv){
    
    init_mpi(argc, argv);
    
    load_and_allocate_images(argc, argv);
    
    create_types();
    
    distribute_image();

    grow_region();
    //draw_chessboard();
    
    gather_region();
    
    MPI_Finalize();
    
    write_image();
    
    exit(0);
}
