#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

#include "bmp.h"

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512


#define GROW_REGION_SERIAL 0
#define RAYCAST_SERIAL 1
#define RAYCAST_TEXTURE 0

texture<unsigned char, cudaTextureType3D, cudaReadModeElementType> data_texture;
texture<unsigned char, cudaTextureType3D, cudaReadModeElementType> region_texture;


// Stack for the serial region growing
typedef struct{
    int size;
    int buffer_size;
    int3* pixels;
} stack_t;

stack_t* new_stack(){
    stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
    stack->size = 0;
    stack->buffer_size = 1024;
    stack->pixels = (int3*)malloc(sizeof(int3)*1024);

    return stack;
}

void push(stack_t* stack, int3 p){
    if(stack->size == stack->buffer_size){
        stack->buffer_size *= 2;
        int3* temp = stack->pixels;
        stack->pixels = (int3*)malloc(sizeof(int3)*stack->buffer_size);
        memcpy(stack->pixels, temp, sizeof(int3)*stack->buffer_size/2);
        free(temp);

    }
    stack->pixels[stack->size] = p;
    stack->size += 1;
}

int3 pop(stack_t* stack){
    stack->size -= 1;
    return stack->pixels[stack->size];
}

// float3 utilities
__host__ __device__ float3 cross(float3 a, float3 b){
    float3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

__host__ __device__ float3 normalize(float3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;

    return v;
}

__host__ __device__ float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}

__host__ __device__ float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;

    return a;
}


// Prints CUDA device properties
void print_properties(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);

    cudaDeviceProp p;
    cudaSetDevice(0);
    cudaGetDeviceProperties (&p, 0);
    printf("Compute capability: %d.%d\n", p.major, p.minor);
    printf("Name: %s\n" , p.name);
    printf("\n\n");
}


// Fills data with values
unsigned char func(int x, int y, int z){
    unsigned char value = rand() % 20;

    int x1 = 300;
    int y1 = 400;
    int z1 = 100;
    float dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

    if(dist < 100){
        value  = 30;
    }

    x1 = 100;
    y1 = 200;
    z1 = 400;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

    if(dist < 50){
        value = 50;
    }

    if(x > 200 && x < 300 && y > 300 && y < 500 && z > 200 && z < 300){
        value = 45;
    }
    if(x > 0 && x < 100 && y > 250 && y < 400 && z > 250 && z < 400){
        value =35;
    }
    return value;
}

unsigned char* create_data(){
    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_DIM*DATA_DIM*DATA_DIM);

    for(int i = 0; i < DATA_DIM; i++){
        for(int j = 0; j < DATA_DIM; j++){
            for(int k = 0; k < DATA_DIM; k++){
                data[i*DATA_DIM*DATA_DIM + j*DATA_DIM+ k]= func(k,j,i);
            }
        }
    }

    return data;
}

// Checks if position is inside the volume (float3 and int3 versions)
__host__ __device__ int inside(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

    return x && y && z;
}

__host__ __device__ int inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);

    return x && y && z;
}

// Indexing function (note the argument order)
__host__ __device__ int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Trilinear interpolation
__host__ __device__ float value_at(float3 pos, unsigned char* data){
    if(!inside(pos)){
        return 0;
    }

    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);

    int x_u = ceil(pos.x);
    int y_u = ceil(pos.y);
    int z_u = ceil(pos.z);

    float rx = pos.x - x;
    float ry = pos.y - y;
    float rz = pos.z - z;

    float a0 = rx*data[index(z,y,x)] + (1-rx)*data[index(z,y,x_u)];
    float a1 = rx*data[index(z,y_u,x)] + (1-rx)*data[index(z,y_u,x_u)];
    float a2 = rx*data[index(z_u,y,x)] + (1-rx)*data[index(z_u,y,x_u)];
    float a3 = rx*data[index(z_u,y_u,x)] + (1-rx)*data[index(z_u,y_u,x_u)];

    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;

    float c0 = rz*b0 + (1-rz)*b1;


    return c0;
}


// Serial ray casting
unsigned char* raycast_serial(unsigned char* data, unsigned char* region){
    unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    // For each pixel
    for(int y = -(IMAGE_DIM/2); y < (IMAGE_DIM/2); y++){
        for(int x = -(IMAGE_DIM/2); x < (IMAGE_DIM/2); x++){

            // Find the ray for this pixel
            float3 screen_center = add(camera, forward);
            float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
            ray = add(ray, scale(camera, -1));
            ray = normalize(ray);
            float3 pos = camera;

            // Move along the ray, we stop if the color becomes completely white,
            // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
            // to be big enough to let rays pass through the entire volume)
            int i = 0;
            float color = 0;
            while(color < 255 && i < 5000){
                i++;
                pos = add(pos, scale(ray, step_size));          // Update position
                int r = value_at(pos, region);                  // Check if we're in the region
                color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
            }

            // Write final color to image
            image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
        }
    }

    return image;
}


// Check if two values are similar, threshold can be changed.
__host__ __device__ int similar(unsigned char* data, int3 a, int3 b){
    unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
    unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];

    int i = abs(va-vb) < 1;
    return i;
}


// Serial region growing, same algorithm as in assignment 2
unsigned char* grow_region_serial(unsigned char* data){
    unsigned char* region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);

    stack_t* stack = new_stack();

    int3 seed = {.x=50, .y=300, .z=300};
    push(stack, seed);
    region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 1;

    int dx[6] = {-1,1,0,0,0,0};
    int dy[6] = {0,0,-1,1,0,0};
    int dz[6] = {0,0,0,0,-1,1};

    while(stack->size > 0){
        int3 pixel = pop(stack);
        for(int n = 0; n < 6; n++){
            int3 candidate = pixel;
            candidate.x += dx[n];
            candidate.y += dy[n];
            candidate.z += dz[n];

            if(!inside(candidate)){
                continue;
            }

            if(region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
                continue;
            }

            if(similar(data, pixel, candidate)){
                push(stack, candidate);
                region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 1;
            }
        }
    }

    return region;
}


__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    printf("hello\n");
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    // Find the ray for this pixel
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    // Move along the ray, we stop if the color becomes completely white,
    // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
    // to be big enough to let rays pass through the entire volume)
    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = value_at(pos, region);                  // Check if we're in the region
        color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
    }

    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
}


__global__ void raycast_kernel_texture(unsigned char* image){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    // Find the ray for this pixel
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    // Move along the ray, we stop if the color becomes completely white,
    // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
    // to be big enough to let rays pass through the entire volume)
    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));                        // Update position
        int r = tex3D(region_texture, pos.x, pos.y, pos.z);           // Check if we're in the region
        color += tex3D(data_texture, pos.x, pos.y, pos.z)*(0.01 + r); // Update the color based on data value, and if we're in the region
    }

    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;

}


unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    dim3 gridBlock, threadBlock;

    gridBlock.x = 16;
    gridBlock.y = 16;

    threadBlock.x = 32;
    threadBlock.y = 32;

    unsigned char *device_image;
    unsigned char *device_data;
    unsigned char *device_region;

    cudaMalloc(&device_image, sizeof(unsigned char) * IMAGE_DIM * IMAGE_DIM);
    cudaMalloc(&device_data, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);
    cudaMalloc(&device_region, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);

    cudaMemcpy(device_data, data, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(device_region, region, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM, cudaMemcpyHostToDevice);

    raycast_kernel<<<gridBlock, threadBlock>>>(device_data, device_image, device_region);

    unsigned char *image = (unsigned char *)malloc(sizeof(unsigned char) * IMAGE_DIM * IMAGE_DIM);
    cudaMemcpy(image, device_image, sizeof(unsigned char) * IMAGE_DIM * IMAGE_DIM, cudaMemcpyDeviceToHost);

    return image;
}


unsigned char* raycast_gpu_texture(unsigned char* data, unsigned char* region){
    dim3 gridBlock, threadBlock;

    gridBlock.x = 16;
    gridBlock.y = 16;

    threadBlock.x = 32;
    threadBlock.y = 32;

    unsigned char *device_image;

    cudaMalloc(&device_image, sizeof(unsigned char) * IMAGE_DIM * IMAGE_DIM);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);

    cudaExtent extent = make_cudaExtent(DATA_DIM, DATA_DIM, DATA_DIM);

    cudaArray* device_data;

    cudaMalloc3DArray(&device_data, &channelDesc, extent, 0);

    cudaMemcpy3DParms data_copyParams = {0};
    data_copyParams.srcPtr   = make_cudaPitchedPtr(data, DATA_DIM * sizeof(unsigned char), DATA_DIM, DATA_DIM);
    data_copyParams.dstArray = device_data;
    data_copyParams.extent   = extent;
    data_copyParams.kind     = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&data_copyParams);

    cudaBindTextureToArray(data_texture, device_data);

    cudaArray* device_region;

    cudaMalloc3DArray(&device_region, &channelDesc, extent, 0);

    cudaMemcpy3DParms region_copyParams = {0};
    region_copyParams.srcPtr   = make_cudaPitchedPtr(region, DATA_DIM * sizeof(unsigned char), DATA_DIM, DATA_DIM);
    region_copyParams.dstArray = device_region;
    region_copyParams.extent   = extent;
    region_copyParams.kind     = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&region_copyParams);

    cudaBindTextureToArray(region_texture, device_region);

    raycast_kernel_texture<<<gridBlock, threadBlock>>>(device_image);

    unsigned char *image = (unsigned char *)malloc(sizeof(unsigned char) * IMAGE_DIM * IMAGE_DIM);
    cudaMemcpy(image, device_image, sizeof(unsigned char) * IMAGE_DIM * IMAGE_DIM, cudaMemcpyDeviceToHost);

    return image;
}


__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* finished){
    printf("hello\n");
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    int threadZ = blockIdx.z * blockDim.z + threadIdx.z;

    int threadId = index(threadZ, threadY, threadX);
    printf("z: %d\n", threadZ);

    if (region[threadId] == 2) {
        printf("match\n");
        region[threadId] = 1;

        int dx[6] = {-1,1,0,0,0,0};
        int dy[6] = {0,0,-1,1,0,0};
        int dz[6] = {0,0,0,0,-1,1};

        int3 pixel = {.x=threadX, .y=threadY, .z=threadZ};

        for (int n = 0; n < 6; n++) {
            int3 candidate = pixel;
            candidate.x += dx[n];
            candidate.y += dy[n];
            candidate.z += dz[n];

            if (!inside(candidate)) {
                continue;
            }

            if (region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]) {
                continue;
            }

            if (similar(data, pixel, candidate)) {
                region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 2;
                *finished = 0;
            }
        }
    }
}


__global__ void region_grow_kernel_shared(unsigned char* data, unsigned char* region, int* finished){
    /*
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    int threadZ = blockIdx.z * blockDim.z + threadIdx.z;

    int threadId = index(threadZ, threadY, threadX);

    int localThreadId = threadZ * 17 * 17 + threadY * 17 + threadX;

    __shared__ unsigned char *local_region[17*17*17];

    stack_t* stack = new_stack();

    int3 seed = {.x=50, .y=300, .z=300};
    push(stack, seed);
    region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 1;


    if (local_region[localThreadId] == 2) {
        local_region[localThreadId] = 1;

        int dx[6] = {-1,1,0,0,0,0};
        int dy[6] = {0,0,-1,1,0,0};
        int dz[6] = {0,0,0,0,-1,1};

        int3 pixel = {.x=threadIdx.x, .y=threadIdx.y, .z=threadIdx.z};

        for (int n = 0; n < 6; n++) {
            int3 candidate = pixel;
            candidate.x += dx[n];
            candidate.y += dy[n];
            candidate.z += dz[n];

            if (!inside(candidate)) {
                continue;
            }

            if (region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]) {
                continue;
            }

            if (similar(data, pixel, candidate)) {
                region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 2;
                *finished = 0;
            }
        }
    }
    */
}


unsigned char* grow_region_gpu(unsigned char* data){
    unsigned char *region = (unsigned char *)malloc(sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);

    dim3 gridBlock, threadBlock;

    gridBlock.x = 64;
    gridBlock.y = 64;
    gridBlock.z = 64;

    threadBlock.x = DATA_DIM / 64;
    threadBlock.y = DATA_DIM / 64;
    threadBlock.z = DATA_DIM / 64;

    region[index(300, 300, 50)] = 2;

    unsigned char *device_region;
    unsigned char *device_data;
    int *device_finished;

    cudaMalloc(&device_region, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);
    cudaMalloc(&device_data, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);
    cudaMalloc(&device_finished, sizeof(int));

    cudaMemcpy(device_data, data, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(device_region, region, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM, cudaMemcpyHostToDevice);

    int finished = 0;
    while (!finished) {
        finished = 1;

        cudaMemcpy(device_finished, &finished, sizeof(int), cudaMemcpyHostToDevice);
        region_grow_kernel<<<gridBlock, threadBlock>>>(device_data, device_region, device_finished);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(&finished, device_finished, sizeof(int), cudaMemcpyDeviceToHost);
        printf("iter\n");
    }

    cudaMemcpy(region, device_region, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM, cudaMemcpyDeviceToHost);

    return region;
}


unsigned char* grow_region_gpu_shared(unsigned char* data){
    unsigned char *region = (unsigned char *)malloc(sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);

    dim3 gridBlock, threadBlock;

    gridBlock.x = 32;
    gridBlock.y = 32;
    gridBlock.z = 32;

    threadBlock.x = DATA_DIM / 32 + 1;
    threadBlock.y = DATA_DIM / 32 + 1;
    threadBlock.z = DATA_DIM / 32 + 1;

    unsigned char *device_region;
    unsigned char *device_data;
    int *device_finished;

    cudaMalloc(&device_region, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);
    cudaMalloc(&device_data, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);
    cudaMalloc(&device_finished, sizeof(int));

    cudaMemcpy(device_data, data, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM, cudaMemcpyHostToDevice);

    int finished = 0;
    while (!finished) {
        finished = 1;

        cudaMemcpy(device_finished, &finished, sizeof(int), cudaMemcpyHostToDevice);
        region_grow_kernel_shared<<<gridBlock, threadBlock>>>(device_data, device_region, device_finished);
        cudaMemcpy(&finished, device_finished, sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(region, device_region, sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM, cudaMemcpyDeviceToHost);

    return region;
}

int main(int argc, char** argv){
    clock_t tic, toc;

    print_properties();

    unsigned char* data = create_data();

#if GROW_REGION_SERIAL
    tic = clock();
    unsigned char* region = grow_region_serial(data);
    toc = clock();

    printf("Grow region serial: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
#else
    tic = clock();
    unsigned char *region = grow_region_gpu(data);
    toc = clock();
    printf("Grow region gpu: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
#endif

#if RAYCAST_SERIAL
    tic = clock();
    unsigned char* image = raycast_serial(data, region);
    toc = clock();
    printf("Raycast serial: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
#else

#if RAYCAST_TEXTURE
    tic = clock();
    unsigned char *image = raycast_gpu_texture(data, region);
    toc = clock();
    printf("Raycast w/texture: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
#else
    tic = clock();
    unsigned char *image = raycast_gpu(data, region);
    toc = clock();
    printf("Raycast gpu: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
#endif

#endif


    write_bmp(image, IMAGE_DIM, IMAGE_DIM);
}
