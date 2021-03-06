#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>
#include "clutil.h"

#include "bmp.h"

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 64

typedef struct{
    float x;
    float y;
    float z;
} float3;

typedef struct{
    int x;
    int y;
    int z;
} int3;


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
float3 cross(float3 a, float3 b){
    float3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

float3 normalize(float3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;

    return v;
}

float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}

float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;

    return a;
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
int inside_float(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

    return x && y && z;
}

int inside_int(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);

    return x && y && z;
}

// Indexing function (note the argument order)
int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Trilinear interpolation
float value_at(float3 pos, unsigned char* data){
    if(!inside_float(pos)){
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
int similar(unsigned char* data, int3 a, int3 b){
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

            if(!inside_int(candidate)){
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

unsigned char* grow_region_gpu(unsigned char* data){
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_int err;
    char *source;
    int i;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    printPlatformInfo(platform);
    printDeviceInfo(device);

    queue = clCreateCommandQueue(context, device, 0, &err);
    kernel = buildKernel("region.cl", "region_grow_kernel", NULL, context, device);

    unsigned char *region = (unsigned char *)malloc(sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM);

    region[index(300, 300, 50)] = 2;

    cl_mem device_region = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM,
            NULL,
            &err
            );
    cl_mem device_data = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM,
            NULL,
            &err
            );
    cl_mem device_finished = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(int),
            NULL,
            &err
            );

    err = clEnqueueWriteBuffer(
            queue,
            device_region,
            CL_TRUE,
            0,
            sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM,
            region,
            0,
            NULL,
            NULL
            );
    err = clEnqueueWriteBuffer(
            queue,
            device_data,
            CL_TRUE,
            0,
            sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM,
            data,
            0,
            NULL,
            NULL
            );

    int finished = 0;
    while (!finished) {
        finished = 1;

        err = clEnqueueWriteBuffer(
                queue,
                device_finished,
                CL_TRUE,
                0,
                sizeof(int),
                &finished,
                0,
                NULL,
                NULL
                );

        err = clSetKernelArg(kernel, 0, sizeof(device_data), &device_data);
        err = clSetKernelArg(kernel, 1, sizeof(device_region), &device_region);
        err = clSetKernelArg(kernel, 2, sizeof(device_finished), &device_finished);
        clError("error setting arguments", err);

        const size_t size = DATA_DIM*DATA_DIM*DATA_DIM;
        err = clEnqueueNDRangeKernel(queue, kernel,
                1, NULL, &size, NULL, 0, NULL, NULL);
        clError("Error queueing kernel", err);

        err = clFinish(queue);
        clError("Error finishing", err);


        err = clEnqueueReadBuffer(queue, device_finished, CL_TRUE,
                0, sizeof(int), &finished, 0, NULL, NULL);

        clFinish(queue);
    }

    err = clEnqueueReadBuffer(
            queue,
            device_region,
            CL_TRUE,
            0,
            sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM,
            region,
            0,
            NULL,
            NULL
            );

    clReleaseMemObject(device_region);
    clReleaseMemObject(device_data);
    clReleaseMemObject(device_finished);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return region;

}

unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_int err;
    char *source;
    int i;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    printPlatformInfo(platform);
    printDeviceInfo(device);

    queue = clCreateCommandQueue(context, device, 0, &err);
    kernel = buildKernel("raycast.cl", "raycast_kernel", NULL, context, device);

    cl_mem device_image = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(cl_uchar) * IMAGE_DIM * IMAGE_DIM,
            NULL,
            &err
            );
    clError("create device image", err);
    cl_mem device_data = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(cl_uchar) * DATA_DIM * DATA_DIM * DATA_DIM,
            NULL,
            &err
            );
    clError("create device data", err);
    cl_mem device_region = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(cl_uchar) * DATA_DIM * DATA_DIM * DATA_DIM,
            NULL,
            &err
            );
    clError("create device region", err);

    err = clEnqueueWriteBuffer(
            queue,
            device_data,
            CL_TRUE,
            0,
            sizeof(cl_uchar) * DATA_DIM * DATA_DIM * DATA_DIM,
            data,
            0,
            NULL,
            NULL
            );
    clError("write device data", err);
    err = clEnqueueWriteBuffer(
            queue,
            device_region,
            CL_TRUE,
            0,
            sizeof(cl_uchar) * DATA_DIM * DATA_DIM * DATA_DIM,
            region,
            0,
            NULL,
            NULL
            );
    clError("write device region", err);

    err = clSetKernelArg(kernel, 0, sizeof(device_data), &device_data);
    clError("error setting arguments", err);
    err = clSetKernelArg(kernel, 1, sizeof(device_image), &device_image);
    clError("error setting arguments", err);
    err = clSetKernelArg(kernel, 2, sizeof(device_region), &device_region);
    clError("error setting arguments", err);

    const size_t global_size[] = {IMAGE_DIM, IMAGE_DIM};
    const size_t local_size[] = {8, 8};
    err = clEnqueueNDRangeKernel(queue, kernel,
            2, NULL, global_size, local_size, 0, NULL, NULL);
    clError("Error queueing kernel", err);

    err = clFinish(queue);
    clError("Error finishing", err);


    unsigned char *image = (unsigned char *)malloc(sizeof(unsigned char) * IMAGE_DIM * IMAGE_DIM);
    err = clEnqueueReadBuffer(queue, device_image, CL_TRUE,
            0, sizeof(cl_uchar) * IMAGE_DIM * IMAGE_DIM, image, 0, NULL, NULL);

    clError("reading buffer", err);

    clReleaseMemObject(device_region);
    clReleaseMemObject(device_data);
    clReleaseMemObject(device_image);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return image;
}


int main(int argc, char** argv){

    unsigned char* data = create_data();

    //unsigned char* region = grow_region_serial(data);

    unsigned char* region = grow_region_gpu(data);

    //unsigned char* image = raycast_serial(data, region);

    unsigned char* image = raycast_gpu(data, region);

    write_bmp(image, IMAGE_DIM, IMAGE_DIM);
}
