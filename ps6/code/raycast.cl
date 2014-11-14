#define IMAGE_DIM 64
#define DATA_DIM 512

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

int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

int inside_float(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

    return x && y && z;
}

float value_at(float3 pos, __global unsigned char* data){
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

__kernel void raycast_kernel(__global unsigned char* data, __global unsigned char* image, __global unsigned char* region){
    int x = get_global_id(0) - IMAGE_DIM/2;
    int y = get_global_id(1) - IMAGE_DIM/2;

    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {1000, 1000, 1000};
    float3 forward = {-1, -1, -1};
    float3 z_axis = {0, 0, 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity length vectors
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
