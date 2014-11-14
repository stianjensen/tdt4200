#define DATA_DIM 512

/*
typedef struct{
    int x;
    int y;
    int z;
} int3;
*/

int index(int z, int y, int x){
    return z * 512 * 512 + y * 512 + x;
}

// Check if two values are similar, threshold can be changed.
int similar(__global unsigned char* data, int3 a, int3 b){
    unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
    unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];

    int i = abs(va-vb) < 1;
    return i;
}

int inside_int(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);

    return x && y && z;
}

__kernel void region_grow_kernel(__global unsigned char* data, __global unsigned char* region, __global int* finished){
    int threadId = get_global_id(0);
    int threadX = threadId % DATA_DIM;
    int threadY = threadId/DATA_DIM % DATA_DIM;
    int threadZ = threadId / DATA_DIM / DATA_DIM;

    if (region[threadId] == 2) {
        region[threadId] = 1;

        int dx[6] = {-1,1,0,0,0,0};
        int dy[6] = {0,0,-1,1,0,0};
        int dz[6] = {0,0,0,0,-1,1};

        int3 pixel = {threadX, threadY, threadZ};

        for (int n = 0; n < 6; n++) {
            int3 candidate = pixel;
            candidate.x += dx[n];
            candidate.y += dy[n];
            candidate.z += dz[n];

            if (!inside_int(candidate)) {
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
