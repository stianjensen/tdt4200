#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

typedef struct{
    int n_row_ptr;
    int* row_ptr;
    int* col_ind;
    int n_values;
    float* values;
} csr_matrix_t;

typedef struct{
} s_matrix_t;

int diag_count(int dim, int n){
    return n*dim - ((n*(n+1))/2);
}

void print_raw_csr_matrix(csr_matrix_t* m){
    printf("row_ptr = {");
    for(int i = 0; i < m->n_row_ptr; i++)
        printf("%d ", m->row_ptr[i]);
    printf("}\n");

    printf("col_ind = {");
    for(int i = 0; i < m->n_values; i++)
        printf("%d ", m->col_ind[i]);
    printf("}\n");

    printf("values = {");
    for(int i = 0; i < m->n_values; i++)
        printf("%f ", m->values[i]);
    printf("}\n");
}

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
void print_formated_csr_matrix(csr_matrix_t* m){

    for(int i = 0; i < m->n_row_ptr-1; i++){
        
        int col = m->row_ptr[i];
        for(int j = 0; j < m->n_row_ptr-1; j++){
            if(j == m->col_ind[col] && col < m->row_ptr[i+1]){
                printf("%s%.2f ", KRED, m->values[col]);
                //printf("%.2f ", m->values[col]);
                col++;
            }
            else{
                printf("%s%.2f ", KNRM, 0.0);
                //printf("%.2f ", KNRM, 0.0);
            }
        }
        printf("%s\n", KNRM);
        //printf("\n");
    }
}

csr_matrix_t* create_csr_matrix(int n_rows, int n_cols, int a, int b, int c, int d, int e){

    csr_matrix_t* matrix = (csr_matrix_t*)malloc(sizeof(csr_matrix_t));

    matrix->row_ptr = (int*)malloc(sizeof(int) * (n_rows+1));
    matrix->n_row_ptr = n_rows+1;

    int ah = a/2;
    int size = diag_count(n_rows,ah);
    size += (diag_count(n_rows,ah+b+c) - diag_count(n_rows,ah+b));
    size += (diag_count(n_rows,ah+b+c+d+e) - diag_count(n_rows,ah+b+c+d));
    size = size*2 + n_rows;

    matrix->col_ind = (int*)malloc(sizeof(int)*size);
    matrix->values = (float*)malloc(sizeof(float)*size);
    matrix->n_values = size;

    int limits[10];
    limits[5] = ah;
    limits[6] = ah + b;
    limits[7] = ah + b + c;
    limits[8] = ah + b + c + d;
    limits[9] = ah + b + c + d + e;
    limits[0] = -limits[9];
    limits[1] = -limits[8];
    limits[2] = -limits[7];
    limits[3] = -limits[6];
    limits[4] = -limits[5];

    limits[5]++;
    limits[6]++;
    limits[7]++;
    limits[8]++;
    limits[9]++;

    int index = 0;
    int index2 = 0;
    int index3 = 0;
    matrix->row_ptr[0] = 0;
    for(int i = 0; i < n_rows; i++){

        int row_width = index;
        for(int j = fmax(0, limits[0]); j < fmax(0, limits[1]); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmax(0, limits[2]); j < fmax(0, limits[3]); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmax(0,limits[4]); j < fmin(limits[5], n_cols); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmin(n_cols, limits[6]); j < fmin(n_cols, limits[7]); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmin(n_cols, limits[8]); j < fmin(n_cols, limits[9]); j++)
            matrix->col_ind[index++] = j;

        row_width = index - row_width;
        matrix->row_ptr[index2+1] = matrix->row_ptr[index2] + row_width;
        index2++;
        
        for(int j = 0; j < row_width; j++)
            matrix->values[index3++] = (float)rand()/RAND_MAX;
        

        for(int j = 0; j < 10; j++)
            limits[j]++;
    }

    return matrix;
}

float* create_vector(int n){
    float* v = (float*)malloc(sizeof(float)*n);
    for(int i = 0; i < n; i++){
        v[i] = (float)rand()/RAND_MAX;
    }
    return v;
}

void print_vector(float* v, int n, int orientation){
    for(int i = 0; i < n; i++){
        printf("%f%s", v[i], orientation ? " " : "\n");
    }
    if(orientation)
        printf("\n");
}

void print_time(struct timeval start, struct timeval end){
    long int ms = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    double s = ms/1e6;
    printf("Time : %f s\n", s);
}

void multiply_naive(csr_matrix_t* m, float* v, float* r){
    for(int i = 0; i < m->n_row_ptr-1; i++){
        
        for(int j = m->row_ptr[i]; j < m->row_ptr[i+1]; j++){
            r[i] += v[m->col_ind[j]] * m->values[j];
        }
    }
}

void compare(float* a, float* b, int n){
    int n_errors = 0;
    for(int i = 0; i < n; i++){
        if(fabs(a[i] - b[i]) > 1e-4){
            n_errors++;
            if(n_errors < 10){
                printf("Error at: %d, expected: %f, actual: %f\n", i, a[i], b[i]);
            }
        }
    }
    printf("%d more errors...\n", n_errors - 10);
}


s_matrix_t* create_s_matrix(int dim, int a, int b, int c, int d, int e){
    return NULL;
}

s_matrix_t* convert_to_s_matrix(csr_matrix_t* csr){
    return NULL;
}

void multiply(s_matrix_t* m, float* v, float* r){
}


int main(int argc, char** argv){
    
    if(argc != 7){
        printf("useage %s dim a b c d e\n", argv[0]);
        exit(-1);
    }
    int dim = atoi(argv[1]);
    int a = atoi(argv[2]);
    int b = atoi(argv[3]);
    int c = atoi(argv[4]);
    int d = atoi(argv[5]);
    int e = atoi(argv[6]);

    csr_matrix_t* m = create_csr_matrix(dim,dim,a,b,c,d,e);
        
    float* v = create_vector(dim);
    float* r1 = (float*)calloc(dim, sizeof(float));
    float* r2 = (float*)calloc(dim, sizeof(float));
    
    struct timeval start, end;

    gettimeofday(&start, NULL);
    multiply_naive(m,v,r1);
    gettimeofday(&end, NULL);
    
    print_time(start, end);
    
    s_matrix_t* s = create_s_matrix(dim, a, b, c, d, e);
    //s_matrix_t* s = convert_to_s_matrix(m);
    
    gettimeofday(&start, NULL);
    multiply(s,v,r2);
    gettimeofday(&end, NULL);
    
    print_time(start, end);
    
    compare(r1,r2,dim);
}
