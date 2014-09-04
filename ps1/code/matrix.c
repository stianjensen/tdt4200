#include <stdio.h>
#include <stdlib.h>

typedef struct{
    int rows;
    int cols;
    float *array;
} matrix_t;


float get_value(matrix_t* matrix, int row, int col);
void set_value(matrix_t* matrix, int row, int col, float value);


matrix_t* new_matrix(int rows, int cols){
    matrix_t *matrix = malloc(sizeof(matrix_t));

    matrix->rows = rows;
    matrix->cols = cols;

    float *array = malloc(sizeof(float) * rows * cols);
    matrix->array = array;

    return matrix;
}


void print_matrix(matrix_t* matrix){
    for (int row = 0; row < matrix->rows; row++) {
        for (int col = 0; col < matrix->cols; col++) {
            printf("%.2f ", get_value(matrix, row, col));
        }
        printf("\n");
    }
}


void set_value(matrix_t* matrix, int row, int col, float value){
    int index = row * matrix->cols + col;
    matrix->array[index] = value;
}


float get_value(matrix_t* matrix, int row, int col){
    int index = row * matrix->cols + col;
    return matrix->array[index];
}


int is_sparse(matrix_t matrix, float sparse_threshold){
    int n_zero_elements = 0;
    int n_non_zero_elements = 0;
    for (int i = 0; i < matrix.rows * matrix.cols; i++) {
        if (matrix.array[i] == 0) {
            n_zero_elements++;
        } else {
            n_non_zero_elements++;
        }
    }
    return n_zero_elements / n_non_zero_elements > sparse_threshold;
}
      

int matrix_multiply(matrix_t* a, matrix_t* b, matrix_t** c){
    *c = new_matrix(a->rows, b->cols);
    if (a->cols != b->rows) {
        for (int i=0; i < a->rows * b->cols; i++) {
            (*c)->array[i] = 0;
        }
        return -1;
    }
    for (int row = 0; row < a->rows; row++) {
        for (int col = 0; col < b->cols; col++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += get_value(a, row, k) * get_value(b, k, col);
            }
            set_value(*c, row, col, sum);
        }
    }
    return 0;
}


void change_size(matrix_t* matrix, int new_rows, int new_cols){
    float *new_array = malloc(sizeof(float) * new_rows * new_cols);
    for (int row = 0; row < new_rows; row++) {
        for (int col = 0; col < new_cols; col++) {
            int index = row * new_cols + col;
            if (row < matrix->rows && col < matrix->cols) {
                new_array[index] = get_value(matrix, row, col);
            } else {
                new_array[index] = 0;
            }
        }
    }
    free(matrix->array);
    matrix->rows = new_rows;
    matrix->cols = new_cols;
    matrix->array = new_array;
}


void free_matrix(matrix_t* matrix){
    free(matrix->array);
    free(matrix);
}
        

int main(int argc, char** argv){
  
  // Create and fill matrix m
  matrix_t* m = new_matrix(3,4);
  for(int row = 0; row < 3; row++){
    for(int col = 0; col < 4; col++){
      set_value(m, row, col, row*10+col);
    }
  }
  
  // Create and fill matrix n
  matrix_t* n = new_matrix(4,4);
  for(int row = 0; row < 4; row++){
    for(int col = 0; col < 4; col++){
      set_value(n, row, col, col*10+row);
    }
  }
  
  // Create and fill matrix o
  matrix_t* o = new_matrix(5,5);
  for(int row = 0; row < 5; row++){
    for(int col = 0; col < 5; col++){
      set_value(o, row, col, row==col? 1 : 0);
    }
  }
  // Printing matrices
  printf("Matrix m:\n");
  print_matrix(m);
  /*
  Should print:
  0.00 1.00 2.00 3.00 
  10.00 11.00 12.00 13.00 
  20.00 21.00 22.00 23.00
  */
  
  printf("Matrix n:\n");
  print_matrix(n);
  /*
  Should print:
  0.00 10.00 20.00 30.00 
  1.00 11.00 21.00 31.00 
  2.00 12.00 22.00 32.00 
  3.00 13.00 23.00 33.00 
  */
  
  
  printf("Matrix o:\n");
  print_matrix(o);
  /*
  Should print:
  1.00 0.00 0.00 0.00 0.00 
  0.00 1.00 0.00 0.00 0.00 
  0.00 0.00 1.00 0.00 0.00 
  0.00 0.00 0.00 1.00 0.00 
  0.00 0.00 0.00 0.00 1.00
  */
  
  // Checking if matrices are sparse (more than 75% 0s)
  printf("Matrix m is sparse: %d\n", is_sparse(*m, 0.75)); // Not sparse, should print 0
  printf("Matrix o is sparse: %d\n", is_sparse(*o, 0.75)); // Sparse, should print 1
  
  // Attempting to multiply m and o, should not work
  matrix_t* p;
  int error = matrix_multiply(m,o,&p);
  printf("Error (m*o): %d\n", error); // Should print -1 

  // Attempting to multiply m and n, should work
  error = matrix_multiply(m,n,&p);
  print_matrix(p);
  /*
  Should print:
  14.00 74.00 134.00 194.00 
  74.00 534.00 994.00 1454.00 
  134.00 994.00 1854.00 2714.00 
  */
  
  // Shrinking m, expanding n
  change_size(m, 2,2);
  change_size(n, 5,5);
  
  printf("Matrix m:\n");
  print_matrix(m);
  /*
  Should print:
  0.00 1.00 
  10.00 11.00 
  */
  printf("Matrix n:\n");
  print_matrix(n);
  /*
  Should print:
  0.00 10.00 20.00 30.00 0.00 
  1.00 11.00 21.00 31.00 0.00 
  2.00 12.00 22.00 32.00 0.00 
  3.00 13.00 23.00 33.00 0.00 
  0.00 0.00 0.00 0.00 0.00
  */
  
  // Freeing memory
  free_matrix(m);
  free_matrix(n);
  free_matrix(o);
  free_matrix(p);
}
