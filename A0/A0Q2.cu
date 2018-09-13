/*
 *To find sum of two large matrices
 *Compute the speed up obtained by GPU 
 */
#include <stdio.h>

const int ROW_SIZE = 300, COL_SIZE = 400;
const int MATRIX_BYTES = ROW_SIZE * COL_SIZE * sizeof(int);

//kernal
__global__ void sum(int d_sum[][ROW_SIZE], int d_a[][ROW_SIZE], int d_b[][ROW_SIZE]) {
        d_sum[blockIdx.x][threadIdx.x] = d_a[blockIdx.x][threadIdx.x] + d_b[blockIdx.x][threadIdx.x];
}

//to check the final result
int checkSum(int h_a[][ROW_SIZE], int h_b[][ROW_SIZE], int h_sum[][ROW_SIZE]) {
        int flag = 1;

    for(int i = 0; i < COL_SIZE; i++) {
        for(int j = 0; j < ROW_SIZE; j++) {
           if(h_sum[i][j] != h_a[i][j] + h_b[i][j]) {
                flag = 0;
                break;
           }
        }
    }

    return flag;
}

int main() {
    int h_a[COL_SIZE][ROW_SIZE], h_b[COL_SIZE][ROW_SIZE], h_sum[COL_SIZE][ROW_SIZE];

    for(int i = 0; i < COL_SIZE; i++) {
        for(int j = 0; j < ROW_SIZE; j++) {
           h_a[i][j] = ((int)rand())%1000;
           h_b[i][j] = ((int)rand())%1000;
        }
    }

    int (*d_a)[ROW_SIZE], (*d_b)[ROW_SIZE], (*d_sum)[ROW_SIZE];

    cudaMalloc((void**) &d_a, MATRIX_BYTES);
    cudaMalloc((void**) &d_b, MATRIX_BYTES);
    cudaMalloc((void**) &d_sum, MATRIX_BYTES);

    cudaMemcpy(d_a, h_a, MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, MATRIX_BYTES, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    sum<<<COL_SIZE, ROW_SIZE>>>(d_sum, d_a, d_b);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_sum, d_sum, MATRIX_BYTES, cudaMemcpyDeviceToHost);

    if(checkSum(h_a, h_b, h_sum)) {
        printf("The result is computed successfully!\n");
        
        cudaEventElapsedTime(&time, start, stop);
        printf("Computation time taken by device: %f\n", time);

        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        for(int i = 0; i < COL_SIZE; i++) {
            for(int j = 0; j < ROW_SIZE; j++) {
                h_sum[i][j] = h_a[i][j] + h_b[i][j];
            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);
        printf("Computation time taken by host: %f\n", time);
    }
    else {
        printf("The result is not computed correctly!");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sum);

    return 0;
}
