//Tiled Matrix Multiplication
#include <stdio.h>
#define BLOCK_SIZE 32
#define CEIL(a, b) ((a-1)/b +1)
#define TILE 32
const int N = 10000;

__global__ void matrix_multiplicaton(int *d_out, int *d_in1, int *d_in2) {
	__shared__ int in_tile1[TILE][TILE], in_tile2[TILE][TILE];

	int cj = blockIdx.x * blockDim.x + threadIdx.x;
	int ci = blockIdx.y * blockDim.y + threadIdx.y;

	int c = 0;

	for(int k = 0; k < N; k += BLOCK_SIZE) {

		int i = ci;
		int j = k + threadIdx.x;

		if(i < N && j < N) {
			in_tile1[threadIdx.y][threadIdx.x] = d_in1[i*N+j];
		}
		else {
			in_tile1[threadIdx.y][threadIdx.x] = 0;
		}

		i = k + threadIdx.y;
		j = cj;

		if(i < N && j < N) {
			in_tile2[threadIdx.y][threadIdx.x] = d_in2[i*N+j];
		}
		else {
			in_tile2[threadIdx.y][threadIdx.x] = 0;
		}

		__syncthreads();

		for(int l = 0; l < BLOCK_SIZE; l++) {
			c += (in_tile1[threadIdx.y][l] * in_tile2[l][threadIdx.x]);
		}
		
		__syncthreads();
	}

	if(ci < N && cj < N) {
		d_out[ci*N+cj] = c;
	}

}

bool test_solution(int *h_out, int *h_in1, int *h_in2) {
	bool flag = true;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			int sum = 0;
			for(int k = 0; k < N; k++) {
				sum += (h_in1[i*N+k] * h_in2[k*N+j]);
			}
			if(sum != h_out[i*N+j]) {
				flag = false;
				break;
			}	
		}
	}
	if(flag) return true;
	return false;
}

int main() {
	srand(time(0));

	int * h_in1, * h_in2, * h_out;

	h_in1 = (int *)malloc(N*N*sizeof(int));
        h_in2 = (int *)malloc(N*N*sizeof(int));
 	h_out = (int *)malloc(N*N*sizeof(int));

	for(int i = 0; i < N * N; i++) {
        h_in1[i] = rand()%1000;
        }


	for(int i = 0; i < N * N; i++) {
		h_in2[i] = rand()%1000;
	}

	int *d_in1, *d_in2, *d_out;

	cudaMalloc(&d_in1, N * N * sizeof(int));
	cudaMalloc(&d_in2, N * N * sizeof(int));
	cudaMalloc(&d_out, N * N * sizeof(int));

	cudaMemcpy(d_in1, h_in1, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, h_in2, N * N * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	
	dim3 grid(CEIL(N, BLOCK_SIZE), CEIL(N, BLOCK_SIZE), 1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

	matrix_multiplicaton<<<grid, block>>>(d_out, d_in1, d_in2);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(h_out, d_out, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime=0;
	cpu_startTime = clock();
	
	bool flag = test_solution(h_out, h_in1, h_in2);

	cpu_endTime = clock();
	cpu_ElapseTime = ((cpu_endTime - cpu_startTime)/(1.0 * CLOCKS_PER_SEC)) * 1000;

	if(flag) {
		printf("The computed matrix is correct!\n");
		printf("Time taken by GPU : %f ms\n", milliseconds);
		printf("Time taken by CPU : %f ms\n", cpu_ElapseTime);
	}
	else {
		printf("The computed matrix is incorrect!\n");
	}

	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out);

	return 0;
}

