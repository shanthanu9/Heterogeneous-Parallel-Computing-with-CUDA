//Matrix Multiplication
#include <stdio.h>
#define BLOCK_SIZE 32
#define CEIL(a, b) ((a-1)/b +1)
#define COMMON_SIZE 700
const int WIDTH1 = COMMON_SIZE, HEIGHT1 = 800, WIDTH2 = 900, HEIGHT2 = COMMON_SIZE;

__global__ void matrix_multiplicaton(int *d_out, int *d_in1, int *d_in2) {
	int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	if(i >= HEIGHT1 || j >= WIDTH2)
	    return;


	int sum = 0;
	int I = i * WIDTH1, J = j;
	for(int k = 0; k < COMMON_SIZE; k++) {
		sum += d_in1[I] * d_in2[J];
		I += 1;
		J += WIDTH2;
	}

	d_out[i*WIDTH2+j] = sum;
}

bool test_solution(int *h_out, int *h_in1, int *h_in2) {
	bool flag = true;
	for(int i = 0; i < HEIGHT1; i++) {
		for(int j = 0; j < WIDTH2; j++) {
			int sum = 0;
			for(int k = 0; k < COMMON_SIZE; k++) {
				sum += (h_in1[i*WIDTH1+k] * h_in2[k*WIDTH2+j]);
			}
			if(sum != h_out[i*WIDTH2+j]) {
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

	int h_in1[WIDTH1 * HEIGHT1], h_in2[WIDTH2 * HEIGHT2], h_out[WIDTH2 * HEIGHT1];

	for(int i = 0; i < WIDTH1 * HEIGHT1; i++) {
        h_in1[i] = rand();
    }


	for(int i = 0; i < WIDTH2 * HEIGHT2; i++) {
		h_in2[i] = rand();
	}

	int *d_in1, *d_in2, *d_out;

	cudaMalloc(&d_in1, WIDTH1 * HEIGHT1 * sizeof(int));
	cudaMalloc(&d_in2, WIDTH2 * HEIGHT2 * sizeof(int));
	cudaMalloc(&d_out, WIDTH2 * HEIGHT1 * sizeof(int));

	cudaMemcpy(d_in1, h_in1, WIDTH1 * HEIGHT1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2, h_in2, WIDTH2 * HEIGHT2 * sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	matrix_multiplicaton<<<dim3(CEIL(WIDTH2, BLOCK_SIZE), CEIL(HEIGHT1, BLOCK_SIZE), 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(d_out, d_in1, d_in2);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(h_out, d_out, WIDTH2 * HEIGHT1 * sizeof(int), cudaMemcpyDeviceToHost);

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
