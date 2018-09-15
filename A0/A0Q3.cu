#include <stdio.h>

const int ARRAY_SIZE = 1e6;
const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
const int BLOCK_SIZE = 1024;

__global__ void sumOfArray(int * d_sum, int * d_a) {
	__shared__ int s_data[BLOCK_SIZE];
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	s_data[threadIdx.x] = d_a[id];
	__syncthreads();
	
	for(int step = 1; step < BLOCK_SIZE; step *= 2) {
		int threadId = 2 * step * threadIdx.x;

		if(threadId + step< BLOCK_SIZE) {
			s_data[threadId] += s_data[threadId + step];
		}
			
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		atomicAdd(d_sum, s_data[0]);
	}
}

int main() {
	srand(time(NULL));

	int h_a[ARRAY_SIZE], *h_sum;
	h_sum = (int*) malloc(sizeof(int));
	for(int i = 0; i < ARRAY_SIZE; i++) {
		h_a[i] = rand();
	}

	int * d_a, * d_sum;

	cudaMalloc((void**) &d_a, ARRAY_BYTES);
	cudaMalloc((void**) &d_sum, sizeof(int));

	cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	float d_time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	sumOfArray<<<(int)ceil(ARRAY_SIZE/float(BLOCK_SIZE)), BLOCK_SIZE>>>(d_sum, d_a);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&d_time, start, stop);

	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

	int h_checksum = 0;
	
	clock_t tim;
	tim = clock();	

	for(int i = 0; i < ARRAY_SIZE; i++) {
		h_checksum += h_a[i];
	}
	
	tim = clock() - tim;
	float h_time = float(tim);
	
	
	if(h_checksum == *h_sum) {
		printf("The result is computed successfully!\n");
		printf("  Device sum = %d\n  Host sum   = %d\n\n", *h_sum, h_checksum);
		printf("Computation time by: \n");
		printf("  Device : %f ms\n", d_time);
		printf("  Host   : %f ms\n", h_time/CLOCKS_PER_SEC*1000);
	}
	else {
		printf("The computed result is incorrect!\n");
	}
	
	cudaFree(d_a);
	free(h_sum);

	return 0;
}
