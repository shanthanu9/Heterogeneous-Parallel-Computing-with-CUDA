#include <stdio.h>

const long long ARRAY_SIZE = 320000; 
const long long ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

//kernal
__global__ void sum(float *d_sum, float *d_a, float *d_b) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	d_sum[id] = d_a[id] + d_b[id];
}

//to check the final result
int checkSum(float *h_a, float *h_b, float *h_sum) {
	int flag = 1;

    for(int i = 0; i < ARRAY_SIZE; i++) {
    	if(h_sum[i] != h_a[i] + h_b[i]) {
    		flag = 0;
    	}
    }

    return flag;
}

int main() {
    float h_a[ARRAY_SIZE], h_b[ARRAY_SIZE], h_sum[ARRAY_SIZE];

    for(int i = 0; i < ARRAY_SIZE; i++) {
    	h_a[i] = rand();
    	h_b[i] = rand();
    }
    	
    float *d_a, *d_b, *d_sum;

    cudaMalloc((void**) &d_a, ARRAY_BYTES);
    cudaMalloc((void**) &d_b, ARRAY_BYTES);
    cudaMalloc((void**) &d_sum, ARRAY_BYTES);

    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, ARRAY_BYTES, cudaMemcpyHostToDevice);

    sum<<<(int)ceil(ARRAY_SIZE/32.0), 32>>>(d_sum, d_a, d_b);

    cudaMemcpy(h_sum, d_sum, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    if(checkSum(h_a, h_b, h_sum)) {
    	printf("The result is computed correctly!");
    }
    else {
    	printf("The result is not computed correctly!");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sum);

    return 0;
}