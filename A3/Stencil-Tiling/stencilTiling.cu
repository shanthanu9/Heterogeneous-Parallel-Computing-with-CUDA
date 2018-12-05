#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define CEIL(a, b) ((a-1)/b +1)
#define Clamp(a, start, end) (max(min(a, end), start))
#define value(arry, i, j, k) (arry[((i)*width + (j)) * depth + (k)])
#define output(i, j, k) value(output, i, j, k)
#define input(i, j, k) value(input, i, j, k)


#define wbCheck(stmt)                                                           \
    do {                                                                        \
        cudaError_t err = stmt;                                                 \
        if (err != cudaSuccess) {                                               \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
            return -1;                                                          \
        }                                                                       \
    } while (0)

__global__ void compute_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth) {

	#define in(i, j, k) value(deviceInputData, i, j, k)
	#define out(i, j, k) value(deviceOutputData, i, j, k)

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	__shared__ float data[11][11][11];

	int x = threadIdx.x; 	//corresponds to i
	int y = threadIdx.y;	//corresponds to j
	int z = threadIdx.z;	//corresponds to k

	if(i<height && j<width && k<depth) {
        data[x+1][y+1][z+1] = in(i,j,k);
    }

    __syncthreads();

    if(i>0 && x == 0) {
        data[x][y+1][z+1] = in(i-1,j,k);
    }

    if(i<height-1 && x == 8) {
        data[x+2][y+1][z+1] = in(i+1,j,k);
    }

    if(j>0 && y == 0) {
        data[x+1][y][z+1] = in(i,j-1,k);
    }

    if(j<width-1 && y == 8) {
        data[x+1][y+2][z+1] = in(i,j+1,k);
    }

    if(k>0 && z == 0) {
        data[x+1][y+1][z] = in(i,j,k-1);
    }

    if(k<depth-1 && z == 8) {
        data[x+1][y+1][z+2] = in(i,j,k+1);
    }

    __syncthreads();

    if(i > 0 && i < height-1 && j > 0 && j < width-1 && k > 0 && k < depth-1) {

        x++;y++;z++;

		float res = data[x-1][y][z] + data[x+1][y][z] + data[x][y-1][z] +
                    data[x][y+1][z] + data[x][y][z-1] + data[x][y][z+1] -
                    6*data[x][y][z];

		out(i,j,k) = Clamp(res, 0.0, 1.0);

	}

    #undef in
    #undef out

}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, 
    int width, int height, int depth) {

    dim3 grid(CEIL(height, 9), CEIL(width, 9), CEIL(depth, 9));
    dim3 block(9, 9, 9);

    compute_stencil<<<grid, block>>>(deviceOutputData, deviceInputData, width, height, depth);
}

int main(int argc, char *argv[]) {

    wbArg_t arg;
    int width;
    int height;
    int depth;
    char *inputFile;
    wbImage_t input;
    wbImage_t output;
    float *hostInputData;
    float *deviceInputData;
    float *deviceOutputData;

    arg = wbArg_read(argc, argv);

    inputFile = wbArg_getInputFile(arg, 0);
    input = wbImport(inputFile);

    width  = wbImage_getWidth(input);
    height = wbImage_getHeight(input);
    depth  = wbImage_getChannels(input);

    output = wbImage_new(width, height, depth);

    hostInputData  = wbImage_getData(input);

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
    cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float),
        cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(output.data, deviceOutputData, width * height * depth * sizeof(float),
        cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    
    wbSolution(arg, output);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    wbImage_delete(output);
    wbImage_delete(input);
}