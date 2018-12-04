#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define CEIL(a, b) ((a-1)/b +1)
#define Clamp(a, start, end) (max(min(a, end), start))
#define value(arry, i, j, k) (arry[((i)*width + (j)) * depth + (k)])
#define output(i, j, k) value(output, i, j, k)
#define input(i, j, k) value(input, i, j, k)
#define data(i, j, k) data[i*121 + j*11 + k]
#define BLOCK_SIZE 32

#define wbCheck(stmt)                                                           \
    do {                                                                        \
        cudaError_t err = stmt;                                                 \
        if (err != cudaSuccess) {                                               \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
            return -1;                                                          \
        }                                                                       \
    } while (0)


__global__ void launch_stencil(float *deviceOutputData, float *deviceInputData, 
    int width, int height, int depth) {

    #define in(i,j,k) value(deviceInputData, i, j, k)
    #define out(i,j,k) value(deviceOutputData, i, j, k)

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(j > 0 && j < width - 1 && i > 0 && i < height - 1)
        for(int k = 1; k < depth-1; k++) {
            float res = in(i, j, k + 1) + in(i, j, k - 1) + in(i, j + 1, k) +
                    in(i, j - 1, k) + in(i + 1, j, k) + in(i - 1, j, k) - 6 * in(i, j, k);
            res = Clamp(res, 0.0, 255.0);
            out(i, j, k) = res;
        }

    #undef in
    #undef out
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

    dim3 grid(CEIL(height, BLOCK_SIZE), CEIL(width, BLOCK_SIZE), 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

    launch_stencil<<<grid, block>>>(deviceOutputData, deviceInputData, width, height, depth);

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