#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define wbCheck(stmt)                                               \
	do {                                                            \
		cudaError_t err = stmt;                                     \
		if (err != cudaSuccess) {                                   \
			wbLog(ERROR, "Failed to run stmt ", #stmt);             \
			return -1;                                              \
		}                                                           \
	} while (0)

#define Mask_width 5
#define Mask_radius (Mask_width / 2)
#define TILE_WIDTH 32
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))
#define CEIL(a, b) ((a-1)/b +1)

const int num_channels = 3;

__global__ void convolution (float *deviceInputImageData, float* __restrict__ deviceMaskData,
	float *deviceOutputImageData, int imageChannels, int imageWidth, int imageHeight) {

	int out_x =  blockDim.x * blockIdx.x + threadIdx.x;
	int out_y =  blockDim.y * blockIdx.y + threadIdx.y;

	if(out_x >= imageWidth || out_y >= imageHeight)
		return;

	for (int c = 0; c < num_channels; ++c) { // channels
		float acc = 0;
		for (int off_y = -Mask_radius; off_y <= Mask_radius; ++off_y) {
			for (int off_x = -Mask_radius; off_x <= Mask_radius; ++off_x) {

				int in_y   = out_y + off_y;
				int in_x   = out_x + off_x;
				int mask_y = Mask_radius + off_y;
				int mask_x = Mask_radius + off_x;
				if (in_y < imageHeight && in_y >= 0 && in_x < imageWidth && in_x >= 0) {
					acc += deviceInputImageData[(in_y * imageWidth + in_x) * num_channels + c] *
						deviceMaskData[mask_y * Mask_width + mask_x];
				}
			}
		}

		deviceOutputImageData[(out_y * imageWidth + out_x) * num_channels + c] = clamp(acc);
	}
}

int main(int argc, char *argv[]) {

	wbArg_t arg;
	int maskRows = Mask_width;
	int maskColumns = Mask_width;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile  = wbArg_getInputFile(arg, 1);

	inputImage   = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData  = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, 
		imageHeight * imageWidth * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, 
		imageHeight * imageWidth * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceMaskData, 
		Mask_width * Mask_width * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData, 
		imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutputImageData, hostOutputImageData,
		imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, 
		Mask_width * Mask_width * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	dim3 grid(CEIL(imageWidth, TILE_WIDTH), CEIL(imageHeight, TILE_WIDTH), 1);
	dim3 block(TILE_WIDTH, TILE_WIDTH, 1);

	convolution <<<grid, block>>> (deviceInputImageData, deviceMaskData,
		deviceOutputImageData, imageChannels, imageWidth, imageHeight);
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//@@ Insert code here
	free(hostInputImageData);
	free(hostOutputImageData);
	free(hostMaskData);

	cudaFree(deviceMaskData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceInputImageData);
}
