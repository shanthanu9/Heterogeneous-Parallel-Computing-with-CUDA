#include <bits/stdc++.h>
#include "wb.h"
using namespace std;

#define BLUR_SIZE 5
#define CHANNELS 3
#define CEIL(a, b) ((a-1)/b +1)

//@@ INSERT CODE HERE
__global__
void compute(float * deviceOutputImageData, float * deviceInputImageData, int imageHeight, int imageWidth) {
	int y = blockDim.y * blockIdx.y + threadIdx.y, x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x < imageWidth && y < imageHeight)
	for(int channel = 0; channel < CHANNELS; channel++) {
		float value = 0;
		int pixel = 0;
		for(int i = x - BLUR_SIZE; i <= x + BLUR_SIZE; i++) {
			for(int j = y - BLUR_SIZE; j <= y + BLUR_SIZE; j++) {
				if((i >= 0 && j >= 0) && (i < imageWidth && j < imageHeight)) {
					pixel++;
					value += deviceInputImageData[CHANNELS*(j*imageWidth+i) + channel];
				}
			}
		}
		value /= pixel;
		deviceOutputImageData[CHANNELS*(imageWidth*y+x)+channel] = value;
	}
}

int main(int argc, char *argv[]) {

	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *deviceInputImageData;
	float *deviceOutputImageData;


	/* parse the input arguments */
	wbArg_t args = wbArg_read(argc, argv);
	inputImageFile = wbArg_getInputFile(args, 0);

	inputImage = wbImport(inputImageFile);

	imageWidth  = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, CHANNELS);

	hostInputImageData  = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	// Allocate data
	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * CHANNELS * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * CHANNELS * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	// Copy data
	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
		imageWidth * imageHeight * CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");

	// Kernel call
	//@@ Insert Code here
	compute<<<dim3(CEIL(imageHeight, 32), CEIL(imageWidth, 32), 1), dim3(32, 32, 1)>>>(deviceOutputImageData, deviceInputImageData, imageHeight, imageWidth);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	// Copy data back
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * CHANNELS * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	// Check solution
	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);
}

