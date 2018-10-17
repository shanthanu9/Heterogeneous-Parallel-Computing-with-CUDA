#include <bits/stdc++.h>
#include "wb.h"
using namespace std;

#define CEIL(a, b) ((a-1)/b +1)
#define BLOCK_SIZE 1024

__global__ void compute(float * deviceOutputImageData, float * deviceInputImageData) {
	int Id = blockDim.x * blockIdx.x + threadIdx.x;
	float r = deviceInputImageData[3*Id];
	float g = deviceInputImageData[3*Id+1];
	float b = deviceInputImageData[3*Id+2];
	deviceOutputImageData[Id] = (0.21f * r + 0.71f * g + 0.07f * b);
} 

int main(int argc, char *argv[]) {

	int imageChannels;
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
	imageChannels = wbImage_getChannels(inputImage); // For this lab the value is always 3

	// Since the image is monochromatic, it only contains one channel
	outputImage = wbImage_new(imageWidth, imageHeight, 1);

	hostInputImageData  = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	cudaMalloc((void **)&deviceInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData,
		imageWidth * imageHeight * sizeof(float));
	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	cudaMemcpy(deviceInputImageData, hostInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Compute, "Doing the computation on the GPU");

	compute<<<CEIL(imageWidth*imageHeight, BLOCK_SIZE), BLOCK_SIZE>>>(deviceOutputImageData, deviceInputImageData);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	///////////////////////////////////////////////////////
	wbTime_start(Copy, "Copying data from the GPU");
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(args, outputImage);

	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
