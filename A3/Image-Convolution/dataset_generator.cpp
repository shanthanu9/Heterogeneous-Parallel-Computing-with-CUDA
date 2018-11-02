#include <bits/stdc++.h>
#include <sstream>
using namespace std;
#define CHANNELS 3

static char baseDir[200];

float clamp(float x) {
	return std::min(std::max(x, 0.0f), 1.0f);
}

unsigned char * compute(unsigned char *data, float *mask, int height, int width) {

	const int num_channels = 3;

	float inputData[height * width * num_channels];
	for(int i =0 ;i<height*width*num_channels;++i){
		inputData[i] = ((int)data[i])/255.0;
	}

	float *outputData = (float *) malloc(height*width*3*sizeof(float));

	int img_width  = width;
	int img_height = height;
	int mask_rows = 5;
	int mask_cols = 5;
	int mask_radius_y = mask_rows / 2; // 5 X 5 mask matrix is fixed
	int mask_radius_x = mask_cols / 2;

	for (int out_y = 0; out_y < img_height; ++out_y) {
		for (int out_x = 0; out_x < img_width; ++out_x) {
			for (int c = 0; c < num_channels; ++c) { // channels
				float acc = 0;
				for (int off_y = -mask_radius_y; off_y <= mask_radius_y; ++off_y) {
					for (int off_x = -mask_radius_x; off_x <= mask_radius_x; ++off_x) {

						int in_y   = out_y + off_y;
						int in_x   = out_x + off_x;
						int mask_y = mask_radius_y + off_y;
						int mask_x = mask_radius_x + off_x;
						if (in_y < img_height && in_y >= 0 && in_x < img_width && in_x >= 0) {
							acc += inputData[(in_y * img_width + in_x) * num_channels + c] *
								mask[mask_y * mask_cols + mask_x];
						}
					}
				}
				// fprintf(stderr, "%f %f\n", clamp(acc));
				outputData[(out_y * img_width + out_x) * num_channels + c] =
				clamp(acc);
			}
		}
	}
	
	unsigned char *output = (unsigned char *) malloc(height*width*3*sizeof(unsigned char));
	for(int i =0;i<height*width*num_channels;++i){
		output[i] = (unsigned char) floor(outputData[i] * 255);
	}
	return output;
}

float *generate_data_mask(const unsigned int y, const unsigned int x) {

	unsigned int i;
	const int maxVal = 5;
	float *data = (float *)malloc(y * x * sizeof(float));

	float *p = data;
	for (i = 0; i < y * x; ++i) {
		float r = (rand()%25 + 1.00)/125.00;
		*p++ = r;
	}
	return data;
}

static unsigned char *generate_data(const unsigned int y, const unsigned int x) {

	unsigned int i;

	const int maxVal = 256;
	unsigned char *data = (unsigned char *) malloc(y * x * 3*sizeof(unsigned char));

	unsigned char *p = data;
	for (i = 0; i < y * x; ++i) {
		unsigned char r = rand() % maxVal;
		unsigned char g = rand() % maxVal;
		unsigned char b = rand() % maxVal;
		*p++ = r;
		*p++ = g;
		*p++ = b;
	}
	return data;
}

static void write_data(const char *file_name, unsigned char *data,
	unsigned int width, unsigned int height, unsigned int channels) {

	FILE *handle = fopen(file_name, "w");
	fprintf(handle, "P6\n");
	fprintf(handle, "#Created by %s\n", __FILE__);
	fprintf(handle, "%d %d\n", width, height);
	fprintf(handle, "255\n");
	
	for(int i=0;i<width*height*channels;++i){
		fprintf(handle,"%u ",data[i]);
	}

	fflush(handle);
	fclose(handle);
}

static void write_data_mask(const char *file_name, float *data,
	unsigned int width, unsigned int height, unsigned int channels) {
	
	FILE *handle = fopen(file_name, "w");
	for(int j=0;j<height;++j){
		for(int i=0;i<width*channels;++i){
			fprintf(handle,"%f ",data[j*width + i]);
		}
	}  
	fflush(handle);
	fclose(handle);
}

void generate(int datasetNum, int height, int width, int minVal, int maxVal) {

	unsigned char * data = generate_data(height,width);

	float * data_mask = generate_data_mask(5,5);
	string sin,sout,sin2;
	sin = "input";
	sout = "output";
	sin2 = "mask";
	// sin.push_back(char(datasetNum + '0'));
	sin += ".ppm";
	// sin2.push_back(char(datasetNum + '0'));
	sin2 += ".raw";
	// sout.push_back(char(datasetNum + '0'));
	sout += ".ppm";

	const char *input_image_file_name = sin.c_str();
	const char *input_mask_file_name = sin2.c_str();
	const char *output_image_file_name = sout.c_str();

	write_data(input_image_file_name,data,width,height,3); 
	write_data_mask(input_mask_file_name,data_mask,5,5,1);

	unsigned char *ans = compute(data,data_mask,height,width);

	write_data(output_image_file_name,ans,width,height,3);
}

int main(void) {

	// generate(0, 64, 64, 0, 1);
	// generate(1, 128, 64, 0, 1);
	// generate(2, 64, 128, 0, 1);
	// generate(3, 64, 5, 0, 1);
	// generate(4, 64, 3, 0, 1);
	generate(5, 228, 128, 0, 1);//generate(5, 228, 128, 0, 1)
	// generate(6, 256, 256, 0, 1);
	// generate(7, 5, 5, 0, 1);
}
