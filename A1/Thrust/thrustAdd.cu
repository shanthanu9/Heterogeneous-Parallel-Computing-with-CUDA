#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <bits/stdc++.h>
#include "wb.h"
using namespace std;

template <class T>
void testSolution(T *h_a, T *h_b, T *h_c, int n, float precision=0.0) {

	int errors = 0;
	for(int i=0; i<n; i++)
		if(abs(h_c[i] - h_a[i] - h_b[i]) > precision) {
			errors++;
			if(errors <= 10)
				printf("Test failed at index : %d\n", i);
		}

	if(errors)
		printf("\n%d Tests failed!\n\n", errors);
	else
		printf("All tests passed !\n\n");
}

int main(int argc, char *argv[]) {

	float *hostInput1 = NULL;
	float *hostInput2 = NULL;
	float *hostOutput = NULL;
	int inputLength;

	/* parse the input arguments */
	wbArg_t arguments = wbArg_read(argc, argv);
	char *output = wbArg_getInputFile(arguments, 0);
	char *input1 = wbArg_getInputFile(arguments, 1);
	char *input2 = wbArg_getInputFile(arguments, 2);

	// Import host input data
	ifstream ifile1(input1);
	ifstream ifile2(input2);

	ifile1 >> inputLength;
	ifile2 >> inputLength;

	printf("\nLength of vector : %d\n", inputLength);

	hostInput1 = new float[inputLength];
	hostInput2 = new float[inputLength];

	for(int i=0; i<inputLength; i++) {
		ifile1 >> hostInput1[i];
		ifile2 >> hostInput2[i];
	}

	// Allocate memory to host output
	hostOutput = new float[inputLength];

	// Declare and allocate thrust device input and output vectors and copy to device
	//@@ Insert Code here
	thrust::host_vector<float> h_input1(hostInput1, hostInput1+inputLength), h_input2(hostInput2, hostInput2+inputLength);
	thrust::device_vector<float> d_input1 = h_input1, d_input2 = h_input2, d_output;
	// Execute vector addition
	//@@ Insert Code here
	thrust::transform(d_input1.begin(), d_input1.end(), d_input2.begin(), d_output.begin(), thrust::plus<float>());
	/////////////////////////////////////////////////////////

	// Copy data back to host
	//@@ Insert Code here
	thrust::host_vector<float> h_output = d_output;
	thrust::copy(h_output.begin(), h_output.end(), hostOutput);

	testSolution(hostInput1, hostInput2, hostOutput, inputLength, 1e-6);

	delete[] hostInput1, hostInput2, hostOutput;
}
