#include "wb.h"
#include <bits/stdc++.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
using namespace std;

int main(int argc, char *argv[]) {

	wbArg_t args;
	int inputLength, num_bins;
	unsigned int *hostInput, *hostBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);

	// Copy the input to the GPU
	wbTime_start(GPU, "Allocating GPU memory");
	
	thrust::device_vector<unsigned int> deviceInput(hostInput, hostInput + inputLength); 
	//copy(hostInput, hostInput + inputLength, deviceInput);

	wbTime_stop(GPU, "Allocating GPU memory");

	// Determine the number of bins (num_bins) and create space on the host
	thrust::sort(deviceInput.begin(), deviceInput.end());

	num_bins = deviceInput.back() + 1;
	hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

	// Allocate a device vector for the appropriate number of bins
	thrust::device_vector<unsigned int> histogram(num_bins);

	// Create a cumulative histogram. Use thrust::counting_iterator and
	// thrust::upper_bound
	thrust::counting_iterator<int> it(0);
	thrust::upper_bound(deviceInput.begin(), deviceInput.end(), it, it + num_bins, histogram.begin());
	

	// Use thrust::adjacent_difference to turn the culumative histogram into a histogram.
	thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());

	// Copy the histogram to the host
	thrust::copy(histogram.begin(), histogram.end(), hostBins);

	// Check the solution is correct
	wbSolution(args, hostBins, num_bins);

	// Free space on the host
	free(hostInput);
	free(hostBins);
}

