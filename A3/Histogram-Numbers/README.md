# Histogram Numbers

Implementation of an efficient histogramming algorithm for an input array of integers within a given range.

Each integer will map into a single bin, so the values will range from 0 to (NUM_BINS - 1). The histogram bins will use unsigned 32-bit counters that must be saturated at 127 (i.e. no roll back to 0 allowed). The input length can be assumed to be less than 2^32 . NUM_BINS is fixed at 4096 for this question.

This can be split into two kernels: one that does a histogram without saturation, and a final kernel that cleans up the bins if they are too large. These two stages can also be combined into a single kernel

To compile and run:  
```
$ g++ dataset_generator.cpp  
$ ./a.out  
$ nvcc histogramNumbers.cu  
$ ./a.out input.raw output.raw  
```

Execution:

1. Exectution with one kernel (histogramNumber1.cu) 

```
[Generic] 0.253602048 Importing data and creating memory on host
Trace main::49 The input length is 500000
Trace main::50 The number of bins is 4096
[GPU    ] 0.000315136 Allocating GPU memory.
[GPU    ] 0.000579072 Copying input memory to the GPU.
Trace main::68 Launching kernel
[Compute] 0.000205056 Performing CUDA computation
[Copy   ] 0.000030208 Copying output memory to the CPU
[GPU    ] 0.000284160 Freeing GPU Memory
Solution is correct.
```

2. Execution with two kernals (histogramNumbers2.cu)

```
Trace main::56 The input length is 500000
Trace main::57 The number of bins is 4096
[GPU    ] 0.000280064 Allocating GPU memory.
[GPU    ] 0.000577792 Copying input memory to the GPU.
Trace main::75 Launching kernel
[Compute] 0.000176896 Performing CUDA computation
[Copy   ] 0.000028928 Copying output memory to the CPU
[GPU    ] 0.000290048 Freeing GPU Memory
Solution is correct.
```
