# Histogram ASCII

Implementation of an efficient histogram algorithm for an input array of ASCII characters. There are 128 ASCII characters
and each character will map into its own bin for a fixed total of 128 bins. The histogram bins will be unsigned 32-bit
counters that do not saturate. Used the approach of creating a privatized histogram in shared memory for each thread block,
then atomically modifying the global histogram.

To compile and run:
```
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc histogramASCII.cu
$ ./a.out input.txt output.raw
```

Execution:

```
Trace main::64 The input length is 512
Trace main::65 The number of bins is 128
[GPU    ] 0.000267008 Allocating GPU memory.
[GPU    ] 0.000035840 Copying input memory to the GPU.
Trace main::84 Launching kernel
[Compute] 0.000093952 Performing CUDA computation
[Copy   ] 0.000024064 Copying output memory to the CPU
[GPU    ] 0.000284160 Freeing GPU Memory
Solution is correct.
```
