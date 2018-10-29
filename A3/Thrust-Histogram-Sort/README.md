# Thrust Histogram Sort

Implementation of a histogramming algorithm for an input array of integers. This approach
composes several distinct algorithmic steps to compute a histogram, which makes Thrust a valuable tools for its
implementation.

To compile and run:

```
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc thrustHistogramSort.cu
$ ./a.out input.raw output.raw
```

Execution:

```
[Generic] 0.001990144 Importing data and creating memory on host
Trace main::23 The input length is 4000
[GPU    ] 0.000204800 Allocating GPU memory
Solution is correct.
```
