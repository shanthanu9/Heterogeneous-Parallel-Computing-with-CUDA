# Image Convolution

Implementation of tiled image convolution using both shared and constant memory. A constant
5x5 convolution mask is used.

To compile and run:  

```
$ g++ dataset_generator.cpp  
$ ./a.out  
$ nvcc imageConvolution.cu
$ ./a.out input.ppm mask.raw output.ppm
```

Execution:

```
[GPU    ] 0.000262912 Doing GPU memory allocation
[Copy   ] 0.000050944 Copying data to the GPU
[Compute] 0.000167936 Doing the computation on the GPU
[Copy   ] 0.000034048 Copying data from the GPU
[GPU    ] 0.000802048 Doing GPU Computation (memory + compute)
Solution is correct.
```
