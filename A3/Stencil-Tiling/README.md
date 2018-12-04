# Stencil Tiling

Shared-memory tiling by implementing a 7-point stencil.

To compile and run:

```
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc stencilTiling.cu
$ ./a.out input.ppm output.ppm
```

Execution:

```
[GPU    ] 0.000468992 Doing GPU memory allocation
[Copy   ] 0.006686976 Copying data to the GPU
[Compute] 0.004561152 Doing the computation on the GPU
[Copy   ] 0.028345856 Copying data from the GPU
Solution is correct!
```

