# Thrust Transform
Implemention of vector addition using Thrust. 

>Thrust is a Standard Template Library for CUDA that contains a collection of data parallel primitives 
(eg. vectors) and implementations (eg. Sort, Scan, saxpy) that can be used in writing high performance 
CUDA code.

The executable generated as a result of compiling the lab can be run using the following command:

```
./ThrustVectorAdd_Template <expected.raw> <input0.raw> <input1.raw> <output.raw>
```

where <expected.raw> is the expected output, <input0.raw>, <input1.raw> is the input dataset, and <output.raw> is an optional 
path to store the results. The datasets can begenerated using the dataset generator.
