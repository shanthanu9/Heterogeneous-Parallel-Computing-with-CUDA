# Parallel image blurring algorithm

* An image is represented as ```RGB float``` values. 
* Operated directly on the RGB float values and use a 5x5 Box Filter to blur the original image to produce the blurred image (Gaussian Blur).
* The executable generated as a result of compiling the lab can be run using the following command:
    > ```./ImageBlur_Template ­ <expected.ppm> ­<input.ppm> <output.ppm>```   

    where <expected.ppm> is the expected output, <input.ppm> is the input dataset, and <output.ppm> is an optional path to store the results.
* The datasets can be generated using the dataset generator built as part of the compilation process.
