# HoG_parallel

Parallelised HoG feature detection algorithm for reverse image search
Using CUDA, C++, OpenMP

How to run:
Serial version - g++ hog.cpp -o output1 -w $(pkg-config --libs --cflags opencv4)
CUDA version - nvcc hog.cu -o output -w $(pkg-config --libs --cflags opencv4)
