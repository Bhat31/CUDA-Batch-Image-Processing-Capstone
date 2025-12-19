# GPU-Accelerated Batch Image Processing using CUDA

## Project Overview
This project demonstrates GPU-accelerated image processing by applying Sobel edge detection to batches of grayscale images using NVIDIA CUDA. Image processing is an ideal use case for GPUs because each pixel can be processed independently and in parallel.

## Motivation
CPU-based image processing becomes slow when working with large batches of images. GPUs provide massive parallelism, allowing thousands of pixels to be processed simultaneously, making them well suited for convolution-based operations such as edge detection.

## Project Description
The goal of this project was to design and implement a GPU-accelerated application that performs meaningful computation on large datasets while clearly demonstrating the benefits of GPU parallelism. Batch image processing using Sobel edge detection was chosen because it is a data-parallel workload where each output pixel depends only on a small neighborhood of input pixels, making it ideal for CUDA-based acceleration.

In real-world applications such as computer vision, medical imaging, and video analytics, edge detection is often applied to large collections of images. As dataset sizes increase, CPU-based implementations become a performance bottleneck, whereas GPUs are designed to efficiently handle such massively parallel workloads.

## Technologies Used
- NVIDIA CUDA
- CUDA Runtime API
- Custom CUDA kernels
- C++

## How GPU Acceleration Is Used
Each image pixel is mapped to a CUDA thread using a two-dimensional grid and block configuration. The Sobel operator computes horizontal and vertical gradients for each pixel, and the GPU executes thousands of these computations concurrently.

## Build Instructions
### Requirements
- NVIDIA GPU
- CUDA Toolkit installed

### Build
```bash
make
