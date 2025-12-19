# GPU-Accelerated Batch Image Processing using CUDA and NPP

## Project Overview
This project demonstrates GPU-accelerated image processing by applying Sobel edge detection to batches of grayscale images using NVIDIA CUDA. Image processing is an ideal use case for GPUs because each pixel can be processed independently and in parallel.

## Motivation
CPU-based image processing becomes slow when working with large batches of images. GPUs provide massive parallelism, allowing thousands of pixels to be processed simultaneously.

## Technologies Used
- NVIDIA CUDA
- CUDA Runtime API
- Custom CUDA kernels
- C++

## How GPU Acceleration Is Used
Each image pixel is mapped to a CUDA thread. The Sobel operator computes gradients using neighboring pixels, which is well-suited for GPU parallel execution.

## Build Instructions
Requirements:
- NVIDIA GPU
- CUDA Toolkit installed

Build:
