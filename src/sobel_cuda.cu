#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 512
#define HEIGHT 512

__global__ void sobelKernel(unsigned char* input, unsigned char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < WIDTH - 1 && y > 0 && y < HEIGHT - 1) {
        int gx =
            -input[(y-1)*WIDTH + (x-1)] - 2*input[y*WIDTH + (x-1)] - input[(y+1)*WIDTH + (x-1)] +
             input[(y-1)*WIDTH + (x+1)] + 2*input[y*WIDTH + (x+1)] + input[(y+1)*WIDTH + (x+1)];

        int gy =
            -input[(y-1)*WIDTH + (x-1)] - 2*input[(y-1)*WIDTH + x] - input[(y-1)*WIDTH + (x+1)] +
             input[(y+1)*WIDTH + (x-1)] + 2*input[(y+1)*WIDTH + x] + input[(y+1)*WIDTH + (x+1)];

        int magnitude = min(255, abs(gx) + abs(gy));
        output[y * WIDTH + x] = (unsigned char)magnitude;
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(unsigned char);

    unsigned char *h_input = (unsigned char*)malloc(size);
    unsigned char *h_output = (unsigned char*)malloc(size);

    for (int i = 0; i < WIDTH * HEIGHT; i++)
        h_input[i] = rand() % 256;

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(WIDTH / 16, HEIGHT / 16);

    sobelKernel<<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("GPU Sobel Edge Detection completed successfully.\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
