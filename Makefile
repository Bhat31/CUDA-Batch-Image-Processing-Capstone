all:
	nvcc src/sobel_cuda.cu -o sobel

clean:
	rm -f sobel
