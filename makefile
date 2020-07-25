CC=nvcc

cuda : src/cuda.cu
	$(CC) -o prog src/cuda.cu

clean : 
	rm prog
