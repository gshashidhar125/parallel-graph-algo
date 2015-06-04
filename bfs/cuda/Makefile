./cuda-BFS.out: main.o CudaParallelBFS.o
	nvcc  main.o CudaParallelBFS.o -o cuda-BFS.out
main.o: main.cpp
	nvcc main.cpp -c -o main.o
CudaParallelBFS.o: CudaParallelBFS.cu
	nvcc CudaParallelBFS.cu -c -o CudaParallelBFS.o
