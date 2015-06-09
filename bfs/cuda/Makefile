./cuda-BFS.out: main.o CudaParallelBFS.o
	nvcc -g main.o CudaParallelBFS.o -o cuda-BFS.out
main.o: main.cpp CudaGraph.h
	nvcc -g main.cpp -c -o main.o
CudaParallelBFS.o: CudaParallelBFS.cu CudaGraph.h
	nvcc -g CudaParallelBFS.cu -c -o CudaParallelBFS.o

clean:
	rm *.o *.out
