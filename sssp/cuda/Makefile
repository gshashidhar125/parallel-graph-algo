.phony: clean
./cuda-SSSP.out: main.o CudaParallelSSSP.o
	nvcc -g main.o CudaParallelSSSP.o -o cuda-SSSP.out
main.o: main.cpp CudaGraph.h
	nvcc -g main.cpp -c -o main.o
CudaParallelSSSP.o: CudaParallelSSSP.cu CudaGraph.h
	nvcc -g CudaParallelSSSP.cu -c -o CudaParallelSSSP.o

clean:
	rm *.o *.out
