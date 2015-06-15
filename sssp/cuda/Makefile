.phony: clean
CPPFLAGS = -DCUDA_EDGE_LIST -g
./cuda-SSSP.out: main.o CudaParallelSSSP.o
	nvcc $(CPPFLAGS) main.o CudaParallelSSSP.o -o cuda-SSSP.out
main.o: main.cpp CudaGraph.h
	nvcc $(CPPFLAGS) main.cpp -c -o main.o
CudaParallelSSSP.o: CudaParallelSSSP.cu CudaGraph.h
	nvcc $(CPPFLAGS) CudaParallelSSSP.cu -c -o CudaParallelSSSP.o

clean:
	rm *.o *.out
