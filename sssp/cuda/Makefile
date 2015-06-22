.phony: clean edge_list csr

edge_list: cuda-SSSP_EdgeList.out
	

cuda-SSSP_EdgeList.out: main.o CudaParallelSSSP.o
	nvcc -DCUDA_EDGE_LIST -g main.o CudaParallelSSSP.o -o cuda-SSSP.out
main.o: main.cpp CudaGraph.h
	nvcc -g main.cpp -c -o main.o
CudaParallelSSSP.o: CudaParallelSSSP.cu CudaGraph.h
	nvcc -DCUDA_EDGE_LIST -g CudaParallelSSSP.cu -c -o CudaParallelSSSP.o

csr: cuda-SSSP_CSR.out
	

cuda-SSSP_CSR.out: main-CSR.o CudaParallelSSSP-CSR.o
	nvcc -DCUDA_CSR -g main-CSR.o CudaParallelSSSP-CSR.o -o cuda-SSSP_CSR.out
main-CSR.o: main.cpp CudaGraph.h
	nvcc -DCUDA_CSR -g main.cpp -c -o main-CSR.o
CudaParallelSSSP-CSR.o: CudaParallelSSSP-CSR.cu CudaGraph.h
	nvcc -DCUDA_CSR -g CudaParallelSSSP-CSR.cu -c -o CudaParallelSSSP-CSR.o

clean:
	rm *.o *.out
