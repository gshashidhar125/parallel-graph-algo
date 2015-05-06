./a.out: main.cpp parallelPrims.cpp Thread.h
	g++ -ggdb main.cpp parallelPrims.cpp -lpthread
