#gcc=/usr/local/Cellar/gcc/4.9.1/bin/g++-4.9
.phony: random clean
./a.out: main.cpp 
	g++ -DTEST -ggdb  main.cpp 

random: main.cpp
	g++ -DRANDOM -ggdb main.cpp

clean: 
	rm *.o *.out
