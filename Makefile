all: dgemv_test dgemm_test

dgemv_test: dgemv_test.o
	gfortran dgemv_test.o $(HOME)/Libraries/BLAS/libblas.a -o dgemv_test.exe -lstdc++

dgemv_test.o: dgemv_test.cc
	g++ -std=c++11 -g -c dgemv_test.cc

dgemm_test: dgemm_test.o
	gfortran dgemm_test.o $(HOME)/Libraries/BLAS/libblas.a -o dgemm_test.exe -lstdc++

dgemm_test.o: dgemm_test.cc
	g++ -std=c++11 -g -c dgemm_test.cc

clean:
	rm -f *.o *exe
