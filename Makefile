linux: dgemv_test dgemm_test dgels_test

dgemv_test: dgemv_test.o
	gfortran dgemv_test.o $(HOME)/Libraries/BLAS/libblas.a -o dgemv_test.exe -lstdc++

dgemv_test.o: dgemv_test.cc
	g++ -std=c++11 -g -c dgemv_test.cc

dgemm_test: dgemm_test.o
	gfortran dgemm_test.o $(HOME)/Libraries/BLAS/libblas.a -o dgemm_test.exe -lstdc++

dgemm_test.o: dgemm_test.cc
	g++ -std=c++11 -g -c dgemm_test.cc

dgels_test: dgels_test.o
	gfortran dgels_test.o $(HOME)/Libraries/lapack-3.4.1/liblapack.a $(HOME)/Libraries/BLAS/libblas.a -o dgels_test.exe -lstdc++

dgels_test.o: dgels_test.cc
	g++ -std=c++11 -g -c dgels_test.cc

macos: dgemv_test_mos

dgemv_test_mos: dgemv_test_mos.o
	g++ dgemv_test.o -framework Accelerate -o dgemv_test_mos.exe -lstdc++

dgemv_test_mos.o: dgemv_test.cc
	g++ -std=c++11 -g -c dgemv_test.cc

clean:
	rm -f *.o *exe
