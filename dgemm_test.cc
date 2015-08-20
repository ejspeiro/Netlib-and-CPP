#include <iostream>
#include <iomanip>

#include <vector>

int main () {

  std::cout << "Testing BLAS' dgemm_ in C++11." << std::endl;
  std::cout << std::endl;

  // Explore the impact of row-wise ordering and the usage of the C++11 STL,
  // when using BLAS's dgemm_ routine.

  // Dimensions of the matrices used in every test case:

  int mm{9};  // Rows of aa and rows of cc.
  int nn{2};  // Cols of aa and cols of cc.
  int kk{2};  // Cols of aa and rows of bb.

  // First test case: matrices with transposed data:

  std::vector<double> aat{
    -0.0263766,  0.150229, -0.313059,  0.200632, 0.281068, -0.650341, 0.537914,  -0.214473,   0.0344071,
     0.0233982, -0.156083,  0.437434, -0.657155, 0.549301, -0.221726, 0.00200615, 0.0305288, -0.00770386};

  std::vector<double> bbt{
    -12.8,    -57.1678,
    -50.724, -356.35};

  // These values correspond to the non-transposed versions of the matrices.
  char ta{'N'};
  char tb{'N'};
  int lda{std::max(1,mm)};
  int ldb{std::max(1,kk)};
  int ldc{std::max(1,mm)};

  std::cout << "aat =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    for (int jj = 0; jj < mm; ++jj) {
      std::cout << std::setw(12) << aat[ii*mm + jj] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "bbt =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    for (int jj = 0; jj < kk; ++jj) {
      std::cout << std::setw(12) << bbt[ii*kk + jj] << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
