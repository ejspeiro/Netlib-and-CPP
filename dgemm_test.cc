#include <iostream>
#include <iomanip>

#include <vector>

/*!
\fn dgemv_

\brief Double-Precision General-Format Matrix-Matrix multiplier.

Performs:

C := alpha*op( A )*op( B ) + beta*C

\sa http://www.math.utah.edu/software/lapack/lapack-blas/dgemm.html

\param[in]      transa  Is this the transpose of the matrix a?
\param[in]      transb  Is this the transpose of the matrix b?
\param[in]      m       The number of rows of the matrices a and c.  m >= 0.
\param[in]      n       The number of cols of the matrices b and c.  n >= 0.
\param[in]      k       The number of cols of a and rows of c.  k >= 0.
\param[in]      alpha   The scalar alpha.
\param[in,out]  a       Matrix a.
\param[in]      lda     The leading dimension of a.
\param[in,out]  b       Matrix b.
\param[in]      ldb     The leading dimension of b.
\param[in]      beta    The scalar beta.
\param[in,out]  c       Matrix c.
\param[in]      ldc     The leading dimension of c.
*/
extern "C" void dgemm_(char *ta,
                       char* tb,
                       int *m,
                       int *n,
                       int *k,
                       double *alpha,
                       double *a,
                       int *lda,
                       double *b,
                       int *ldb,
                       double *beta,
                       double *c,
                       int *ldc);

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

  char ta{'N'}; // Is aa's data transposed, i.e. is it in row-major ordering?
  char tb{'N'}; // Is bb's data transposed, i.e. is it in row-major ordering?

  int lda{std::max(1,mm)};  // Leading dimension of the aa matrix.
  int ldb{std::max(1,kk)};  // Leading dimension of the bb matrix.
  int ldc{std::max(1,mm)};  // Leading dimension of the cc matrix.

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

  // Execute matrix-matrix multiplication.

  double alpha{1.0};  // First scalar coefficient.
  double beta{0.0};   // Second scalar coefficient.

  std::vector<double> cct(mm*kk); // Output matrix.

  dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha, aat.data(), &lda, bbt.data(), &ldb, &beta, cct.data(), &ldc);

  std::cout << "cct =" <<std::endl;
  for (int ii = 0; ii < kk; ++ii) {
    for (int jj = 0; jj < mm; ++jj) {
      std::cout << std::setw(12) << cct[ii*mm + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
