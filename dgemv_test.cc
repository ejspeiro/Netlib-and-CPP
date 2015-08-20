#include <iostream>
#include <iomanip>

#include <vector>

/*!
\fn dgemv_

\brief Double precision General Matrix-Vector Multiplier:

Performs one of the following matrix-vector operations:

y := alpha*A*x + beta*y, or
y := alpha*A'*x + beta*y.

\sa http://www.math.utah.edu/software/lapack/lapack-blas/dgemv.html

\param[in]      ta      Is this the transpose of the matrix?
\param[in]      m       The number of rows of the matrix a.  m >= 0.
\param[in]      n       The number of columns of the matrix a. n >= 0.
\param[in]      alpha   The scalar alpha.
\param[in,out]  a       The leading m by n part of the array.
\param[in]      lda     The leading dimension of a. lda >= max(1,m).
\param[in,out]  x       Array of DIMENSION at least:
                        (1 + (n - 1)*abs(incx)), and (1 + (m - 1)*abs(incx)), if
                        ta = 'N' or 'n'.
\param[in]      incx    The increment for the elements of x. incx > 0.
\param[in]      beta    The scalar beta.
\param[in,out]  y       Array of DIMENSION at least:
                        (1 + (m - 1)*abs(incy)), and (1 + (n - 1)*abs(incy)), if
                        ta = 'N' or 'n'.
\param[in]      incy    The increment for the elements of y. incy > 0.
*/
extern "C" void dgemv_(char *ta,
                       int *m,
                       int *n,
                       double *alpha,
                       double *a,
                       int *lda,
                       double *x,
                       int *incx,
                       double *beta,
                       double *y,
                       int *incy);

int main () {

  std::cout << "Testing BLAS' dgemv_ in C++11." << std::endl;
  std::cout << std::endl;

  // Explore the impact of row-wise ordering and the usage of the C++11 STL,
  // when using BLAS's dgemv_ routine.

  // Dimensions of the matrices used in every test case:

  int mm{9};  // Rows of aa.
  int nn{2};  // Cols of aa.

  // First test case: matrices with transposed data.

  std::vector<double> aat{
    -1.0, 6.99999, -21.0, 35.0, -35.0, 21.0, -6.99999, 0.99999, 1.84771e-06,
    -7.00002, 48.0, -140.0, 224.0, -210.0, 112.0, -28.0, -9.428e-06, 1.0};

  std::vector<double> xx{
    -0.0932962,
    0.0197727};

  std::vector<double> yy{
    -0.892509,
    0.625705,
    0.420205,
    -0.0602338,
    -0.175969,
    0.0245964,
    0.131729,
    -0.0932962,
    0.0197727};

  // These values correspond to the non-transposed versions of the matrices.

  char ta{'N'}; // Is aa's data transposed, i.e. is it in row-major ordering?

  int lda{std::max(1,mm)};  // Leading dimension of the aa matrix.
  int incx{1};              // Increment for the elements of xx. incx >= 0.
  int incy{1};              // Increment for the elements of yy. incy >= 0.

  std::cout << "aat =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    for (int jj = 0; jj < mm; ++jj) {
      std::cout << std::setw(12) << aat[ii*mm + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "xx =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    std::cout << std::setw(12) << xx[ii] << std::endl;
  }
  std::cout << std::endl;

  std::cout << "yy =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    std::cout << std::setw(12) << yy[ii] << std::endl;
  }
  std::cout << std::endl;

  // Execute matrix-vector multiplication.

  double alpha{-1.0}; // Scalar for the matrix.
  double beta{1.0};   // Scalar for the first vector.

  dgemv_(&ta, &mm, &nn, &alpha, aat.data(), &lda, xx.data(), &incx, &beta, yy.data(), &incy);

  std::cout << "yy =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    std::cout << std::setw(12) << yy[ii] << std::endl;
  }
  std::cout << std::endl;

  // Second test case: What if we DO NOT want to transpose the input matrices?

  std::vector<double> aa{
    -1.0, -7.00002,
    6.99999, 48.0,
    -21.0, -140.0,
    35.0, 224.0,
    -35.0, -210.0,
    21.0, 112.0,
    -6.99999, -28.0,
    0.99999, -9.428e-06,
    1.84771e-06, 1.0};

  // Redeclare yy for the second test (it got rewritten).

  std::vector<double> yy2{
    -0.892509,
    0.625705,
    0.420205,
    -0.0602338,
    -0.175969,
    0.0245964,
    0.131729,
    -0.0932962,
    0.0197727};

  std::cout << "aa =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    for (int jj = 0; jj < nn; ++jj) {
      std::cout << std::setw(12) << aa[ii*nn + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // These values correspond to the TRANSPOSED versions of the matrices.

  ta = 'T'; // State that now, the input WILL be in row-wise ordering.

  // Intuition would say (as in the case for dgemm_) that we should use
  // max(1,nn). BUT this causes BLAS to issue an 'illegal value' issue. Ergo,
  // we do max(1,mm), but this yields an incorrect answer :(

  lda = std::max(1,mm); // Leading dimension of the aa matrix.

  dgemv_(&ta, &mm, &nn, &alpha, aat.data(), &lda, xx.data(), &incx, &beta, yy2.data(), &incy);

  std::cout << "yy =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    std::cout << std::setw(12) << yy2[ii] << std::endl;
  }
  std::cout << std::endl;

  // As of 2015-08-20, the matrix has to be transposed to be used.

  return EXIT_SUCCESS;
}
