#include <iostream>
#include <iomanip>

#include <vector>

/*!
\fn dgemv_

\brief Double-Precision General Format Matrix-Vector Multiplier.

Performs one of the following matrix-vector operations:

y := alpha*A*x + beta*y, or
y := alpha*A'*x + beta*y.

\sa http://www.math.utransah.edu/software/lapack/lapack-blas/dgemv.html

\param[in]      transa  Is this matrix in row-major order?
\param[in]      m       The number of rows of the matrix a.  m >= 0.
\param[in]      n       The number of columns of the matrix a. n >= 0.
\param[in]      alpha   The scalar alpha.
\param[in,out]  a       The leading m by n part of the array.
\param[in]      lda     The leading dimension of a. lda >= max(1,m).
\param[in,out]  x       Array of DIMENSION at least:
                        (1 + (n - 1)*abs(ix)), and (1 + (m - 1)*abs(ix)), if
                        transa = 'N' or 'n'.
\param[in]      incx    The increment for the elements of x. ix > 0.
\param[in]      beta    The scalar beta.
\param[in,out]  y       Array of DIMENSION at least:
                        (1 + (m - 1)*abs(incy)), and (1 + (n - 1)*abs(incy)), if
                        transa = 'N' or 'n'.
\param[in]      incy    The increment for the elements of y. incy > 0.
*/
extern "C" void dgemv_(char *transa,
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

  int mm{9};  // Rows of aa.
  int nn{2};  // Cols of aa.

  // First test case: column-major order of the matrix, i.e. transa = 'N'.

  char transa{'N'}; // Is aa's data in row-major ordering?

  std::vector<double> aa_col_maj_ord{
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

  // These values correspond to the column-major versions of the matrices.

  int lda{std::max(1,mm)};  // Leading dimension of the aa matrix.
  int incx{1};              // Increment for the elements of xx. ix >= 0.
  int incy{1};              // Increment for the elements of yy. incy >= 0.

  std::cout << "aa_col_maj_ord =" << std::endl;
  if (transa == 'N') {
    std::swap(mm,nn);
  }
  for (int ii = 0; ii < mm; ++ii) {
    for (int jj = 0; jj < nn; ++jj) {
      std::cout << std::setw(12) << aa_col_maj_ord[ii*nn + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  if (transa == 'N') {
    std::swap(mm,nn);
  }

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

  dgemv_(&transa, &mm, &nn, &alpha, aa_col_maj_ord.data(), &lda,
         xx.data(), &incx, &beta, yy.data(), &incy);

  std::cout << "yy =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    std::cout << std::setw(12) << yy[ii] << std::endl;
  }
  std::cout << std::endl;

  // Second test case: What if we need the ordering to be row-major?

  transa = 'T'; // State that now, the input WILL be in row-wise ordering.

  std::vector<double> aa_row_maj_ord{
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

  if (transa == 'N') {
    std::swap(mm,nn);
  }
  std::cout << "aa_row_maj_ord =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    for (int jj = 0; jj < nn; ++jj) {
      std::cout << std::setw(12) << aa_row_maj_ord[ii*nn + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  if (transa == 'N') {
    std::swap(mm,nn);
  }

  // Intuition would say (as in the case for dgemm_) that we should use
  // max(1,nn). BUT this causes BLAS to issue an 'illegal value' issue. Ergo,
  // we do max(1,mm), but this yields an incorrect answer :(

  lda = std::max(1,nn); // Leading dimension of the aa matrix.
  std::swap(mm,nn);
  dgemv_(&transa, &mm, &nn, &alpha, aa_row_maj_ord.data(), &lda,
         xx.data(), &incx, &beta, yy2.data(), &incy);
  std::swap(mm,nn);

  std::cout << "yy =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    std::cout << std::setw(12) << yy2[ii] << std::endl;
  }
  std::cout << std::endl;

  // In the row-major case, we must swap the number of rows and cols to avoid
  // converting the ordering of the data.

  return EXIT_SUCCESS;
}
