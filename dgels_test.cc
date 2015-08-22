#include <iostream>
#include <iomanip>

#include <vector>

#include <algorithm>

/*!
\brief Double-Precision General Matrix Overdetermined or Underdetermined Solver.

DGELS solves overdetermined or underdetermined real linear systems involving an
M-by-N matrix A, or its transpose, using a QR or LQ factorization of A.  It is
assumed that A has full rank.

The following options are provided:

1. If TRANS = 'N' and m >= n:  find the least squares solution of an
overdetermined system, i.e., solve the least squares problem

                minimize || B - A*X ||.

2. If TRANS = 'N' and m < n:  find the minimum norm solution of an
underdetermined system A * X = B.

3. If TRANS = 'T' and m >= n:  find the minimum norm solution of an
undetermined system A**T * X = B.

4. If TRANS = 'T' and m < n:  find the least squares solution of an
overdetermined system, i.e., solve the least squares problem

                minimize || B - A**T * X ||.

Several right hand side vectors b and solution vectors x can be handled in a
single call; they are stored as the columns of the M-by-NRHS right hand side
matrix B and the N-by-NRHS solution matrix X.

\sa http://www.math.utah.edu/software/lapack/lapack-d/dgels.html

\param[in]      trans Am I giving the transpose of the matrix?
\param[in]      m     The number of rows of the matrix a.  m >= 0.
\param[in]      n     The number of columns of the matrix a.  n >= 0.
\param[in]      nrhs  The number of right-hand sides.
\param[in,out]  a     On entry, the m-by-n matrix a.
\param[in]      lda   The leading dimension of a. lda >= max(1,m).
\param[in,out]  b     On entry, matrix b of right-hand side vectors.
\param[in]      ldb   The leading dimension of b. ldb >= max(1,m,n).
\param[in,out]  work  On exit, if info = 0, work(1) is optimal lwork.
\param[in,out]  lwork The dimension of the array work.
\param[in,out]  info  If info = 0, then successful exit.
*/
extern "C" void dgels_(char* trans,
                       int* m,
                       int* n,
                       int* nrhs,
                       double* a,
                       int* lda,
                       double* b,
                       int* ldb,
                       double* work,
                       int* lwork,
                       int* info);

int main () {

  std::cout << "Testing LAPACK's dgels_ in C++11." << std::endl;
  std::cout << std::endl;

  // Explore the impact of row-wise ordering and the usage of the C++11 STL,
  // when using LAPACK's dgels_ routine.

  // Dimensions of the matrices used in every test case:

  int mm{6};  // Rows of aa.
  int nn{5};  // Cols of aa.

  // First test case: matrices with transposed data.

  int ldx_options[] = {1, mm, nn};
  int ldx{*std::max_element(ldx_options, ldx_options + 3)}; // LD of xx vector.

  std::vector<double> xx{0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

  char ta{'N'}; // Is aa's data transposed, i.e. is it in row-major ordering?

  int lda{std::max(1,mm)};  // Leading dimension of the aa matrix.

  std::vector<double> aat{
    1.0,     1.0,    1.0,     1.0,      1.0,     1.0,
   -0.5,     0.5,    1.5,     2.5,      3.5,     4.5,
    0.25,    0.25,   2.25,    6.25,    12.25,   20.25,
   -0.125,   0.125,  3.375,  15.625,   42.875,  91.125,
    0.0625,  0.0625, 5.0625, 39.0625, 150.062, 410.062};

  std::cout << "aat =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    for (int jj = 0; jj < mm; ++jj) {
      std::cout << std::setw(12) << aat[ii*mm + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  int nrhs{1};  // Number of arrays on the right-hand side matrix.

  // We are ready to solve the system. However, we must allocate the required
  // working memory. We can use the dgels_ routine to inquire regarding how much
  // working memory we need. We will do that first.

  std::vector<double> ww(1);  // Working memory for the dgels_ solver.

  int info{};
  int lwork{-1};

  dgels_(&ta, &mm, &nn, &nrhs, aat.data(), &lda, xx.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  lwork = (int) ww[0];

  std::cout << "Computed lwork: " << lwork << std::endl;
  std::cout << std::endl;

  ww.resize(lwork);

  dgels_(&ta, &mm, &nn, &nrhs, aat.data(), &lda, xx.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  std::cout << "xx =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    std::cout << std::setw(12) << xx[ii] << std::endl;
  }
  std::cout << std::endl;

  // Second test case: What if we DO NOT want to transpose the input matrices?

  std::vector<double> xx2{0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

  std::vector<double> aa{
    1.0, -0.5, 0.25,  -0.125,   0.0625,
    1.0,  0.5, 0.25,   0.125,   0.0625,
    1.0,  1.5, 2.25,   3.375,   5.0625,
    1.0,  2.5, 6.25,  15.625,  39.0625,
    1.0,  3.5, 12.25, 42.875, 150.062,
    1.0,  4.5, 20.25, 91.125, 410.062};

  std::cout << "aa =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    for (int jj = 0; jj < nn; ++jj) {
      std::cout << std::setw(12) << aa[ii*nn + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  ta = 'T'; // Is aa's data transposed, i.e. is it in row-major ordering?

  lda = std::max(1,mm); // Leading dimension of the aa matrix.

  lwork = -1;

  dgels_(&ta, &mm, &nn, &nrhs, aa.data(), &lda, xx2.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  lwork = (int) ww[0];

  std::cout << "Computed lwork: " << lwork << std::endl;
  std::cout << std::endl;

  ww.resize(lwork);

  dgels_(&ta, &mm, &nn, &nrhs, aa.data(), &lda, xx.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  std::cout << "xx =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    std::cout << std::setw(12) << xx[ii] << std::endl;
  }
  std::cout << std::endl;

  // When the matrix is transposed, changing the ta variable does not help. We
  // get a different solution than in the first case.

  // What if we are interested in solving the system for its transpose? That is,
  // if we assume aa := aat?

  // Third case: column-wise ordering of the transpose.

  std::vector<double> AAT{
    1.0, -0.5, 0.25,  -0.125,   0.0625,
    1.0,  0.5, 0.25,   0.125,   0.0625,
    1.0,  1.5, 2.25,   3.375,   5.0625,
    1.0,  2.5, 6.25,  15.625,  39.0625,
    1.0,  3.5, 12.25, 42.875, 150.062,
    1.0,  4.5, 20.25, 91.125, 410.062};

  std::swap(mm,nn);

  std::cout << "AAT =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    for (int jj = 0; jj < mm; ++jj) {
      std::cout << std::setw(12) << AAT[ii*mm + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
