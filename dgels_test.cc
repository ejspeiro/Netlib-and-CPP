#include <iostream>
#include <iomanip>

#include <vector>

#include <algorithm>

/*!
\brief Double-precision GEneral matrix Least Squares solver.

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

  int mm{6};  // Rows of aa.
  int nn{5};  // Cols of aa.

  // First test case: matrices with column-major ordering data.

  char ta{'N'}; // Is aa's data transposed, i.e. is it in row-major ordering?

  std::vector<double> xx{0.0, 1.0, 0.0, 0.0, 0.0, 0.0}; // xx vector.

  // These values correspond to the column-major versions of the matrices.

  int lda{std::max(1,mm)};                                  // LD of aa matrix.
  int ldx_options[] = {1, mm, nn};                          // Used to get LD.
  int ldx{*std::max_element(ldx_options, ldx_options + 3)}; // LD of xx vector.

  // Data array of the aa matrix, in column-major ordering.
  std::vector<double> aa_col_maj_ord{
    1.0,     1.0,    1.0,     1.0,      1.0,     1.0,
   -0.5,     0.5,    1.5,     2.5,      3.5,     4.5,
    0.25,    0.25,   2.25,    6.25,    12.25,   20.25,
   -0.125,   0.125,  3.375,  15.625,   42.875,  91.125,
    0.0625,  0.0625, 5.0625, 39.0625, 150.062, 410.062};

  std::cout << "aa_col_maj_ord =" << std::endl;
  if (ta == 'N') {
    std::swap(mm,nn);
  }
  for (int ii = 0; ii < mm; ++ii) {
    for (int jj = 0; jj < nn; ++jj) {
      std::cout << std::setw(12) << aa_col_maj_ord[ii*nn + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  if (ta == 'N') {
    std::swap(mm,nn);
  }

  int nrhs{1};  // Number of arrays on the right-hand side matrix.

  // We are ready to solve the system. However, we must allocate the required
  // working memory. We can use the dgels_ routine to inquire regarding how much
  // working memory we need. We will do that first.

  std::vector<double> ww(1);  // Working memory for the dgels_ solver.

  int info{};     // Outcome of the querying process.
  int lwork{-1};  // Length of the work array. If -1, the go into inquire mode.

  dgels_(&ta, &mm, &nn, &nrhs, aa_col_maj_ord.data(), &lda,
         xx.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  lwork = (int) ww[0];

  std::cout << "Computed lwork: " << lwork << std::endl;
  std::cout << std::endl;

  ww.resize(lwork);

  dgels_(&ta, &mm, &nn, &nrhs, aa_col_maj_ord.data(), &lda,
         xx.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  std::cout << "xx =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    std::cout << std::setw(12) << xx[ii] << std::endl;
  }
  std::cout << std::endl;

  // Second test case: What if we have a row-major ordered input matrix?

  ta = 'T'; // Is aa's data transposed, i.e. is it in row-major ordering?

  // Data array of aa matrix, in row-major ordering.
  std::vector<double> aa_row_maj_ord{
    1.0, -0.5, 0.25,  -0.125,   0.0625,
    1.0,  0.5, 0.25,   0.125,   0.0625,
    1.0,  1.5, 2.25,   3.375,   5.0625,
    1.0,  2.5, 6.25,  15.625,  39.0625,
    1.0,  3.5, 12.25, 42.875, 150.062,
    1.0,  4.5, 20.25, 91.125, 410.062};

  std::cout << "aa_row_maj_ord =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    for (int jj = 0; jj < nn; ++jj) {
      std::cout << std::setw(12) << aa_row_maj_ord[ii*nn + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::vector<double> xx2{0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

  lda = std::max(1,nn); // Leading dimension of the aa_row_maj_ord matrix.

  lwork = -1;

  std::swap(mm,nn);
  dgels_(&ta, &mm, &nn, &nrhs, aa_row_maj_ord.data(), &lda,
         xx2.data(), &ldx, ww.data(), &lwork, &info);
  std::swap(mm,nn);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  lwork = (int) ww[0];

  std::cout << "Computed lwork: " << lwork << std::endl;
  std::cout << std::endl;

  ww.resize(lwork);

  std::swap(mm,nn);
  dgels_(&ta, &mm, &nn, &nrhs, aa_row_maj_ord.data(), &lda,
         xx2.data(), &ldx, ww.data(), &lwork, &info);
  std::swap(mm,nn);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  std::cout << "xx2 =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    std::cout << std::setw(12) << xx2[ii] << std::endl;
  }
  std::cout << std::endl;

  // Third and fourth test cases.

  // What if we want to solve the system for the transpose of the original
  // matrix, given to us in an row-major ordered fashion?

  ta = 'T'; // Is aat's data transposed, i.e. is it in row-major ordering?

  // That would mean, swapping mm and nn!

  mm = 5;
  nn = 6;

  // Data array of the aaT matrix, in row-major ordering.
  std::vector<double> aat_row_maj_ord{
    1.0,     1.0,    1.0,     1.0,      1.0,     1.0,
   -0.5,     0.5,    1.5,     2.5,      3.5,     4.5,
    0.25,    0.25,   2.25,    6.25,    12.25,   20.25,
   -0.125,   0.125,  3.375,  15.625,   42.875,  91.125,
    0.0625,  0.0625, 5.0625, 39.0625, 150.062, 410.062};

  std::cout << "aat_row_maj_ord =" << std::endl;
  for (int ii = 0; ii < mm; ++ii) {
    for (int jj = 0; jj < nn; ++jj) {
      std::cout << std::setw(12) << aat_row_maj_ord[ii*nn + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::vector<double> xx3{0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

  lda = std::max(1,nn); // Leading dimension of the aat_row_maj_ord matrix.

  lwork = -1;

  std::swap(mm,nn);
  dgels_(&ta, &mm, &nn, &nrhs, aat_row_maj_ord.data(), &lda,
         xx3.data(), &ldx, ww.data(), &lwork, &info);
  std::swap(mm,nn);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  lwork = (int) ww[0];

  std::cout << "Computed lwork: " << lwork << std::endl;
  std::cout << std::endl;

  ww.resize(lwork);

  std::swap(mm,nn);
  dgels_(&ta, &mm, &nn, &nrhs, aat_row_maj_ord.data(), &lda,
         xx3.data(), &ldx, ww.data(), &lwork, &info);
  std::swap(mm,nn);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  std::cout << "xx3 =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    std::cout << std::setw(12) << xx3[ii] << std::endl;
  }
  std::cout << std::endl;

  // Final case: What if we want to solve the system for the transpose of the
  // original matrix, given to us in an column-major ordered fashion?

  ta = 'N'; // Is aat's data in row-major ordering?

  // Data array of aat matrix, in column-major ordering.
  std::vector<double> aat_col_maj_ord{
    1.0, -0.5, 0.25,  -0.125,   0.0625,
    1.0,  0.5, 0.25,   0.125,   0.0625,
    1.0,  1.5, 2.25,   3.375,   5.0625,
    1.0,  2.5, 6.25,  15.625,  39.0625,
    1.0,  3.5, 12.25, 42.875, 150.062,
    1.0,  4.5, 20.25, 91.125, 410.062};

  std::vector<double> xx4{0.0, 1.0, 0.0, 0.0, 0.0, 0.0};

  lda = std::max(1,mm); // Leading dimension of the aat_row_maj_ord matrix.

  lwork = -1;

  dgels_(&ta, &mm, &nn, &nrhs, aat_col_maj_ord.data(), &lda,
         xx4.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  lwork = (int) ww[0];

  std::cout << "Computed lwork: " << lwork << std::endl;
  std::cout << std::endl;

  ww.resize(lwork);

  dgels_(&ta, &mm, &nn, &nrhs, aat_col_maj_ord.data(), &lda,
         xx4.data(), &ldx, ww.data(), &lwork, &info);

  if (info != 0) {
    return EXIT_FAILURE;
  }

  std::cout << "xx4 =" << std::endl;
  for (int ii = 0; ii < nn; ++ii) {
    std::cout << std::setw(12) << xx4[ii] << std::endl;
  }
  std::cout << std::endl;

  return EXIT_SUCCESS;
}
