#include <iostream>
#include <iomanip>

#include <vector>

/*!
\fn dgemv_

\brief Double-precision GEneral matrices Matrix-Matrix multiplier.

Performs:

C := alpha*op(a)*op(b) + beta*c

\sa http://www.math.utah.edu/software/lapack/lapack-blas/dgemm.html

\param[in]      transa  Is a in row-major storage?
\param[in]      transb  Is b in row-major storage?
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
extern "C" void dgemm_(char *ta,      // 1.
                       char *tb,
                       int *m,
                       int *n,
                       int *k,
                       double *alpha,
                       double *a,
                       int *lda,      // 8.
                       double *b,
                       int *ldb,
                       double *beta,
                       double *c,
                       int *ldc);     // 13.

int main () {

  std::cout << "Testing BLAS' dgemm_ in C++11." << std::endl;
  std::cout << std::endl;

  // Explore the impact of row-major storage and the usage of the C++11 STL,
  // when using BLAS's dgemm_ routine.

  int mm{9};  // Rows of aa and rows of cc.
  int nn{2};  // Cols of bb and cols of cc.
  int kk{2};  // Cols of aa and rows of bb.

  int aa_num_rows{mm};  // Rows of aa.
  int aa_num_cols{kk};  // Columns of aa.

  int bb_num_rows{kk};  // Rows of bb.
  int bb_num_cols{nn};  // Columns of bb.

  int cc_num_rows{mm};  // Rows of cc.
  int cc_num_cols{nn};  // Columns of cc.

  // First test case: matrices with column-major ordering.

  char ta{'N'}; // Is aa's data in row-major storage?
  char tb{'N'}; // Is bb's data in row-major storage?

  // Data array of the aa matrix, in column-major storage.
  std::vector<double> aa_col_maj_sto{-0.0263766, 0.150229, -0.313059, 0.200632, 0.281068, -0.650341, 0.537914, -0.214473, 0.0344071, 0.0233982, -0.156083, 0.437434, -0.657155, 0.549301, -0.221726, 0.00200615, 0.0305288, -0.00770386};

  // Data array of bb matrix, in column-major storage.
  std::vector<double> bb_col_maj_sto{-12.8, -57.1678, -50.724, -356.35};

  // These values correspond to the column-major versions of the matrices.

  int lda{std::max(1,mm)};  // Leading dimension of the aa matrix.
  int ldb{std::max(1,kk)};  // Leading dimension of the bb matrix.
  int ldc{std::max(1,mm)};  // Leading dimension of the cc matrix.

  std::cout << "aa_col_maj_sto =" << std::endl;
  if (ta == 'N') {
    std::swap(aa_num_rows,aa_num_cols);
  }
  for (int ii = 0; ii < aa_num_rows; ++ii) {
    for (int jj = 0; jj < aa_num_cols; ++jj) {
      std::cout << std::setw(12) << aa_col_maj_sto[ii*aa_num_cols + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  if (ta == 'N') {
    std::swap(aa_num_rows,aa_num_cols);
  }

  std::cout << "bb_col_maj_sto =" << std::endl;
  if (tb == 'N') {
    std::swap(bb_num_rows,bb_num_cols);
  }
  for (int ii = 0; ii < bb_num_rows; ++ii) {
    for (int jj = 0; jj < bb_num_cols; ++jj) {
      std::cout << std::setw(12) << bb_col_maj_sto[ii*bb_num_cols + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  if (tb == 'N') {
    std::swap(bb_num_rows,bb_num_cols);
  }

  // Execute matrix-matrix multiplication.

  double alpha{1.0};  // First scalar coefficient.
  double beta{0.0};   // Second scalar coefficient.

  std::vector<double> cc_col_maj_sto(cc_num_rows*cc_num_cols);  // Output matrix.

  dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha, aa_col_maj_sto.data(), &lda,
         bb_col_maj_sto.data(), &ldb, &beta, cc_col_maj_sto.data(), &ldc);

  std::cout << "Test 1:" << std::endl;
  std::cout << std::endl;

  std::cout << "cc_col_maj_sto =" << std::endl;
  std::swap(cc_num_rows,cc_num_cols);
  for (int ii = 0; ii < cc_num_rows; ++ii) {
    for (int jj = 0; jj < cc_num_cols; ++jj) {
      double vv = cc_col_maj_sto[ii*cc_num_cols + jj];
      std::cout << std::setw(12) << vv;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::swap(cc_num_rows,cc_num_cols);

  // Output matrix will always be given in column-major storage.

  // Second test case: What if we have a row-major storage input matrix?

  ta = 'T'; // State that now, the input will be in row-major storage.
  tb = 'T'; // State that now, the input will be in row-major storage.

  std::vector<double> aa_row_maj_sto = {
    -0.0263766, 0.0233982,
     0.150229, -0.156083,
    -0.313059,  0.437434,
     0.200632, -0.657155,
     0.281068,  0.549301,
    -0.650341, -0.221726,
     0.537914,  0.00200615,
    -0.214473,  0.0305288,
     0.0344071,-0.00770386};  // Data array of aa matrix, in row-major ordering.

  std::vector<double> bb_row_maj_sto = {
    -12.8,     -50.724,
    -57.1678, -356.35}; // Data array of bb matrix, in row-major ordering.

  // These values correspond to the row-major ordered versions of the matrices.

  lda = std::max(1,kk);
  ldb = std::max(1,nn);
  ldc = std::max(1,mm);


  dgemm_(&ta, &tb, &mm, &nn, &kk, &alpha, aa_row_maj_sto.data(), &lda,
         bb_row_maj_sto.data(), &ldb, &beta, cc_col_maj_sto.data(), &ldc);

  std::cout << "Test 2:" << std::endl;
  std::cout << std::endl;

  std::cout << "cc_col_maj_sto =" << std::endl;
  std::swap(cc_num_rows,cc_num_cols);
  for (int ii = 0; ii < cc_num_rows; ++ii) {
    for (int jj = 0; jj < cc_num_cols; ++jj) {
      std::cout << std::setw(12) << cc_col_maj_sto[ii*cc_num_cols + jj];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::swap(cc_num_rows,cc_num_cols);

  // The product matrix will always be col-major ordered! ;)

  return EXIT_SUCCESS;
}
