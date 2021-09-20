#ifndef CUDAXCORE_LINALG_HPP
#define CUDAXCORE_LINALG_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace cv
{

namespace cuda
{
///
/// \brief SVDecomp : decomposes matrix and stores the results to user-provided matrices
/// \param _X : decomposed matrix. The depth has to be CV_32F or CV_64F.
/// \param _u : calculated left singular vectors.
/// \param _s : calculated singular values.
/// \param _vt : transposed matrix of right singular vectors.
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void SVDecomp(InputArray _X, OutputArray _u, OutputArray _s, OutputArray _vt, Stream& stream = Stream::Null());

///
/// \brief QR : decomposes matrix and stores the results to user-provided matrices
/// \param _X : decomposed matrix. The depth has to be CV_32F or CV_64F.
/// \param _Q : is the orthogonal matrix from QR decomposition.
/// \param _R : is the upper triangular matrix from QR decomposition.
/// \param _tau : contains additional informations needed for solving a least squares problem using q and r.
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void QR(InputArray _X, OutputArray _Q, OutputArray _R, OutputArray _tau = noArray(), Stream& stream = Stream::Null());

///
/// \brief LU : decomposes matrix and stores the results to user-provided matrices
/// \param _X : decomposed matrix. The depth has to be CV_32F or CV_64F.
/// \param _L : contains the lower triangular matrix of the LU decomposition
/// \param _U : contains the upper triangular matrix of the LU decomposition
/// \param _pivot : contains the permutation indices to map the input to the decomposition
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void LU(InputArray _X, OutputArray _L, OutputArray _U, OutputArray _pivot, Stream& stream = Stream::Null());

///
/// \brief lstsq : compute the solution vector, a, of the equation a * x = y, by the least square method.
/// \param _a : M x N coefficient matrix.
///             If its depth correspond to an integer type it will converted single precision floating point.
///             It must have a single channel.
/// \param _b : M x K of M x 1 matrix. If _b is not a vector the least square solution is computed for each column.
///             If its depth correspond to an integer type it will converted single precision floating point.
///             It must have a single channel.
/// \param _x : Least-squares solution.
/// \param _residues : Sum of the residuals.
/// \param _s : singular values of _a.
/// \param stream : Stream of the asynchronous version.
/// \return the rank of the matrix _a.
///
CV_EXPORTS_W int lstsq(InputArray _a, InputArray _b, OutputArray _x, OutputArray _residues, OutputArray _s, Stream& stream = Stream::Null());



} // cuda

} // cv

#endif // CUDAXCORE_LINALG_HPP
