#ifndef XCORE_LINALG_HPP
#define XCORE_LINALG_HPP

#include "opencv2/core.hpp"

namespace cv
{

namespace linalg
{

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
/// \return the rank of the matrix _a.
///
CV_EXPORTS_W int lstsq(InputArray _a, InputArray _b, OutputArray _x, OutputArray _residues, OutputArray _s);

} // linalg

} // cv

#endif // XCORE_LINALG_HPP
