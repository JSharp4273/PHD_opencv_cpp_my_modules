#ifndef CUDAXCORE_ARRAYFIRE_HPP
#define CUDAXCORE_ARRAYFIRE_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace af
{

class array;

} // af

namespace cv
{

namespace cuda
{

namespace arrayfire
{

///
/// \brief GpuMat2Array : copy an OpenCV's GpuMat to an ArrayFire array
/// \param _src : data to copy
/// \param _dst : data copied
/// \param stream : Stream of the asynchronous version.
///
void GpuMat2Array(const GpuMat& _src, af::array& _dst, Stream& stream = Stream::Null());

///
/// \brief Array2GpuMat : copy an ArrayFire array to an OpenCV's GpuMat
/// \param _src : data to copy
/// \param _dst : data copied
/// \param stream : Stream of the asynchronous version.
///
void Array2GpuMat(const af::array& _src, GpuMat& _dst, Stream& stream = Stream::Null());

} // arrayfire

}// cuda

}// cv

#endif // CUDAXCORE_ARRAYFIRE_HPP
