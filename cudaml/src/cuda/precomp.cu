#include "../precomp.hpp"

#include "opencv2/cudaarithm.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/cudev.hpp"


#include <curand.h>
#include <curand_kernel.h>

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

namespace cv
{

namespace cuda
{

namespace device
{

template<class SrcType, class DstType>
__global__ void centreToTheMeanAxis0Kernel(const PtrStep<SrcType> X, const PtrStep<SrcType> mu_X, PtrStepSz<DstType> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    dst(y,x) = device::saturate_cast<DstType>(X(y,x) - mu_X(0,x));
}

template<class SrcType, class DstType>
void centreToTheMeanAxis0Impl(const GpuMat& X, const GpuMat& mu_X, GpuMat& dst, Stream& stream)
{
    dim3 block(32,8);
    dim3 grid (divUp (X.cols, block.x), divUp (X.rows, block.y));

    cudaSafeCall( cudaFuncSetCacheConfig (centreToTheMeanAxis0Kernel<SrcType, DstType>, cudaFuncCachePreferL1) );
    centreToTheMeanAxis0Kernel<SrcType, DstType><<<grid, block, 0, StreamAccessor::getStream(stream)>>>(X, mu_X, dst);
    cudaSafeCall ( cudaGetLastError () );

    if (stream == 0)
         cudaSafeCall( cudaDeviceSynchronize() );
}

} // device

} // cuda

} // cv


#define SPEC_CTM(stype, dtype)\
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype, dtype>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype ## 2, dtype ## 2>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype ## 3, dtype ## 3>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype ## 4, dtype ## 4>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

SPEC_CTM(uchar, uchar)
SPEC_CTM(uchar, ushort)
SPEC_CTM(uchar, short)
SPEC_CTM(uchar, int)
SPEC_CTM(uchar, float)
SPEC_CTM(uchar, double)

SPEC_CTM(ushort, uchar)
SPEC_CTM(ushort, ushort)
SPEC_CTM(ushort, short)
SPEC_CTM(ushort, int)
SPEC_CTM(ushort, float)
SPEC_CTM(ushort, double)

SPEC_CTM(short, uchar)
SPEC_CTM(short, ushort)
SPEC_CTM(short, short)
SPEC_CTM(short, int)
SPEC_CTM(short, float)
SPEC_CTM(short, double)

SPEC_CTM(int, uchar)
SPEC_CTM(int, ushort)
SPEC_CTM(int, short)
SPEC_CTM(int, int)
SPEC_CTM(int, float)
SPEC_CTM(int, double)

SPEC_CTM(float, uchar)
SPEC_CTM(float, ushort)
SPEC_CTM(float, short)
SPEC_CTM(float, int)
SPEC_CTM(float, float)
SPEC_CTM(float, double)

SPEC_CTM(double, uchar)
SPEC_CTM(double, ushort)
SPEC_CTM(double, short)
SPEC_CTM(double, int)
SPEC_CTM(double, float)
SPEC_CTM(double, double)

#undef SPEC_CTM

#define SPEC_CTM_X_CHAR(stype)\
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype, schar>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype ## 2, char ## 2>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype ## 3, char ## 3>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<stype ## 4, char ## 4>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

SPEC_CTM_X_CHAR(uchar)
SPEC_CTM_X_CHAR(ushort)
SPEC_CTM_X_CHAR(short)
SPEC_CTM_X_CHAR(int)
SPEC_CTM_X_CHAR(float)
SPEC_CTM_X_CHAR(double)

#undef SPEC_CTM_X_CHAR

#define SPEC_CTM_CHAR_X(dtype)\
template void cv::cuda::device::centreToTheMeanAxis0Impl<schar, dtype>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<char ## 2, dtype ## 2>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<char ## 3, dtype ## 3>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::centreToTheMeanAxis0Impl<char ## 4, dtype ## 4>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

SPEC_CTM_CHAR_X(uchar)
SPEC_CTM_CHAR_X(ushort)
SPEC_CTM_CHAR_X(short)
SPEC_CTM_CHAR_X(int)
SPEC_CTM_CHAR_X(float)
SPEC_CTM_CHAR_X(double)

#undef SPEC_CTM_CHAR_X

#define SPEC_CTM_CHAR_CHAR\
    template void cv::cuda::device::centreToTheMeanAxis0Impl<schar, schar>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
    template void cv::cuda::device::centreToTheMeanAxis0Impl<char ## 2, char ## 2>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
    template void cv::cuda::device::centreToTheMeanAxis0Impl<char ## 3, char ## 3>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
    template void cv::cuda::device::centreToTheMeanAxis0Impl<char ## 4, char ## 4>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

SPEC_CTM_CHAR_CHAR

#undef SPEC_CTM_CHAR_CHAR

#endif
