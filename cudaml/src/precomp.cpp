#include "precomp.hpp"

#include "opencv2/core/cuda/common.hpp"


namespace cv
{

namespace cuda
{

namespace device
{

template<class SrcType, class DstType>
void centreToTheMeanAxis0Impl(const GpuMat& X, const GpuMat&, GpuMat&, Stream&);

} // device

void centreToTheMeanAxis0(const GpuMat& X, const GpuMat& mu_X, GpuMat& dst, const int& dtype, Stream& stream)
{

    CV_Assert(X.type() == mu_X.type() && X.channels() <=4);

    typedef void (*function_type)(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

    static const function_type funcs[7][7][4] = { {
                                                      {device::centreToTheMeanAxis0Impl<uchar, uchar >, device::centreToTheMeanAxis0Impl<uchar2, uchar2 >, device::centreToTheMeanAxis0Impl<uchar3, uchar3 >, device::centreToTheMeanAxis0Impl<uchar4, uchar4 > },
                                                      {device::centreToTheMeanAxis0Impl<uchar, schar >, device::centreToTheMeanAxis0Impl<uchar2, char2 >, device::centreToTheMeanAxis0Impl<uchar3, char3 >, device::centreToTheMeanAxis0Impl<uchar4, char4 > },
                                                      {device::centreToTheMeanAxis0Impl<uchar, ushort>, device::centreToTheMeanAxis0Impl<uchar2, ushort2>, device::centreToTheMeanAxis0Impl<uchar3, ushort3>, device::centreToTheMeanAxis0Impl<uchar4, ushort4> },
                                                      {device::centreToTheMeanAxis0Impl<uchar, short >, device::centreToTheMeanAxis0Impl<uchar2, short2 >, device::centreToTheMeanAxis0Impl<uchar3, short3 >, device::centreToTheMeanAxis0Impl<uchar4, short4 > },
                                                      {device::centreToTheMeanAxis0Impl<uchar, int   >, device::centreToTheMeanAxis0Impl<uchar2, int2   >, device::centreToTheMeanAxis0Impl<uchar3, int3   >, device::centreToTheMeanAxis0Impl<uchar4, int4   > },
                                                      {device::centreToTheMeanAxis0Impl<uchar, float >, device::centreToTheMeanAxis0Impl<uchar2, float2 >, device::centreToTheMeanAxis0Impl<uchar3, float3 >, device::centreToTheMeanAxis0Impl<uchar4, float4 >,},
                                                      {device::centreToTheMeanAxis0Impl<uchar, double>, device::centreToTheMeanAxis0Impl<uchar2, double2>, device::centreToTheMeanAxis0Impl<uchar3, double3>, device::centreToTheMeanAxis0Impl<uchar4, double4>,}
                                                  },
                                               {
                                                      {device::centreToTheMeanAxis0Impl<schar, uchar >, device::centreToTheMeanAxis0Impl<char2, uchar2 >, device::centreToTheMeanAxis0Impl<char3, uchar3 >, device::centreToTheMeanAxis0Impl<char4, uchar4 > },
                                                      {device::centreToTheMeanAxis0Impl<schar, schar >, device::centreToTheMeanAxis0Impl<char2, char2 >, device::centreToTheMeanAxis0Impl<char3, char3 >, device::centreToTheMeanAxis0Impl<char4, char4 > },
                                                      {device::centreToTheMeanAxis0Impl<schar, ushort>, device::centreToTheMeanAxis0Impl<char2, ushort2>, device::centreToTheMeanAxis0Impl<char3, ushort3>, device::centreToTheMeanAxis0Impl<char4, ushort4> },
                                                      {device::centreToTheMeanAxis0Impl<schar, short >, device::centreToTheMeanAxis0Impl<char2, short2 >, device::centreToTheMeanAxis0Impl<char3, short3 >, device::centreToTheMeanAxis0Impl<char4, short4 > },
                                                      {device::centreToTheMeanAxis0Impl<schar, int   >, device::centreToTheMeanAxis0Impl<char2, int2   >, device::centreToTheMeanAxis0Impl<char3, int3   >, device::centreToTheMeanAxis0Impl<char4, int4   > },
                                                      {device::centreToTheMeanAxis0Impl<schar, float >, device::centreToTheMeanAxis0Impl<char2, float2 >, device::centreToTheMeanAxis0Impl<char3, float3 >, device::centreToTheMeanAxis0Impl<char4, float4 >,},
                                                      {device::centreToTheMeanAxis0Impl<schar, double>, device::centreToTheMeanAxis0Impl<char2, double2>, device::centreToTheMeanAxis0Impl<char3, double3>, device::centreToTheMeanAxis0Impl<char4, double4>,}
                                                  },
                                               {
                                                      {device::centreToTheMeanAxis0Impl<ushort, uchar >, device::centreToTheMeanAxis0Impl<ushort2, uchar2 >, device::centreToTheMeanAxis0Impl<ushort3, uchar3 >, device::centreToTheMeanAxis0Impl<ushort4, uchar4 > },
                                                      {device::centreToTheMeanAxis0Impl<ushort, schar >, device::centreToTheMeanAxis0Impl<ushort2, char2 >, device::centreToTheMeanAxis0Impl<ushort3, char3 >, device::centreToTheMeanAxis0Impl<ushort4, char4 > },
                                                      {device::centreToTheMeanAxis0Impl<ushort, ushort>, device::centreToTheMeanAxis0Impl<ushort2, ushort2>, device::centreToTheMeanAxis0Impl<ushort3, ushort3>, device::centreToTheMeanAxis0Impl<ushort4, ushort4> },
                                                      {device::centreToTheMeanAxis0Impl<ushort, short >, device::centreToTheMeanAxis0Impl<ushort2, short2 >, device::centreToTheMeanAxis0Impl<ushort3, short3 >, device::centreToTheMeanAxis0Impl<ushort4, short4 > },
                                                      {device::centreToTheMeanAxis0Impl<ushort, int   >, device::centreToTheMeanAxis0Impl<ushort2, int2   >, device::centreToTheMeanAxis0Impl<ushort3, int3   >, device::centreToTheMeanAxis0Impl<ushort4, int4   > },
                                                      {device::centreToTheMeanAxis0Impl<ushort, float >, device::centreToTheMeanAxis0Impl<ushort2, float2 >, device::centreToTheMeanAxis0Impl<ushort3, float3 >, device::centreToTheMeanAxis0Impl<ushort4, float4 >,},
                                                      {device::centreToTheMeanAxis0Impl<ushort, double>, device::centreToTheMeanAxis0Impl<ushort2, double2>, device::centreToTheMeanAxis0Impl<ushort3, double3>, device::centreToTheMeanAxis0Impl<ushort4, double4>,}
                                                  },
                                               {
                                                      {device::centreToTheMeanAxis0Impl<short, uchar >, device::centreToTheMeanAxis0Impl<short2, uchar2 >, device::centreToTheMeanAxis0Impl<short3, uchar3 >, device::centreToTheMeanAxis0Impl<short4, uchar4 > },
                                                      {device::centreToTheMeanAxis0Impl<short, schar >, device::centreToTheMeanAxis0Impl<short2, char2 >, device::centreToTheMeanAxis0Impl<short3, char3 >, device::centreToTheMeanAxis0Impl<short4, char4 > },
                                                      {device::centreToTheMeanAxis0Impl<short, ushort>, device::centreToTheMeanAxis0Impl<short2, ushort2>, device::centreToTheMeanAxis0Impl<short3, ushort3>, device::centreToTheMeanAxis0Impl<short4, ushort4> },
                                                      {device::centreToTheMeanAxis0Impl<short, short >, device::centreToTheMeanAxis0Impl<short2, short2 >, device::centreToTheMeanAxis0Impl<short3, short3 >, device::centreToTheMeanAxis0Impl<short4, short4 > },
                                                      {device::centreToTheMeanAxis0Impl<short, int   >, device::centreToTheMeanAxis0Impl<short2, int2   >, device::centreToTheMeanAxis0Impl<short3, int3   >, device::centreToTheMeanAxis0Impl<short4, int4   > },
                                                      {device::centreToTheMeanAxis0Impl<short, float >, device::centreToTheMeanAxis0Impl<short2, float2 >, device::centreToTheMeanAxis0Impl<short3, float3 >, device::centreToTheMeanAxis0Impl<short4, float4 >,},
                                                      {device::centreToTheMeanAxis0Impl<short, double>, device::centreToTheMeanAxis0Impl<short2, double2>, device::centreToTheMeanAxis0Impl<short3, double3>, device::centreToTheMeanAxis0Impl<short4, double4>,}
                                                  },
                                               {
                                                      {device::centreToTheMeanAxis0Impl<int, uchar >, device::centreToTheMeanAxis0Impl<int2, uchar2 >, device::centreToTheMeanAxis0Impl<int3, uchar3 >, device::centreToTheMeanAxis0Impl<int4, uchar4 > },
                                                      {device::centreToTheMeanAxis0Impl<int, schar >, device::centreToTheMeanAxis0Impl<int2, char2 >, device::centreToTheMeanAxis0Impl<int3, char3 >, device::centreToTheMeanAxis0Impl<int4, char4 > },
                                                      {device::centreToTheMeanAxis0Impl<int, ushort>, device::centreToTheMeanAxis0Impl<int2, ushort2>, device::centreToTheMeanAxis0Impl<int3, ushort3>, device::centreToTheMeanAxis0Impl<int4, ushort4> },
                                                      {device::centreToTheMeanAxis0Impl<int, short >, device::centreToTheMeanAxis0Impl<int2, short2 >, device::centreToTheMeanAxis0Impl<int3, short3 >, device::centreToTheMeanAxis0Impl<int4, short4 > },
                                                      {device::centreToTheMeanAxis0Impl<int, int   >, device::centreToTheMeanAxis0Impl<int2, int2   >, device::centreToTheMeanAxis0Impl<int3, int3   >, device::centreToTheMeanAxis0Impl<int4, int4   > },
                                                      {device::centreToTheMeanAxis0Impl<int, float >, device::centreToTheMeanAxis0Impl<int2, float2 >, device::centreToTheMeanAxis0Impl<int3, float3 >, device::centreToTheMeanAxis0Impl<int4, float4 >,},
                                                      {device::centreToTheMeanAxis0Impl<int, double>, device::centreToTheMeanAxis0Impl<int2, double2>, device::centreToTheMeanAxis0Impl<int3, double3>, device::centreToTheMeanAxis0Impl<int4, double4>,}
                                                  },
                                               {
                                                      {device::centreToTheMeanAxis0Impl<float, uchar >, device::centreToTheMeanAxis0Impl<float2, uchar2 >, device::centreToTheMeanAxis0Impl<float3, uchar3 >, device::centreToTheMeanAxis0Impl<float4, uchar4 > },
                                                      {device::centreToTheMeanAxis0Impl<float, schar >, device::centreToTheMeanAxis0Impl<float2, char2 >, device::centreToTheMeanAxis0Impl<float3, char3 >, device::centreToTheMeanAxis0Impl<float4, char4 > },
                                                      {device::centreToTheMeanAxis0Impl<float, ushort>, device::centreToTheMeanAxis0Impl<float2, ushort2>, device::centreToTheMeanAxis0Impl<float3, ushort3>, device::centreToTheMeanAxis0Impl<float4, ushort4> },
                                                      {device::centreToTheMeanAxis0Impl<float, short >, device::centreToTheMeanAxis0Impl<float2, short2 >, device::centreToTheMeanAxis0Impl<float3, short3 >, device::centreToTheMeanAxis0Impl<float4, short4 > },
                                                      {device::centreToTheMeanAxis0Impl<float, int   >, device::centreToTheMeanAxis0Impl<float2, int2   >, device::centreToTheMeanAxis0Impl<float3, int3   >, device::centreToTheMeanAxis0Impl<float4, int4   > },
                                                      {device::centreToTheMeanAxis0Impl<float, float >, device::centreToTheMeanAxis0Impl<float2, float2 >, device::centreToTheMeanAxis0Impl<float3, float3 >, device::centreToTheMeanAxis0Impl<float4, float4 >,},
                                                      {device::centreToTheMeanAxis0Impl<float, double>, device::centreToTheMeanAxis0Impl<float2, double2>, device::centreToTheMeanAxis0Impl<float3, double3>, device::centreToTheMeanAxis0Impl<float4, double4>,}
                                                  },
                                               {
                                                      {device::centreToTheMeanAxis0Impl<double, uchar >, device::centreToTheMeanAxis0Impl<double2, uchar2 >, device::centreToTheMeanAxis0Impl<double3, uchar3 >, device::centreToTheMeanAxis0Impl<double4, uchar4 > },
                                                      {device::centreToTheMeanAxis0Impl<double, schar >, device::centreToTheMeanAxis0Impl<double2, char2 >, device::centreToTheMeanAxis0Impl<double3, char3 >, device::centreToTheMeanAxis0Impl<double4, char4 > },
                                                      {device::centreToTheMeanAxis0Impl<double, ushort>, device::centreToTheMeanAxis0Impl<double2, ushort2>, device::centreToTheMeanAxis0Impl<double3, ushort3>, device::centreToTheMeanAxis0Impl<double4, ushort4> },
                                                      {device::centreToTheMeanAxis0Impl<double, short >, device::centreToTheMeanAxis0Impl<double2, short2 >, device::centreToTheMeanAxis0Impl<double3, short3 >, device::centreToTheMeanAxis0Impl<double4, short4 > },
                                                      {device::centreToTheMeanAxis0Impl<double, int   >, device::centreToTheMeanAxis0Impl<double2, int2   >, device::centreToTheMeanAxis0Impl<double3, int3   >, device::centreToTheMeanAxis0Impl<double4, int4   > },
                                                      {device::centreToTheMeanAxis0Impl<double, float >, device::centreToTheMeanAxis0Impl<double2, float2 >, device::centreToTheMeanAxis0Impl<double3, float3 >, device::centreToTheMeanAxis0Impl<double4, float4 >,},
                                                      {device::centreToTheMeanAxis0Impl<double, double>, device::centreToTheMeanAxis0Impl<double2, double2>, device::centreToTheMeanAxis0Impl<double3, double3>, device::centreToTheMeanAxis0Impl<double4, double4>,}
                                                  },
                                             };

    const int sdepth = X.depth();
    const int wdepth = dtype == -1 ? X.depth() : CV_MAT_DEPTH(dtype);
//    const int wdepth = dtype == -1 ? std::max(std::max(X.depth(), mu_X.depth()), CV_32F) : CV_MAT_DEPTH(dtype); // if not specified by dtype, any intertype will be converted to float.

    GpuMat tmp(X.size(), wdepth);

    function_type fun = funcs[sdepth][wdepth][X.channels()-1];

    fun(X, mu_X, tmp, stream);

    tmp.copyTo(dst, stream);

}


} // cuda

} // cv
