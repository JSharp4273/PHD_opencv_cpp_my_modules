#include "opencv2/cudaxcore.hpp"

namespace cv
{

namespace cuda
{

namespace device
{

template<class SrcType, class DstType>
void CeilImpl(const GpuMat& _src, GpuMat& _dst, Stream& _stream);

template<class SrcType, class DstType>
void FloorImpl(const GpuMat& _src, GpuMat& _dst, Stream& _stream);

template<class SrcType, class DstType>
void RoundImpl(const GpuMat& _src, GpuMat& _dst, Stream& _stream);

} // device

#define IMPL_ROUND_FUN(name, minus, maj)\
\
void minus ## name(InputArray _src, OutputArray _dst, int dtype, Stream& stream)\
{\
    typedef void (*function_type)(const GpuMat&, GpuMat&, Stream&);\
\
    static const function_type funcs[2][3] = {\
        {device::maj ## name ## Impl<float, int>, device::maj ## name ## Impl<float, float>, nullptr},\
        {device::maj ## name ## Impl<double, int>, nullptr, device::maj ## name ## Impl<double, double>}\
    };\
\
    if(_src.depth()<CV_32F)\
        _src.copyTo(_dst);\
\
    GpuMat src = _src.getGpuMat();\
    GpuMat dst;\
\
\
    int stype = _src.type();\
    int sdepth = CV_MAT_DEPTH(stype);\
    int cn = CV_MAT_CN(stype);\
    int ddepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);\
    int wdepth = ddepth == CV_32S ? ddepth : sdepth;\
    int wtype = CV_MAKETYPE(wdepth, cn);\
\
    if(_dst.empty())\
        _dst.create(src.size(), wtype);\
\
    dst = _dst.getGpuMat();\
\
    if(dst.data == src.data)\
    {\
        GpuMat tmp(dst.size(), dst.type(), Scalar::all(0.));\
        dst = tmp;\
    }\
\
    function_type fun = funcs[sdepth-CV_32F][wdepth-CV_32S];\
\
    CV_Assert(fun);\
\
    fun(src, dst, stream);\
\
    dst.copyTo(_dst, stream);\
}

IMPL_ROUND_FUN(eil,c,C)
IMPL_ROUND_FUN(loor,f,F)
IMPL_ROUND_FUN(ound,r,R)

} // cuda

} // cv
