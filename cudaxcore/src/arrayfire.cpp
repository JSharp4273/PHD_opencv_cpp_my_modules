#include "opencv2/cudaxcore/arrayfire.hpp"
#include "opencv2/cudaxcore/utils.hpp"

#include <arrayfire.h>

namespace cv
{

namespace cuda
{

namespace arrayfire
{

namespace
{

af_dtype get_dtype(const int wtype)
{
    const int wdepth = CV_MAT_DEPTH(wtype);
    const int cn = CV_MAT_CN(wtype);

    af_dtype dst_depth(u8);
//    bool unsuported(false);
//    int advise_depth(CV_8U);

    switch (wdepth)
    {
    case CV_8S:
//        unsuported = true;
        dst_depth = s16;
//        advise_depth = CV_16S;
        break;

    case CV_16U:
        dst_depth = u16;
        break;

    case CV_16S:
        dst_depth = s16;
        break;

    case CV_32S:
        dst_depth = s32;
        break;

    case CV_32F:
        dst_depth = cn == 2 ? c32 : f32;
        break;

    case CV_64F:
        dst_depth = cn == 2 ? c64 : f64;
        break;
    }

    return dst_depth;
}

inline af::dim4 get_dims(const int& type, const Size& size)
{
    const int wdepth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    return (wdepth > CV_32S && cn<=2) || (wdepth < CV_32F && cn == 1) ? af::dim4(size.height, size.width) : af::dim4(size.height, size.width, cn);
}

bool get_cv_type(const af_dtype& type, int& cv_type, af_dtype& advise_depth)
{
//    int cv_type(CV_8U);

    bool unsuported_type(false);
//    af_dtype advise_depth(u8);

    cv_type = CV_8U;
    advise_depth = u8;

    switch (type)
    {

    case u8:
        break;

    case b8:
        unsuported_type = true;
        advise_depth = u8;
        break;
#if AF_API_VERSION >= 32
    case u16:
        cv_type = CV_16U;
        break;

    case s16:
        cv_type = CV_16S;
        break;
#endif

#if AF_API_VERSION >= 37
    case f16:
        unsuported_type = true;
        advise_depth = f32;
        cv_type = CV_32F;
        break;
#endif

    case s32:
        cv_type = CV_32S;
        break;

    case u32:
        unsuported_type = true;
        advise_depth = f32;
        cv_type = CV_32F;
        break;

    case f32:
        cv_type = CV_32F;
        break;

    case c32:
        cv_type = CV_32F;
        break;

    case s64:
        unsuported_type = true;
        advise_depth = f64;
        cv_type = CV_64F;
        break;

    case u64:
        unsuported_type = true;
        advise_depth = f64;
        cv_type = CV_64F;
        break;

    case f64:
        cv_type = CV_64F;
        break;

    case c64:
        cv_type = CV_64F;
        break;

    default:
        unsuported_type = true;
        advise_depth = f64;
        cv_type = CV_64F;
        break;
    }
return !unsuported_type;
}

} // anonymous



void GpuMat2Array(const GpuMat& _src, af::array& _dst, Stream& stream)
{
    _dst = af::array(get_dims(_src.type(), _src.size()), get_dtype(_src.type()));

    GpuMat src;

    if(_src.depth() == CV_8S)
        _src.convertTo(src, CV_16S, stream);
    else
        src = _src;

    bool wasLocked = _dst.isLocked();

    if(wasLocked)
    {
        _dst.unlock();
    }

    RowWise2ColWise(_src.data, _src.step, _dst.device<uchar>(), _src.rows * _src.elemSize(), _src.rows, _src.cols, _src.type(), stream);

    if(wasLocked)
    {
        _dst.lock();
    }
}

void Array2GpuMat(const af::array& _src, GpuMat& _dst, Stream& stream)
{
    af_dtype advised_cvt(u8);
    int cv_depth(CV_8U);
    int cn = _src.iscomplex() ? 2 : static_cast<int>(_src.dims(_src.numdims()));

    const af::array& src = get_cv_type(_src.type(), cv_depth, advised_cvt) ? _src : _src.as(advised_cvt);

    _dst.create(static_cast<int>(src.dims(0)), static_cast<int>(src.dims(1)), CV_MAKETYPE(cv_depth, cn) );

    ColWise2RowWise(_src.device<uchar>(), _dst.rows * _dst.elemSize(), _dst.data, _dst.step, _dst.rows, _dst.cols, _dst.type(), stream);
}


} // arrayfire

} // cuda

} // cv
