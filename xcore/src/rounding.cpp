#include "opencv2/xcore.hpp"
#include "opencv2/xcore/template/arguments_io.hpp"
#include "opencv2/xcore/template/hal.hpp"
#include "opencv2/cvconfig.h"
#include "opencv2/core/simd_intrinsics.hpp"
#include <functional>


namespace cv
{

//namespace xcore
//{

namespace
{

template<class T>
void floor_(InputArray& _src, OutputArray& _dst, int dtype)
{
    if(_src.depth()<CV_32F)
        _src.copyTo(_dst);

    //    int i = (int)value;
    //    return i - (i > value);

    T src = getInput<T>(_src);

    T mask, tmp;

    const int wdepth = _src.depth();

    src.convertTo(tmp, CV_32S);
    tmp.convertTo(tmp, wdepth);

    compare(tmp, src, mask, CMP_GT);
    bitwise_and(mask, 1, mask);

    mask.convertTo(mask, wdepth);

    subtract(tmp, mask, _dst, noArray(), dtype);
}

template<>
void floor_<Mat>(InputArray& _src, OutputArray& _dst, int dtype)
{
    if(_src.depth()<CV_32F)
        _src.copyTo(_dst);

    Mat src = _src.getMat();

    Mat dst;

    int stype = _src.type();
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int ddepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wdepth = ddepth != CV_32S ? sdepth : ddepth;
    int wtype = CV_MAKETYPE(wdepth, cn);

    if(cn>1)
    {
        Mat tmp;

        if(!src.isContinuous() || src.isSubmatrix())
        {
            tmp = src;

            src.release();

            tmp.copyTo(src);
        }

        tmp = Mat(src.rows, src.cols * cn, sdepth, src.data);
        src = tmp;
    }

    dst.create(src.size(), wdepth);

    if(ddepth == CV_32S)
    {
        wdepth == CV_32F ?
                    hal::floor32f(src.ptr<float>(), src.step, dst.ptr<int>(), dst.step, dst.cols, dst.rows) :
                    hal::floor64f(src.ptr<double>(), src.step, dst.ptr<int>(), dst.step, dst.cols, dst.rows);
    }
    else
    {
        wdepth == CV_32F ?
                    hal::floor32f(src.ptr<float>(), src.step, dst.ptr<float>(), dst.step, dst.cols, dst.rows) :
                    hal::floor64f(src.ptr<double>(), src.step, dst.ptr<double>(), dst.step, dst.cols, dst.rows);
    }

    if(cn>1)
    {
        Mat tmp(dst.rows, dst.cols / cn, wtype, dst.data);
        dst = tmp;
    }

    ddepth != sdepth ? dst.convertTo(_dst, ddepth) : dst.copyTo(_dst);
}

template<class T>
void ceil_(InputArray& _src, OutputArray& _dst, int dtype)
{
    if(_src.depth()<CV_32F)
        _src.copyTo(_dst);

    //    int i = (int)value;
    //    return i + (i < value);

    T src = getInput<T>(_src);

    T mask, tmp;

    const int wdepth = _src.depth();

    src.convertTo(tmp, CV_32S);
    tmp.convertTo(tmp, wdepth);

    compare(tmp, src, mask, CMP_LT);
    bitwise_and(mask, 1, mask);

    mask.convertTo(mask, wdepth);

    add(tmp, mask, _dst, noArray(), dtype);
}

template<>
void ceil_<Mat>(InputArray& _src, OutputArray& _dst, int dtype)
{
    if(_src.depth()<CV_32F)
        _src.copyTo(_dst);

    Mat src = _src.getMat();

    Mat dst;

    int stype = _src.type();
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int ddepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wdepth = ddepth != CV_32S ? sdepth : ddepth;
    int wtype = CV_MAKETYPE(wdepth, cn);

    if(cn>1)
    {
        Mat tmp;

        if(!src.isContinuous() || src.isSubmatrix())
        {
            tmp = src;

            src.release();

            tmp.copyTo(src);
        }

        tmp = Mat(src.rows, src.cols * cn, sdepth, src.data);
        src = tmp;
    }

    dst.create(src.size(), wdepth);

    if(ddepth == CV_32S)
    {
        wdepth == CV_32F ?
                    hal::ceil32f(src.ptr<float>(), src.step, dst.ptr<int>(), dst.step, dst.cols, dst.rows) :
                    hal::ceil64f(src.ptr<double>(), src.step, dst.ptr<int>(), dst.step, dst.cols, dst.rows);
    }
    else
    {
        wdepth == CV_32F ?
                    hal::ceil32f(src.ptr<float>(), src.step, dst.ptr<float>(), dst.step, dst.cols, dst.rows) :
                    hal::ceil64f(src.ptr<double>(), src.step, dst.ptr<double>(), dst.step, dst.cols, dst.rows);
    }

    if(cn>1)
    {
        Mat tmp(dst.rows, dst.cols / cn, wtype, dst.data);
        dst = tmp;
    }

    ddepth != sdepth ? dst.convertTo(_dst, ddepth) : dst.copyTo(_dst);
}

namespace
{

template<class T>
inline void castToIntAndBackToType(uchar* data, int n)
{
    T* begin = reinterpret_cast<T*>(data);
    T* end = begin + n;

    std::transform(begin, end, begin,[](const T& v)->T{ return static_cast<T>(static_cast<int>(v));});
}

}

template<class T>
void round_(InputArray& _src, OutputArray& _dst, int dtype)
{
    typedef void(*function_type)(uchar*,int);

    static const function_type funcs[2] = {castToIntAndBackToType<float>, castToIntAndBackToType<double>};

    if(_src.depth()<CV_32F)
        _src.copyTo(_dst);

//(int)(value + (value >= 0 ? 0.5 : -0.5));

    T src = getInput<T>(_src);

    T mask, not_mask, tmp;

    const int wdepth = _src.depth();

    //mask <- value >= 0
    compare(src, 0., mask, CMP_GE);

    //not_mask <- value < 0
    bitwise_not(mask, not_mask);

    bitwise_and(mask, 1., mask);
    bitwise_and(not_mask, 1., not_mask);

    mask.convertTo(mask, wdepth);
    not_mask.convertTo(not_mask, wdepth);

    //mask <- mask * 0.5
    multiply(mask, 0.5, mask);
    //not_mask <- not_mask * -0.5
    multiply(not_mask, -0.5, not_mask);

    add(mask, not_mask, mask);

    add(src, mask, tmp);

    Mat tmp2 = toMat(tmp);

    function_type fun = funcs[tmp2.depth()-CV_32F];

#ifdef HAVE_HIGH_PRIORITY_PARFOR
    highPrioriyParallelFor( Range(0, tmp2.rows),[&tmp2, fun](const int& r)->void
    {
#else
        for(int r=0; r<tmp2.rows; r++)
        {
#endif
            fun(tmp2.ptr(r), tmp2.cols);
#ifdef HAVE_HIGH_PRIORITY_PARFOR
});
#else
        }
#endif

    dtype == -1 ? tmp2.copyTo(_dst) : tmp2.convertTo(_dst, CV_MAT_DEPTH(dtype));
}

template<>
void round_<Mat>(InputArray& _src, OutputArray& _dst, int dtype)
{
    if(_src.depth()<CV_32F)
        _src.copyTo(_dst);

    Mat src = _src.getMat();

    Mat dst;

    int stype = _src.type();
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int ddepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wdepth = ddepth != CV_32S ? sdepth : ddepth;
    int wtype = CV_MAKETYPE(wdepth, cn);

    if(cn>1)
    {
        Mat tmp;

        if(!src.isContinuous() || src.isSubmatrix())
        {
            tmp = src;

            src.release();

            tmp.copyTo(src);
        }

        tmp = Mat(src.rows, src.cols * cn, sdepth, src.data);
        src = tmp;
    }

    dst.create(src.size(), wdepth);

    if(ddepth == CV_32S)
    {
        wdepth == CV_32F ?
                    hal::round32f(src.ptr<float>(), src.step, dst.ptr<int>(), dst.step, dst.cols, dst.rows) :
                    hal::round64f(src.ptr<double>(), src.step, dst.ptr<int>(), dst.step, dst.cols, dst.rows);
    }
    else
    {
        wdepth == CV_32F ?
                    hal::round32f(src.ptr<float>(), src.step, dst.ptr<float>(), dst.step, dst.cols, dst.rows) :
                    hal::round64f(src.ptr<double>(), src.step, dst.ptr<double>(), dst.step, dst.cols, dst.rows);
    }

    if(cn>1)
    {
        Mat tmp(dst.rows, dst.cols / cn, wtype, dst.data);
        dst = tmp;
    }

    ddepth != sdepth ? dst.convertTo(_dst, ddepth) : dst.copyTo(_dst);
}

} // anonymous

#define IMPL_SPEC_FUN(name)\
void name(InputArray _src, OutputArray _dst, int dtype)\
{\
    if(_dst.isUMat())\
    {\
        bool ok(true);\
\
        try {\
            name ## _<UMat>(_src, _dst, dtype);\
        }\
        catch (const cv::Exception&)\
        {\
            ok = false;\
        }\
\
        if(ok)\
            return;\
    }\
    name ## _<Mat>(_src, _dst, dtype);\
}

IMPL_SPEC_FUN(ceil)
IMPL_SPEC_FUN(floor)
IMPL_SPEC_FUN(round)

#undef IMPL_SPEC_FUN


//} // xcore

} // cv
