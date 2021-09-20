#include "precomp.hpp"

#include <sstream>
#include <fstream>

/////////////// REMOVE ME BEFORE MODULE COMPILATION ///////////////
#include "opencv2/core/ocl_genbase.hpp"
namespace cv
{

namespace ocl
{

namespace mlx
{

extern struct cv::ocl::internal::ProgramEntry centreMeanAxis0;

struct cv::ocl::internal::ProgramEntry centreMeanAxis0 =
{
    "mlx",
    "centreMeanAxis0",
    "#ifdef DOUBLE_SUPPORT\n"
    "#ifdef cl_amd_fp64\n"
    "#pragma OPENCL EXTENSION cl_amd_fp64:enable\n"
    "#elif defined (cl_khr_fp64)\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64:enable\n"
    "#endif\n"
    "#endif\n"
    "#define noconvert\n"
    "__kernel void centreToTheMeanAxis0(__global const uchar* src1ptr, const int step_src1, const int offset_src1,\n"
    "__global const uchar* src2ptr, const int step_src2, const int offset_src2,\n"
    "__global uchar* dstptr, const int step_dst, const int offset_dst, const int rows, const int cols)\n"
    "{\n"
    "int x = get_global_id(0);\n"
    "int y0 = get_global_id(1) * rowsPerWI;\n"
    "if(x>=cols || y0>=rows)\n"
    "return;\n"
    "int src1_index = mad24(y0, step_src1, mad24(x, (int)sizeof(srcT1), offset_src1));\n"
    "int src2_index = mad24(x, (int)sizeof(srcT2), offset_src2);\n"
    "int dst_index = mad24(y0, step_dst, mad24(x, (int)sizeof(dstT), offset_dst));\n"
    "for (int y = y0, y1 = min(rows, y0 + rowsPerWI); y < y1; y++, src1_index += step_src1, dst_index += step_dst)\n"
    "{\n"
    "__global const srcT1 * src1 = (__global const srcT1 *)(src1ptr + src1_index);\n"
    "__global const srcT2 * src2 = (__global const srcT2 *)(src2ptr + src2_index);\n"
    "__global dstT * dst = (__global dstT *)(dstptr + dst_index);\n"
    "*dst = convertToDT(convertToWT1(*src1) - convertToWT2(*src2));\n"
    "}\n"
    "}\n",
    "61e807be22793264eddbe34d6e9e28bd",
    nullptr
};




} // mlx

} // ocl

} // cv
/////////////////////////////////////

namespace cv
{
#ifndef CV_DISABLE_OPTIMIZATION
namespace
{

template<class T>
class ParallelCentreToTheMeanAxis0 CV_FINAL : public virtual ParallelLoopBodyInt
{
public:
    inline ParallelCentreToTheMeanAxis0(const Mat& _X, const Mat& _mu_X, Mat& _dst):
        X(_X),
        mu_X(_mu_X),
        dst(_dst)
    {}

    virtual ~ParallelCentreToTheMeanAxis0() = default;

    virtual void operator()(const int& r) const CV_OVERRIDE
    {
        hal::sub(this->X[r], this->X.step, this->mu_X[0], 0, this->dst.template ptr<T>(r), this->dst.step, this->X.cols, this->X.rows, nullptr);
    }

private:
    // using Mat_ rather than Mat ensure that all the inputs have the same type.
    // If an input does not have the proper type it is implicitly converted.
    const Mat_<T> X;
    const Mat_<T> mu_X;

    Mat& dst;
};

template<class T>
void ParallelCentreToTheMeanAxis0_worker(const Mat& X, const Mat& mu_X, Mat& dst)
{
#ifdef HAVE_TBB
   highPrioriyParallelFor(Range(0, X.rows), ParallelCentreToTheMeanAxis0<T>(X, mu_X, dst));
#else
    // Note that unless 'X' and 'mu_X' are very large matrices using a parallel for loop
    // will not improve the computational speed.
    // In such case using intrinsics without parallel loop is generally faster.
    hal::sub(X.ptr<T>(), X.step, mu_X.ptr<T>(), mu_X.step, dst.ptr<T>(), dst.step, dst.cols, dst.rows, nullptr);
#endif
}

}// anonymous

template<>
void centreToTheMeanAxis0<Mat>(const Mat& X, const Mat& mu_X, Mat& dst, const int& dtype)
{
    CV_Assert( (X.channels() == 1) && (mu_X.channels() == 1) && (X.cols == mu_X.cols) );

    const int wdepth = dtype == -1 ? std::max(std::max(X.depth(), mu_X.depth()), CV_32F) : CV_MAT_DEPTH(dtype); // if not specified by dtype, any intertype will be converted to float.

    Mat tmp(X.size(), wdepth);

    Mat tmp_X, tmp_mu_X;

    if(X.depth() == wdepth)
        tmp_X = X;
    else
        X.convertTo(tmp_X, wdepth);

    if(mu_X.depth() == wdepth)
        tmp_mu_X = mu_X;
    else
        mu_X.convertTo(tmp_mu_X, wdepth);

//    dst.create(X.size(), wdepth);

    typedef void(*function_type)(const Mat&, const Mat&, Mat& );

    static const function_type funcs[] = {ParallelCentreToTheMeanAxis0_worker<uchar>,
                                          ParallelCentreToTheMeanAxis0_worker<schar>,
                                          ParallelCentreToTheMeanAxis0_worker<ushort>,
                                          ParallelCentreToTheMeanAxis0_worker<short>,
                                          ParallelCentreToTheMeanAxis0_worker<int>,
                                          ParallelCentreToTheMeanAxis0_worker<float>,
                                          ParallelCentreToTheMeanAxis0_worker<double>
                           };

    function_type fun(nullptr);

    fun = funcs[wdepth];

    CV_Assert(fun);

    fun(tmp_X, tmp_mu_X, tmp);

    dst = tmp;
}



template<>
void centreToTheMeanAxis0<UMat>(const UMat& _X, const UMat& _mu_X, UMat& dst, const int& dtype)
{
    CV_Assert( (_X.channels() == 1) && (_mu_X.channels() == 1) && (_X.cols == _mu_X.cols) );

    const int wdepth = dtype == -1 ? std::max(std::max(_X.depth(), _mu_X.depth()), CV_32F) : dtype;

    UMat tmp(_X.size(), wdepth);

    const ocl::Device& d = ocl::Device::getDefault();

    bool doubleSupport = d.doubleFPConfig() > 0;

//    int kercn = ocl::predictOptimalVectorWidthMax(_X, _mu_X, tmp);
    const int kercn = 1;

    int rowsPerWI = d.isIntel() ? 4 : 1;
    char cvt[3][50];

    String build_opt = format(" -D dstT=%s  -D srcT1=%s -D srcT2=%s"
                         " -D convertToWT1=%s -D convertToWT2=%s -D convertToDT=%s"
                         " -D rowsPerWI=%d%s",
                         ocl::typeToStr(tmp.depth()),
                         ocl::typeToStr(_X.depth()),
                         ocl::typeToStr(_mu_X.depth()),
                         ocl::convertTypeStr(_X.depth(), wdepth, kercn, cvt[0]),
                         ocl::convertTypeStr(_mu_X.depth(), wdepth, kercn, cvt[1]),
                         ocl::convertTypeStr(wdepth, wdepth, kercn, cvt[2]),
                         rowsPerWI,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "");

//    std::ifstream fstream("/home/smile/prog/ml/mlx/opencl/precomp.cl");
//    std::stringstream sstream;

//    sstream<<fstream.rdbuf();

//    ocl::ProgramSource ps(sstream.str());

//    CV_Assert(!ps.empty());

//    std::cout<<ps.source()<<std::endl;

//    ocl::Kernel k("centreToTheMeanAxis0",ps, build_opt);

    ocl::Kernel k("centreToTheMeanAxis0",ocl::mlx::centreMeanAxis0, build_opt);

    k.args(ocl::KernelArg::ReadOnlyNoSize(_X), ocl::KernelArg::ReadOnlyNoSize(_mu_X), ocl::KernelArg::WriteOnly(tmp));

    size_t globalsize[2] = { static_cast<size_t>(_X.cols / kercn), (static_cast<size_t>(_X.rows + rowsPerWI - 1) / rowsPerWI) };

    CV_Assert(k.run(2, globalsize, nullptr, false));

    dst = tmp;
}
#else

template<>
void centreToTheMeanAxis0<Mat>(const Mat& X, const Mat& mu_X, Mat& dst, const int& dtype)
{
    CV_Assert( (X.channels() == 1) && (mu_X.channels() == 1) && (X.cols == mu_X.cols) );    

#ifdef HAVE_LAPACK
    // Note that: when working with Mat container, due to the heavy optimisation used in lapack the folowing code is faster than using repeat and subtract.
    gemm(T::ones(X.rows, 1, wdepth), mu_X, -1., X, 1., dst);
#else
    const int wdepth = dtype == -1 ? std::max(X.depth(), CV_32F) : dtype;
    repeat(mu_X, X.rows, 1, tmp);
    subtract(X, tmp, X, noArray(), wdepth);
#endif
}

template<>
void centreToTheMeanAxis0<UMat>(const UMat& _X, const UMat& _mu_X, UMat& dst, const int& dtype)
{
    CV_Assert( (_X.channels() == 1) && (_mu_X.channels() == 1) && (_X.cols == _mu_X.cols) );

    const int wdepth = dtype == -1 ? std::max(X.depth(), CV_32F) : dtype;
    repeat(mu_X, X.rows, 1, tmp);
    subtract(X, tmp, X, noArray(), wdepth);
}

#endif

namespace
{

template<class T>
inline void centreToTheMeanAxis0_(const T& X, T& dst, const int& dtype)
{
    if(X.rows == 1 || X.cols == 1)
        subtract(X, mean(X), dst, noArray(), dtype);
    else
    {
        T mu_X;

        reduce(X, mu_X, 0, REDUCE_AVG, dtype);
        centreToTheMeanAxis0(X, mu_X, dst);
    }
}

} // anonymous


template<>
void centreToTheMeanAxis0<Mat>(const Mat& X, Mat& dst, const int& dtype)
{
    centreToTheMeanAxis0_(X, dst, dtype);
}

template<>
void centreToTheMeanAxis0<UMat>(const UMat& X, UMat& dst, const int& dtype)
{
    centreToTheMeanAxis0_(X, dst, dtype);
}


} // cv
