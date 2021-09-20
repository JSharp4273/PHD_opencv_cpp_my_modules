#if 0
#include "opencv2/xcore.hpp"
#include "opencv2/xcore/template/arguments_io.hpp"
#include "opencv2/xcore/template/hal.hpp"
#include "opencv2/xcore/template/intrin.hpp"

namespace cv
{

namespace
{

typedef void(*trinary_function_t)(const uchar*, size_t, const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int);
typedef void(*trinary_weighted_function_t)(const uchar*, size_t, const uchar*, size_t, const uchar*, size_t, const uchar*, size_t, const uchar*, size_t, const uchar*, size_t, uchar*, size_t, int, int);

class ParallelProcessMask_BitwiseAnd : public ParallelLoopBodyInt
{
public:

    inline ParallelProcessMask_BitwiseAnd(Mat& _mask):
        mask(_mask)
#if CV_SIMD
      , cols_vec(_mask.cols - (_mask.cols%static_cast<int>(v_uint8::nlanes))), inc(static_cast<int>(v_uint8::nlanes))
#endif
    {}

    virtual ~ParallelProcessMask_BitwiseAnd() = default;

    virtual void operator()(const int& r) const
    {
#if CV_SIMD
        static const v_uint8 v_one = vx_setall_u8(1);
#endif

        int c=0;

        uchar* it = this->mask.ptr(r);

#if CV_SIMD
        for(; c<this->cols_vec; c+=inc, it+=inc)
            vx_store(it, vx_load(it) & v_one);
#endif

#if CV_ENABLE_UNROLLED
        for(; c<this->mask.cols-4; c+=4, it+=4)
        {
            uchar v0 = it[0];
            uchar v1 = it[1];

            v0 &= 1;
            v1 &= 1;

            it[0] = v0;
            it[1] = v1;

            v0 = it[2];
            v1 = it[3];

            v0 &= 1;
            v1 &= 1;

            it[2] = v0;
            it[3] = v1;
        }
#endif

        for(;c<this->mask.cols; c++, it++)
            *it &= 1;
    }

private:

    Mat& mask;
#if CV_SIMD
    int cols_vec;
    int inc;
#endif

};

class ParallelProcessMask_DivideByMaxBitShit : public ParallelLoopBodyInt
{
public:

    inline ParallelProcessMask_DivideByMaxBitShit(Mat& _mask, double& _max):
        mask(_mask),
        shift(static_cast<uchar>(log(_max)/log(2.)))
  #if CV_SIMD
        , cols_vec(_mask.cols - (_mask.cols%static_cast<int>(v_uint8::nlanes))), inc(static_cast<int>(v_uint8::nlanes))
  #endif
    {}

    virtual ~ParallelProcessMask_DivideByMaxBitShit() = default;

    virtual void operator()(const int& r) const
    {

        int c=0;

        uchar* it = this->mask.ptr(r);

#if CV_SIMD
        for(; c<this->cols_vec; c+=inc, it+=inc)
            vx_store(it, vx_load(it) >> static_cast<int>(shift));
#endif

#if CV_ENABLE_UNROLLED
        for(; c<this->mask.cols-4; c+=4, it+=4)
        {
            uchar v0 = it[0];
            uchar v1 = it[1];

            v0 >>= shift;
            v1 >>= shift;

            it[0] = v0;
            it[1] = v1;

            v0 = it[2];
            v1 = it[3];

            v0 >>= shift;
            v1 >>= shift;

            it[2] = v0;
            it[3] = v1;
        }
#endif

        for(;c<this->mask.cols; c++, it++)
            *it >>= shift;
    }

private:

    Mat& mask;
    const uchar shift;
#if CV_SIMD
    int cols_vec;
    int inc;
#endif
};


class ParallelProcessMask_DivideByMax : public ParallelLoopBodyInt
{
public:

    inline ParallelProcessMask_DivideByMax(Mat& _mask, const double& _max):
        mask(_mask),
        den(static_cast<uchar>(_max))
#if CV_SIMD
      , cols_vec(_mask.cols - (_mask.cols%static_cast<int>(v_uint8::nlanes))), inc(static_cast<int>(v_uint8::nlanes))
#endif
    {}

    virtual ~ParallelProcessMask_DivideByMax() = default;

    virtual void operator()(const int& r) const
    {
#if CV_SIMD
        static const v_uint8x32 v_den = vx_setall_u8(this->den);
#endif

        int c=0;

        uchar* it = this->mask.ptr(r);

#if CV_SIMD
        for(; c<this->cols_vec; c+=inc, it+=inc)
            vx_store(it, vx_load(it) / v_den);
#endif

#if CV_ENABLE_UNROLLED
        for(; c<this->mask.cols-4; c+=4, it+=4)
        {
            uchar v0 = it[0];
            uchar v1 = it[1];

            v0 /= den;
            v1 /= den;

            it[0] = v0;
            it[1] = v1;

            v0 = it[2];
            v1 = it[3];

            v0 /= den;
            v1 /= den;

            it[2] = v0;
            it[3] = v1;
        }
#endif

        for(;c<this->mask.cols; c++, it++)
            *it /= den;
    }

private:

    Mat& mask;
    uchar den;
#if CV_SIMD
    int cols_vec;
    int inc;
#endif

};



template<class T>
Mat bufferFromScalar(const Scalar& s, const int& cols, int cn)
{
    Mat buf(1, cols, DataType<T>::depth);

    Scalar_<T> _s = s;

    T* it_buf = buf.ptr<T>();

    switch(cn)
    {
    case 1:
        std::fill_n(_s.val + 1, 3, _s.val[0]);
        break;

    case 2:
        for(int j=2;j<4;j++)
            _s.val[j] = _s.val[j-2];
        break;

    case 3:
        _s.val[3] = _s.val[0];
        break;

    default:
        break;
    }

    int c=0;

#if CV_ENABLE_UNROLLED
    for(;c<cols-4;c+=4)
    {
        T v0 = _s(0);
        T v1 = _s(1);

        it_buf[c] = v0;
        it_buf[c+1] = v0;

        v0 = _s(2);
        v1 = _s(3);

        it_buf[c+2] = v0;
        it_buf[c+3] = v0;
    }
#endif
    for(;c<cols;c++)
        it_buf[c] = _s(c%4);

    return buf;
}


Mat alignIfNeeded(const Mat& m, const int& aligned_cols)
{
    Mat tmp;

    if(!m.isContinuous() || m.isSubmatrix())
        m.copyTo(tmp);
    else
        tmp = m;

    tmp = tmp.reshape(1);

    if(tmp.cols != aligned_cols)
    {
        Mat buffer = Mat::zeros(tmp.rows, aligned_cols, tmp.depth());

        tmp.copyTo(buffer(Rect(0, 0, tmp.cols, tmp.rows)));

        tmp = buffer;
    }

    return tmp;
}

Mat getAlignedInput(InputArray& _src, const int& cols_aligned, const int& cn, const int& wdepth)
{
    return isScalar(_src) ? wdepth == CV_32F ? bufferFromScalar<float>(getScalar(_src), cols_aligned, cn) : bufferFromScalar<double>(getScalar(_src), cols_aligned, cn) : alignIfNeeded(_src.getMat(), cols_aligned);
}

Mat prepareMask(InputArray& _mask, const int& rows, const int& cols, const int& cn, const int& wdepth)
{
    Mat mask = _mask.getMat();

    compare(mask, 0., mask, CMP_GT);

    double mx(0.);

    minMaxLoc(mask, nullptr, &mx);

    if(mx==255.)
    {
        highPrioriyParallelFor(Range(0, rows), ParallelProcessMask_BitwiseAnd(mask));
    }
    else
    {
        if(mx!=1.)
        {
            if((static_cast<int>(mx)%8)==0)
                highPrioriyParallelFor(Range(0, rows), ParallelProcessMask_DivideByMaxBitShit(mask, mx));
            else
                highPrioriyParallelFor(Range(0, rows), ParallelProcessMask_DivideByMax(mask, mx));
        }
        else
            mask.convertTo(mask, wdepth);
    }

    if(cn>1)
    {
        std::vector<Mat> tmp(3, mask);
        merge(tmp, mask);

        mask = mask.reshape(1);
    }

    return alignIfNeeded(mask, cols);
}

template<class T>
class ParallelApplyFMX : public ParallelLoopBody
{
public:

    inline ParallelApplyFMX(const trinary_function_t _fun, const std::vector<Mat>& _src):
        fun(_fun),
        src1_ptr(_src.at(0).data),
        src1_step(_src.at(0).rows == 1 ? 0 :_src.at(0).step),
        src2_ptr(_src.at(1).data),
        src2_step(_src.at(1).rows == 1 ? 0 :_src.at(1).step),
        src3_ptr(_src.at(2).data),
        src3_step(_src.at(2).rows == 1 ? 0 :_src.at(2).step),
        dst_ptr(_src.at(3).data),
        dst_step(_src.at(3).rows == 1 ? 0 :_src.at(3).step),
        mask_ptr(nullptr),
        mask_step(0),
        cols(_src.at(0).cols)
    {
        if(_src.size() == 5)
        {
            Mat tmp = _src.back();

            this->mask_ptr = tmp.data;
            this->mask_step = tmp.step;
        }
    }

    virtual ~ParallelApplyFMX() = default;

    virtual void operator()(const Range& range) const
    {
        const uchar* local_src1_ptr = this->src1_ptr + range.start * this->src1_step;
        const uchar* local_src2_ptr = this->src2_ptr + range.start * this->src2_step;
        const uchar* local_src3_ptr = this->src3_ptr + range.start * this->src3_step;;
        uchar* local_dst_ptr = this->dst_ptr + range.start * this->dst_step;

        const uchar* local_mask_ptr = this->mask_ptr ? this->mask_ptr + range.start * this->mask_step : nullptr;

        fun(local_src1_ptr, this->src1_step, local_src2_ptr, this->src2_step, local_src3_ptr, this->src3_step, local_dst_ptr, this->dst_step, this->cols, range.size());
        if(local_dst_ptr)
        {
            Scalar scales = Scalar::all(1.);
            hal::mul(reinterpret_cast<const T*>(local_dst_ptr), this->dst_step, reinterpret_cast<const T*>(local_mask_ptr), this->mask_step, reinterpret_cast<T*>(local_dst_ptr), this->dst_step, this->cols, range.size(), &scales);
        }
    }

private:

    const trinary_function_t fun;
    const uchar* src1_ptr;
    size_t src1_step;

    const uchar* src2_ptr;
    size_t src2_step;

    const uchar* src3_ptr;
    size_t src3_step;

    uchar* dst_ptr;
    size_t dst_step;

    const uchar* mask_ptr;
    size_t mask_step;

    const int cols;

};

template<class T>
class ParallelApplyWeightedFMX : public ParallelLoopBody
{
public:

    inline ParallelApplyWeightedFMX(const trinary_weighted_function_t _fun, const std::vector<Mat>& _src):
        fun(_fun),
        w1_ptr(_src.at(0).data),
        w1_step(_src.at(0).rows == 1 ? 0 :_src.at(0).step),
        src1_ptr(_src.at(1).data),
        src1_step(_src.at(1).rows == 1 ? 0 :_src.at(1).step),
        w2_ptr(_src.at(2).data),
        w2_step(_src.at(2).rows == 1 ? 0 :_src.at(2).step),
        src2_ptr(_src.at(3).data),
        src2_step(_src.at(3).rows == 1 ? 0 :_src.at(3).step),
        w3_ptr(_src.at(4).data),
        w3_step(_src.at(4).rows == 1 ? 0 :_src.at(4).step),
        src3_ptr(_src.at(5).data),
        src3_step(_src.at(5).rows == 1 ? 0 :_src.at(5).step),
        dst_ptr(_src.at(6).data),
        dst_step(_src.at(6).rows == 1 ? 0 :_src.at(6).step),
        mask_ptr(nullptr),
        mask_step(0),
        cols(_src.at(0).cols)
    {
        if(_src.size() == 5)
        {
            Mat tmp = _src.back();

            this->mask_ptr = tmp.data;
            this->mask_step = tmp.step;
        }
    }

    virtual ~ParallelApplyWeightedFMX() = default;

    virtual void operator()(const Range& range) const
    {
        const uchar* local_w1_ptr = this->src1_ptr + range.start * this->w1_step;
        const uchar* local_src1_ptr = this->src1_ptr + range.start * this->src1_step;
        const uchar* local_w2_ptr = this->src1_ptr + range.start * this->w2_step;
        const uchar* local_src2_ptr = this->src2_ptr + range.start * this->src2_step;
        const uchar* local_w3_ptr = this->src3_ptr + range.start * this->w3_step;
        const uchar* local_src3_ptr = this->src3_ptr + range.start * this->src3_step;;
        uchar* local_dst_ptr = this->dst_ptr + range.start * this->dst_step;

        const uchar* local_mask_ptr = this->mask_ptr ? this->mask_ptr + range.start * this->mask_step : nullptr;

        fun(local_w1_ptr, this->w1_step,
            local_src1_ptr, this->src1_step,
            local_w2_ptr, this->w2_step,
            local_src2_ptr, this->src2_step,
            local_w3_ptr, this->w3_step,
            local_src3_ptr, this->src3_step,
            local_dst_ptr, this->dst_step, this->cols, range.size());

        if(local_dst_ptr)
        {
            Scalar scales = Scalar::all(1.);
            hal::mul(reinterpret_cast<const T*>(local_dst_ptr), this->dst_step, reinterpret_cast<const T*>(local_mask_ptr), this->mask_step, reinterpret_cast<T*>(local_dst_ptr), this->dst_step, this->cols, range.size(), &scales);
        }
    }

private:

    const trinary_weighted_function_t fun;

    const uchar* w1_ptr;
    size_t w1_step;

    const uchar* src1_ptr;
    size_t src1_step;

    const uchar* w2_ptr;
    size_t w2_step;

    const uchar* src2_ptr;
    size_t src2_step;

    const uchar* w3_ptr;
    size_t w3_step;

    const uchar* src3_ptr;
    size_t src3_step;

    uchar* dst_ptr;
    size_t dst_step;

    const uchar* mask_ptr;
    size_t mask_step;

    const int cols;

};



bool ocl_fmx_op(InputArray& _w1, InputArray& _src1,
            InputArray& _w2, InputArray& _src2,
            InputArray& _w3, InputArray& _src3,
            OutputArray& _dst, InputArray& _mask, int dtype, int oclop)
{
    return false;
}


void fmx_op(InputArray& _w1, InputArray& _src1,
            InputArray& _w2, InputArray& _src2,
            InputArray& _w3, InputArray& _src3,
            OutputArray& _dst, InputArray& _mask, int dtype, trinary_function_t* tab, trinary_weighted_function_t* tab_weighted, int oclop)
{
    CV_Assert_7(_w1.kind() == _InputArray::NONE || isScalar(_w1) || isMatrix(_w1),
                isScalar(_src1) || isMatrix(_src1),
                _w2.kind() == _InputArray::NONE || isScalar(_w2) || isMatrix(_w2),
                isScalar(_src2) || isMatrix(_src2),
                _w3.kind() == _InputArray::NONE || isScalar(_w3) || isMatrix(_w3),
                isScalar(_src3) || isMatrix(_src3),
                _mask.kind() == _InputArray::NONE || isScalar(_mask) || isMatrix(_mask)
                );

    if(_dst.isUMat() && ocl_fmx_op(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, oclop))
        return;

    bool have_mask = !_mask.empty();
    bool have_weights = !_w1.empty() || !_w2.empty() || !_w3.empty();

    Mat dst;

    const int sdepth = std::max(std::max(_src1.depth(), _src2.depth()), _src3.depth());
    const int wdepth = std::max(sdepth, CV_32F);
    const int cn = !isScalar(_src1) ? _src1.channels() : !isScalar(_src2) ? _src2.channels() : !isScalar(_src3) ? _src3.channels() : -1;
    const int wtype = CV_MAKETYPE(wdepth, cn);

    const Size sz = !isScalar(_src1) ? _src1.size() : !isScalar(_src2) ? _src2.size() : !isScalar(_src3) ? _src3.size() : Size(-1,1);

    const int rows = sz.area() < 0 ? 0 : sz.height;
    const int cols = sz.area() < 0 ? 0 : sz.width;

    const int aligned_cols = alignSize(cols * cn,  wdepth == CV_32F ? CV_SIMD_WIDTH / sizeof(float) : CV_SIMD_WIDTH / sizeof(double) );

//    dst = Mat::zeros(_src1.size(), wtype);

    bool is_src1_a_scalar = isScalar(_src1);
    bool is_src2_a_scalar = isScalar(_src2);
    bool is_src3_a_scalar = isScalar(_src3);

    std::vector<Mat> args;

    int cnt_scalar = static_cast<int>(is_src1_a_scalar) + static_cast<int>(is_src2_a_scalar) + static_cast<int>(is_src3_a_scalar);

    if(!have_weights)
    {

        trinary_function_t fun = tab[wdepth-CV_32F];


//        int cnt_matrix = 3 - cnt_scalar;

        if(cnt_scalar == 3)
        {
            fun = tab[CV_64F - CV_32F]; // should be 1 (6-5).

            Scalar s1 = getScalar(_src1);
            Scalar s2 = getScalar(_src2);
            Scalar s3 = getScalar(_src3);

            Scalar dst;

            const uchar* ptr_s1 = reinterpret_cast<const uchar*>(s1.val);
            const uchar* ptr_s2 = reinterpret_cast<const uchar*>(s2.val);
            const uchar* ptr_s3 = reinterpret_cast<const uchar*>(s3.val);

            uchar* ptr_dst = reinterpret_cast<uchar*>(dst.val);

            fun(ptr_s1, 0, ptr_s2, 0, ptr_s3, 0, ptr_dst, 0, 4, 1);

            if(have_mask)
            {
                Scalar mask = getScalar(_mask);

                for(int i=0; i<4; i++)
                    if(!mask(i))
                        dst(i) = 0;
            }

            _dst.assign(Mat(1,1,CV_MAKETYPE(sdepth,cn), dst));
        }
        else
        {
            args.reserve(5);

            args = {getAlignedInput(_src1, aligned_cols, cn, wdepth), getAlignedInput(_src2, aligned_cols, cn, wdepth), getAlignedInput(_src3, aligned_cols, cn, wdepth), Mat::zeros(rows, aligned_cols, wdepth)};

            if(have_mask)
                args.push_back(prepareMask(_mask, rows, aligned_cols, cn, wdepth));

            if(wdepth == CV_32F)
                highPrioriyParallelFor(Range(0, rows), ParallelApplyFMX2<float>(fun, args));
            else
                highPrioriyParallelFor(Range(0, rows), ParallelApplyFMX2<double>(fun, args));
        }
    }
    else
    {
        int cnt_weights = static_cast<int>(_w1.kind() != _InputArray::NONE ) + static_cast<int>(_w2.kind() != _InputArray::NONE ) + static_cast<int>(_w3.kind() != _InputArray::NONE );

        if(cnt_weights!=3)
            CV_Error(Error::StsBadArg, "An input weight has been set to noArray(). The case is not manage, weights must either be scalars or matrix.");

        bool is_w1_a_scalar = isScalar(_w1);
        bool is_w2_a_scalar = isScalar(_w2);
        bool is_w3_a_scalar = isScalar(_w3);

        cnt_scalar += static_cast<int>(is_w1_a_scalar) + static_cast<int>(is_w2_a_scalar) + static_cast<int>(is_w3_a_scalar);

        trinary_weighted_function_t fun = tab_weighted[wdepth-CV_32F];

        if(cnt_scalar == 6)
        {
            fun = tab_weighted[CV_64F-CV_32F];

            Scalar w1 = getScalar(_w1);
            Scalar s1 = getScalar(_src1);
            Scalar w2 = getScalar(_w2);
            Scalar s2 = getScalar(_src2);
            Scalar w3 = getScalar(_w3);
            Scalar s3 = getScalar(_src3);

            Scalar dst;

            fun(reinterpret_cast<const uchar*>(w1.val), 0,
                reinterpret_cast<const uchar*>(s1.val), 0,
                reinterpret_cast<const uchar*>(w2.val), 0,
                reinterpret_cast<const uchar*>(s2.val), 0,
                reinterpret_cast<const uchar*>(w3.val), 0,
                reinterpret_cast<const uchar*>(s3.val), 0,
                reinterpret_cast<uchar*>(dst.val), 0, 4, 1);

            if(have_mask)
            {
                Scalar mask = getScalar(_mask);

                for(int i=0; i<4; i++)
                    if(!mask(i))
                        dst(i) = 0;
            }

            _dst.assign(Mat(1,1,CV_MAKETYPE(sdepth,cn), dst));
        }
        else
        {
            args.reserve(7);

            args = {getAlignedInput(_w1, aligned_cols, cn, wdepth), getAlignedInput(_src1, aligned_cols, cn, wdepth),
                    getAlignedInput(_w2, aligned_cols, cn, wdepth), getAlignedInput(_src2, aligned_cols, cn, wdepth),
                    getAlignedInput(_w3, aligned_cols, cn, wdepth), getAlignedInput(_src3, aligned_cols, cn, wdepth),
                    Mat::zeros(rows, aligned_cols, wdepth)};

            if(have_mask)
                args.push_back(prepareMask(_mask, rows, aligned_cols, cn, wdepth));

            if(wdepth == CV_32F)
                highPrioriyParallelFor(Range(0, rows), ParallelApplyWeightedFMX<float>(fun, args));
            else
                highPrioriyParallelFor(Range(0, rows), ParallelApplyWeightedFMX<double>(fun, args));
        }
    }

    if(!args.empty())
    {
        dst = have_mask ? args.at(args.size() - 2) :  args.back();

        if(CV_MAT_DEPTH(dtype)!=wdepth)
            dst.convertTo(_dst, dtype);
        else
            _dst.assign(dst);
    }

}

//void fmx_op(InputArray& _w1, InputArray& _src1, InputArray& _w2, InputArray& _src2, InputArray& _w3, InputArray& _src3, OutputArray& _dst, InputArray& _mask, int dtype, trinary_function_t* tab, int oclop)
//{
//    CV_Assert_10(CHEK_WEIGHT(_w1, _src1),
//                CHEK_WEIGHT(_w2, _src2),
//                CHEK_WEIGHT(_w3, _src3),
//                (_src1.size() == _src2.size()),
//                (_src1.size() == _src3.size()),
//                (_src1.channels() == _src2.channels()),
//                (_src1.channels() == _src3.channels()),
//                (_src1.channels() <= 4),
//                (_mask.empty() || ((_mask.size() == _src1.size()) && (_mask.depth() <= CV_8S) && (_mask.channels() == 1) ) ) ) ;

//    if(_dst.isUMat() && ocl_fmx_op(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, oclop))
//        return;

//    bool have_mask = !_mask.empty();
//    bool have_weights = !_w1.empty() || !_w2.empty() || !_w3.empty();

//    Mat src1, src2, src3, dst;

//    src1 = _src1.getMat();
//    src2 = _src2.getMat();
//    src3 = _src3.getMat();

//    const int sdepth = std::max(std::max(src1.depth(), src2.depth()), src3.depth());
//    const int wdepth = std::max(sdepth, CV_32F);
//    const int cn = src1.channels();
//    const int wtype = CV_MAKETYPE(wdepth, cn);

//    const int rows = src1.rows;
//    const int cols = src1.cols;

//    dst = Mat::zeros(src1.size(), wtype);

//    const trinary_function_t fun = tab[wdepth-CV_32F];


//    int cnt_scalar = static_cast<int>(_w1.isMatx()) + static_cast<int>(_w2.isMatx()) + static_cast<int>(_w3.isMatx());

//    if(cnt_scalar == 3)
//    {
//        Scalar w1 = getScalar(_w1);
//        Scalar w2 = getScalar(_w2);
//        Scalar w3 = getScalar(_w3);

//        switch (cn)
//        {
//        case 1:
//            for(int i=1;i<4;i++)
//            {
//                w1(i) = w1(0);
//                w2(i) = w2(0);
//                w3(i) = w3(0);
//            }
//            break;

//        case 2:
//            for(int i=2;i<4;i++)
//            {
//                w1(i) = w1(i-2);
//                w2(i) = w2(i-2);
//                w3(i) = w3(i-2);
//            }
//            break;
//        case 3:
//            w1(3) = w1(0);
//            w2(3) = w2(0);
//            w3(3) = w3(0);
//            break;
//        default:
//            break;
//        }

//        src1.convertTo(src1, wdepth);
//        src2.convertTo(src2, wdepth);
//        src3.convertTo(src3, wdepth);

//        if(wdepth==CV_32F)
//            highPrioriyParallelFor(Range(0, src1.rows), ParallelApplyScalarWeight<float>(&w1, src1, src1, &w2, src2, src2, &w3, src3, src3));
//        else
//            highPrioriyParallelFor(Range(0, src1.rows), ParallelApplyScalarWeight<double>(&w1, src1, src1, &w2, src2, src2, &w3, src3, src3));

//        // once the weights have been applied to the input then the next step is  the operation.
//        have_weights = false;
//    }

//    if(!have_weights)
//    {

//        if(!have_mask)
//        {
//            if(wdepth == CV_32F)
//                ParallelApplyFMX<float>(fun, src1, src2, src3, dst);
//            else
//                ParallelApplyFMX<double>(fun, src1, src2, src3, dst);
//        }
//        else
//        {
//            Mat mask = _mask.getMat() > 0;

//            double mx(0.);

//            minMaxLoc(mask, nullptr, &mx);

//            if(mx==255.)
//            {
//                highPrioriyParallelFor(Range(0, rows), ParallelProcessMask_BitwiseAnd(mask));
//            }
//            else
//            {
//                if(mx!=1.)
//                {
//                    if((static_cast<int>(mx)%8)==0)
//                        highPrioriyParallelFor(Range(0, rows), ParallelProcessMask_DivideByMaxBitShit(mask, mx));
//                    else
//                        highPrioriyParallelFor(Range(0, rows), ParallelProcessMask_DivideByMax(mask, mx));
//                }
//                else
//                    mask.convertTo(mask, wdepth);
//            }

//            mask.convertTo(mask, wdepth);

//            if(cn>1)
//            {
//                std::vector<Mat> tmp(3, mask);
//                merge(tmp, mask);

//                src1 = Mat(rows, cols * cn, sdepth, src1.data);
//                src2 = Mat(rows, cols * cn, sdepth, src2.data);
//                src3 = Mat(rows, cols * cn, sdepth, src3.data);
//                dst  = Mat(rows, cols * cn, wdepth, dst.data);
//                mask = Mat(rows, cols * cn, wdepth, mask.data);
//            }

//            src1.convertTo(src1, wdepth);
//            src2.convertTo(src2, wdepth);
//            src3.convertTo(src3, wdepth);

//            if(wdepth==CV_32F)
//                highPrioriyParallelFor(Range(0, rows), ParallelApplyFMXWithMask<float>(fun, src1, src2, src3, mask, dst) );
//            else
//                highPrioriyParallelFor(Range(0, rows), ParallelApplyFMXWithMask<double>(fun, src1, src2, src3, mask, dst) );

//            if(cn>1)
//                dst = Mat(rows, cols, wtype, dst.data);

//            _dst.assign(dst);
//        }
//    }
//    else
//    {

//        int cnt_scalar = static_cast<int>(_w1.isMatx()) + static_cast<int>(_w2.isMatx()) + static_cast<int>(_w3.isMatx());

//        if(cnt_scalar == 3)
//        {
//            Scalar w1 = getScalar(_w1);
//            Scalar w2 = getScalar(_w2);
//            Scalar w3 = getScalar(_w3);

//            switch (cn)
//            {
//            case 1:
//                for(int i=1;i<4;i++)
//                {
//                    w1(i) = w1(0);
//                    w2(i) = w2(0);
//                    w3(i) = w3(0);
//                }
//                break;

//            case 2:
//                for(int i=2;i<4;i++)
//                {
//                    w1(i) = w1(i-2);
//                    w2(i) = w2(i-2);
//                    w3(i) = w3(i-2);
//                }
//                break;
//            case 3:
//                w1(3) = w1(0);
//                w2(3) = w2(0);
//                w3(3) = w3(0);
//                break;
//            default:
//                break;
//            }

//            src1.convertTo(src1, wdepth);
//            src2.convertTo(src2, wdepth);
//            src3.convertTo(src3, wdepth);

//            if(wdepth==CV_32F)
//                highPrioriyParallelFor(Range(0, src1.rows), ParallelApplyScalarWeight<float>(&w1, src1, src1, &w2, src2, src2, &w3, src3, src3));
//            else
//                highPrioriyParallelFor(Range(0, src1.rows), ParallelApplyScalarWeight<double>(&w1, src1, src1, &w2, src2, src2, &w3, src3, src3));

////            hal::mu
//        }else
//        {
//            if(cnt_scalar == 0)
//            {

//                if(!have_mask)
//                {

//                }
//                else
//                {

//                }

//            }
//        }

//    }
//}

} // anonymous




} //cv
#endif
