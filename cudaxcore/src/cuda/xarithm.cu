#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/cudev.hpp"

#include "opencv2/cudaxcore.hpp"
#include "../precomp.hpp"

//#include <crt/device_double_functions.h>
//#include <crt/device_functions.h>

using namespace cv::cudev;


namespace cv
{

namespace cuda
{


namespace device
{

#define IMPL_FMX(type)\
\
__device__ __forceinline__  type  fma(const type& a, const type& b, const type& c)\
{\
    return fmaf(a, b, c);\
}\
\
__device__ __forceinline__  type ## 1 fma(const type ## 1& a, const type ## 1& b, const type ## 1& c)\
{\
    type ## 1 ret = make_ ## type ## 1(fmaf(a.x, b.x, c.x));\
\
    return ret;\
}\
\
__device__ __forceinline__  type ## 2 fma(const type ## 2& a, const type ## 2& b, const type ## 2& c)\
{\
    type ## 2 ret = make_ ## type ## 2(fmaf(a.x, b.x, c.x),\
                             fmaf(a.y, b.y, c.y));\
\
    return ret;\
}\
\
__device__ __forceinline__  type ## 3 fma(const type ## 3& a, const type ## 3& b, const type ## 3& c)\
{\
    type ## 3 ret = make_ ## type ## 3(fmaf(a.x, b.x, c.x),\
                             fmaf(a.y, b.y, c.y),\
                             fmaf(a.z, b.z, c.z));\
\
    return ret;\
}\
\
__device__ __forceinline__  type ## 4 fma(const type ## 4& a, const type ## 4& b, const type ## 4& c)\
{\
    type ## 4 ret = make_ ## type ## 4(fmaf(a.x, b.x, c.x),\
                             fmaf(a.y, b.y, c.y),\
                             fmaf(a.z, b.z, c.z),\
                             fmaf(a.w, b.w, c.w));\
\
    return ret;\
} \
    \
    __device__ __forceinline__  type  fms(const type& a, const type& b, const type& c)\
    {\
        return fmaf(a, b, -c);\
    }\
    \
    __device__ __forceinline__  type ## 1 fms(const type ## 1& a, const type ## 1& b, const type ## 1& c)\
    {\
        type ## 1 ret = make_ ## type ## 1(fmaf(a.x, b.x, -c.x));\
    \
        return ret;\
    }\
    \
    __device__ __forceinline__  type ## 2 fms(const type ## 2& a, const type ## 2& b, const type ## 2& c)\
    {\
        type ## 2 ret = make_ ## type ## 2(fmaf(a.x, b.x, -c.x),\
                                 fmaf(a.y, b.y, -c.y));\
    \
        return ret;\
    }\
    \
    __device__ __forceinline__  type ## 3 fms(const type ## 3& a, const type ## 3& b, const type ## 3& c)\
    {\
        type ## 3 ret = make_ ## type ## 3(fmaf(a.x, b.x, -c.x),\
                                 fmaf(a.y, b.y, -c.y),\
                                 fmaf(a.z, b.z, -c.z));\
    \
        return ret;\
    }\
    \
    __device__ __forceinline__  type ## 4 fms(const type ## 4& a, const type ## 4& b, const type ## 4& c)\
    {\
        type ## 4 ret = make_ ## type ## 4(fmaf(a.x, b.x, -c.x),\
                                 fmaf(a.y, b.y, -c.y),\
                                 fmaf(a.z, b.z, -c.z),\
                                 fmaf(a.w, b.w, -c.w));\
    \
        return ret;\
    }

IMPL_FMX(float)
IMPL_FMX(double)

#undef IMPL_FMX



template<class T, int cn = VecTraits<T>::cn, bool isVector = isVectorType<T>::value>
struct OpClip : unary_function<T, T>
{

    typedef typename TypeVec<T, cn>::type value_type;
    typedef typename TypeVec<uchar, cn>::type compare_type;

    value_type l;
    value_type u;

    compare_type ones;

    __device__ __forceinline__ T operator()(const T& v) const;
};

template<class T>
struct OpClip<T, 1, false> : unary_function<T, T>
{
    typedef T value_type;
    typedef uchar compare_type;

    value_type l;
    value_type u;

    compare_type ones;

    __device__ __forceinline__ T operator()(const T& v) const { return v<l ? l : l>u ? u : v; }
};

template<class T>
struct OpClip<T, 1, true> : unary_function<T, T>
{
    typedef typename TypeVec<T, 1>::vec_type value_type;
    typedef uchar1 compare_type;

    value_type l;
    value_type u;

    compare_type ones;

    __device__ __forceinline__ T operator()(const T& v, const T& l, const T& u) const { return v.x<l.x ? l : l.x>u.x ? u : v; }
};

template<class T, int cn>
struct OpClip<T, cn, true> : unary_function<T, T>
{

    typedef typename TypeVec<T, cn>::vec_type value_type;
    typedef typename TypeVec<uchar, cn>::vec_type compare_type;

    value_type l;
    value_type u;

    compare_type ones;


    __device__ __forceinline__ value_type operator()(const value_type& v) const
    {
        value_type ret;

        // check if the element values are higher than the lower boundary.
        compare_type lb = (v < l) & ones;
        // check if the element values are higher than the upper boundary.
        compare_type ub = (v > u) & ones;
        // compute the mask of the elements that are within the boundaries.
        compare_type wb = ((~lb) ^ (~ub)) & ones;

        // convert
        value_type wlb = saturate_cast<value_type>(lb) * l;
        value_type wub = saturate_cast<value_type>(ub) * u;
        value_type wwb = saturate_cast<value_type>(wb) * v;

        //
        ret = wlb + wub + wwb;

        return ret;
    }
};

#define IMPL_SPEC_UCHAR_OPCLIP(cn)\
    template<>\
    struct OpClip<uchar ## cn, cn, true> : unary_function<uchar ## cn, uchar ## cn>\
    {\
        typedef uchar ## cn compare_type;\
        typedef compare_type value_type;\
    \
        value_type l;\
        value_type u;\
    \
        compare_type ones;\
    \
    \
        __device__ __forceinline__ value_type operator()(const value_type& v) const\
        {\
            compare_type ret;\
    \
            compare_type lb = (v < l) & l;\
            compare_type ub = (v > u) & u;\
            compare_type wb = ((~lb) ^ (~ub)) & v;\
    \
            ret = (lb ^ ub) ^ wb;\
    \
            return ret;\
        }\
    };

template<>
struct OpClip<uchar, 1, true> : unary_function<uchar, uchar>
{
    typedef uchar compare_type;
    typedef compare_type value_type;

    value_type l;
    value_type u;

    compare_type ones;


    __device__ __forceinline__ value_type operator()(const value_type& v) const
    {
        compare_type ret;

        // check if the element values are higher than the lower boundary.
        compare_type lb = (v < l) & l;
        // check if the element values are higher than the upper boundary.
        compare_type ub = (v > u) & u;
        // compute the mask of the elements that are within the boundaries.
        compare_type wb = ((~lb) ^ (~ub)) & v;

        //  create the output
        ret = (lb ^ ub) ^ wb;

        return ret;
    }
};


IMPL_SPEC_UCHAR_OPCLIP(2)
IMPL_SPEC_UCHAR_OPCLIP(3)
IMPL_SPEC_UCHAR_OPCLIP(4)

#undef IMPL_SPEC_UCHAR_OPCLIP




#define IMPL_SPEC_OPCLIP(vtype, type)\
template<>\
struct OpClip<vtype ## 2, 2, true> : unary_function<vtype ## 2, vtype ## 2>\
{\
    typedef uchar ## 2 compare_type;\
    typedef vtype ## 2 value_type;\
\
    value_type l;\
    value_type u;\
\
    compare_type ones;\
\
\
    __device__ __forceinline__ value_type operator()(const value_type& v) const\
    {\
        value_type ret;\
\
        \
        compare_type lb = (v < l) & ones;\
        \
        compare_type ub = (v > u) & ones;\
        \
        compare_type wb = ((~lb) ^ (~ub)) & ones;\
\
\
        value_type wlb = make_ ## vtype ## 2(saturate_cast<type>(lb.x), saturate_cast<type>(lb.y));\
        value_type wub = make_ ## vtype ## 2(saturate_cast<type>(ub.x), saturate_cast<type>(ub.y));\
        value_type wwb = make_ ## vtype ## 2(saturate_cast<type>(wb.x), saturate_cast<type>(wb.y));\
\
        ret.x = wlb.x + wub.x + wwb.x;\
        ret.y = wlb.y + wub.y + wwb.y;\
\
        return ret;\
    }\
};\
\
template<>\
struct OpClip<vtype ## 3, 3, true> : unary_function<vtype ## 3, vtype ## 3>\
{\
    typedef uchar3 compare_type;\
    typedef vtype ## 3 value_type;\
\
    value_type l;\
    value_type u;\
\
    compare_type ones;\
\
\
    __device__ __forceinline__ value_type operator()(const value_type& v) const\
{\
        value_type ret;\
\
        \
        compare_type lb = (v < l) & ones;\
        \
        compare_type ub = (v > u) & ones;\
        \
        compare_type wb = ((~lb) ^ (~ub)) & ones;\
\
\
        value_type wlb = make_ ## vtype ## 3(saturate_cast<type>(lb.x), saturate_cast<type>(lb.y), saturate_cast<type>(lb.z));\
        value_type wub = make_ ## vtype ## 3(saturate_cast<type>(ub.x), saturate_cast<type>(ub.y), saturate_cast<type>(ub.z));\
        value_type wwb = make_ ## vtype ## 3(saturate_cast<type>(wb.x), saturate_cast<type>(wb.y), saturate_cast<type>(wb.z));\
\
        ret.x = wlb.x + wub.x + wwb.x;\
        ret.y = wlb.y + wub.y + wwb.y;\
        ret.z = wlb.z + wub.z + wwb.z;\
\
        return ret;\
    }\
};\
\
template<>\
struct OpClip<vtype ## 4, 4, true> : unary_function<vtype ## 4, vtype ## 4>\
{\
    typedef uchar4 compare_type;\
    typedef vtype ## 4 value_type;\
\
    value_type l;\
    value_type u;\
\
    compare_type ones;\
\
\
    __device__ __forceinline__ value_type operator()(const value_type& v) const\
    {\
        value_type ret;\
\
        \
        compare_type lb = (v < l) & ones;\
        \
        compare_type ub = (v > u) & ones;\
        \
        compare_type wb = ((~lb) ^ (~ub)) & ones;\
\
\
        value_type wlb = make_ ## vtype ## 4(saturate_cast<type>(lb.x), saturate_cast<type>(lb.y), saturate_cast<type>(lb.z), saturate_cast<type>(lb.w));\
        value_type wub = make_ ## vtype ## 4(saturate_cast<type>(ub.x), saturate_cast<type>(ub.y), saturate_cast<type>(ub.z), saturate_cast<type>(lb.w));\
        value_type wwb = make_ ## vtype ## 4(saturate_cast<type>(wb.x), saturate_cast<type>(wb.y), saturate_cast<type>(wb.z), saturate_cast<type>(lb.w));\
\
        ret.x = wlb.x + wub.x + wwb.x;\
        ret.y = wlb.y + wub.y + wwb.y;\
        ret.z = wlb.z + wub.z + wwb.z;\
        ret.w = wlb.w + wub.w + wwb.w;\
\
        return ret;\
    }\
};

IMPL_SPEC_OPCLIP(char, schar)
IMPL_SPEC_OPCLIP(ushort, ushort)
IMPL_SPEC_OPCLIP(short, short)

template<int cn>
struct OpClip<float, cn, true> : unary_function<float, float>
{
    typedef typename TypeVec<float, cn>::type value_type;
    typedef typename TypeVec<uchar, cn>::vec_type compare_type;

    value_type l;
    value_type u;

    compare_type ones;


    __device__ __forceinline__ value_type operator()(const value_type& v, const value_type& l, const value_type& u) const
    {
        value_type ret;

        // check if the element values are higher than the lower boundary.
        compare_type lb = (v < l) & ones;
        // check if the element values are higher than the upper boundary.
        compare_type ub = (v > u) & ones;
        // compute the mask of the elements that are within the boundaries.
        compare_type wb = ((~lb) ^ (~ub)) & ones;

        // convert
        value_type wlb = saturate_cast<value_type>(lb) * l;
        value_type wub = saturate_cast<value_type>(ub);
        value_type wwb = saturate_cast<value_type>(wb);

        //
        ret = fma(wwb, v, fma(wub, u, wlb) );

        return ret;
    }
};


template<int cn>
struct OpClip<double, cn, true> : unary_function<double, double>
{
    typedef typename TypeVec<double, cn>::type value_type;
    typedef typename TypeVec<uchar, cn>::vec_type compare_type;

    value_type l;
    value_type u;

    compare_type ones;

    __device__ __forceinline__ value_type operator()(const value_type& v, const value_type& l, const value_type& u) const
    {
        value_type ret;

        // check if the element values are higher than the lower boundary.
        compare_type lb = (v < l) & ones;
        // check if the element values are higher than the upper boundary.
        compare_type ub = (v > u) & ones;
        // compute the mask of the elements that are within the boundaries.
        compare_type wb = ((~lb) ^ (~ub)) & ones;

        // convert
        value_type wlb = saturate_cast<value_type>(lb) * l;
        value_type whb = saturate_cast<value_type>(ub);
        value_type wwb = saturate_cast<value_type>(wb);

        //
        ret = fma(wwb, v, fma(whb, u, wlb) );

        return ret;
    }
};



template <typename Type> struct TransformPolicy : DefaultTransformPolicy {};

template <> struct TransformPolicy<double> : DefaultTransformPolicy
    {
        enum {
            shift = 1
        };
    };

template<class T>
void clipImpl(const GpuMat& _src, const Scalar& _l, const Scalar& _u, GpuMat& _dst, Stream& stream)
{

    typedef typename VecTraits<T>::elem_type channel_type;

    Scalar_<channel_type> l = _l;
    Scalar_<channel_type> u = _u;

    bool use_buffer(false);
    GpuMat buf;

    OpClip<T> op;

    op.l = VecTraits<T>::make(l.val);
    op.u = VecTraits<T>::make(u.val);
    op.ones = VecTraits<typename TypeVec<uchar, VecTraits<T>::cn>::vec_type>::all(1);

    if(_dst.empty() || (_dst.size() != _src.size()) || (_dst.type() != _src.type()) )
    {
        _dst = GpuMat(_src.size(), _src.type(), Scalar::all(0.));

        buf = _dst;
    }
    else
    {
        if(_dst.data == _src.data)
        {
            buf = GpuMat(_src.size(), _src.type(), Scalar::all(0.));
            use_buffer = true;
        }
        else
            buf = _dst;
    }

    gridTransformUnary_<TransformPolicy<T> >(globPtr<T>(_src), globPtr<T>(buf), op, stream);

    if(use_buffer)
        buf.copyTo(_dst);
}

} // device



} // cuda

} // cv


#define DECL_CLIP_IMPL_SPECS(type)\
template void cv::cuda::device::clipImpl<type>(const GpuMat&, const Scalar&, const Scalar&, GpuMat&, Stream&); \
template void cv::cuda::device::clipImpl<type ## 2>(const GpuMat&, const Scalar&, const Scalar&, GpuMat&, Stream&); \
template void cv::cuda::device::clipImpl<type ## 3>(const GpuMat&, const Scalar&, const Scalar&, GpuMat&, Stream&); \
template void cv::cuda::device::clipImpl<type ## 4>(const GpuMat&, const Scalar&, const Scalar&, GpuMat&, Stream&);

DECL_CLIP_IMPL_SPECS(uchar)
DECL_CLIP_IMPL_SPECS(ushort)
DECL_CLIP_IMPL_SPECS(short)
DECL_CLIP_IMPL_SPECS(int)
DECL_CLIP_IMPL_SPECS(float)
DECL_CLIP_IMPL_SPECS(double)

#undef DECL_CLIP_IMPL_SPECS


template void cv::cuda::device::clipImpl<schar>(const GpuMat& , const Scalar& , const Scalar& , GpuMat& _dst, Stream& );
template void cv::cuda::device::clipImpl<char2>(const GpuMat& , const Scalar& , const Scalar& , GpuMat& _dst, Stream& );
template void cv::cuda::device::clipImpl<char3>(const GpuMat& , const Scalar& , const Scalar& , GpuMat& _dst, Stream& );
template void cv::cuda::device::clipImpl<char4>(const GpuMat& , const Scalar& , const Scalar& , GpuMat& _dst, Stream& );



namespace cv
{

namespace cuda
{

namespace device
{

template<class T, class Tvec = typename TypeVec<typename VecTraits<T>::elem_type, 4>::vec_type, int inc = static_cast<int>(VecTraits<Tvec>::cn)/static_cast<int>(VecTraits<T>::cn)>
struct OpKronKernel
{
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;

    typedef Tvec vector_type;
    typedef Tvec* vector_pointer;
    typedef const Tvec* const_vector_pointer;

    Tvec v4;
    T v;

    __forceinline__ __device__ OpKronKernel(const value_type& _v):v(_v){}

    __device__ int compute(const_pointer src, const int& current, const int& cols, const value_type& v, pointer dst)const
    {
        int c=current;

        for(;c<cols; c++, src++, dst++)
            *dst = saturate_cast<value_type>(v * *src);
        return c;
    }

    __device__ int compute_vec(const_pointer src, const int& current, const int& cols, const vector_type& v, pointer dst)const
    {

        const_vector_pointer it_src = reinterpret_cast<const_vector_pointer>(src);
        vector_pointer it_dst = reinterpret_cast<vector_pointer>(dst);

        int c=current;

        int vec_width = cols - cols%inc;

        for(;c<vec_width; c+=inc, it_src++, it_dst++)
            *it_dst = saturate_cast<vector_type>(v * *it_src);

        return c;
    }

    __device__ void operator()(const_pointer src, const int& current, const int& cols, pointer dst)const
    {
//        printf("A\n");

        int c = compute_vec(src, 0, cols, v4, dst);

        if(c == cols)
            return;
        compute(src + c, c, cols, v, dst + c);
    }
};


template<class T, class Tvec>
struct OpKronKernel<T, Tvec, 2>
{
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;

    typedef Tvec vector_type;
    typedef Tvec* vector_pointer;
    typedef const Tvec* const_vector_pointer;

    Tvec v4;
    T v;

    __forceinline__ __device__ OpKronKernel(const value_type& _v):v(_v){v4 = VecTraits<Tvec>::make(v.x, v.y, v.x, v.y);}

    __device__ int compute(const_pointer src, const int& current, const int& cols, const value_type& v, pointer dst)const
    {
        int c=current;

        for(;c<cols; c++, src++, dst++)
            *dst = saturate_cast<value_type>(v * *src);
        return c;
    }

    __device__ int compute_vec(const_pointer src, const int& current, const int& cols, const vector_type& v, pointer dst)const
    {

        const_vector_pointer it_src = reinterpret_cast<const_vector_pointer>(src);
        vector_pointer it_dst = reinterpret_cast<vector_pointer>(dst);

        int c=current;

        int vec_width = cols - cols%2;

        for(;c<vec_width; c+=2, it_src++, it_dst++)
            *it_dst = saturate_cast<vector_type>(v * *it_src);

        return c;
    }

    __device__ void operator()(const_pointer src, const int& current, const int& cols, pointer dst)const
    {
//        printf("B\n");
        int c = compute_vec(src, 0, cols, v4, dst);

        compute(src + c, c, cols, v, dst + c);
    }
};

template<class T, class Tvec>
struct OpKronKernel<T, Tvec, 4>
{
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;

    typedef Tvec vector_type;
    typedef Tvec* vector_pointer;
    typedef const Tvec* const_vector_pointer;

    Tvec v4;
    T v;

    __forceinline__ __device__ OpKronKernel(const value_type& _v):v(_v){v4 = VecTraits<Tvec>::all(v); }

    __device__ int compute(const_pointer src, const int& current, const int& cols, const value_type& v, pointer dst)const
    {
        int c=current;

        for(;c<cols; c++, src++, dst++)
            *dst = saturate_cast<value_type>(v * *src);
        return c;
    }

    __device__ int compute_vec(const_pointer src, const int& current, const int& cols, const vector_type& v, pointer dst)const
    {

        const_vector_pointer it_src = reinterpret_cast<const_vector_pointer>(src);
        vector_pointer it_dst = reinterpret_cast<vector_pointer>(dst);

        int c=current;

        int vec_width = cols - cols%4;

        for(;c<vec_width; c+=4, it_src++, it_dst++)
            *it_dst = saturate_cast<vector_type>(v * *it_src);

        return c;
    }

    __device__ void operator()(const_pointer src, const int& current, const int& cols, pointer dst)const
    {
        int c = compute_vec(src, 0, cols, v4, dst);

        compute(src + c, c, cols, v, dst + c);
    }
};


template<class T>
__global__ void kkron(const PtrStepSz<T> src1, const PtrStepSz<T> src2, PtrStep<T> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src1.cols || y >= src1.rows)
        return;

    int yy = y * src2.rows;
    int xx = x * src2.cols;

    T v = src1(y, x);

    OpKronKernel<T> op(v);

    for(int r=0, rr=yy; r<src2.rows; r++, rr++)
        op(src2.ptr(r), 0, src2.cols, dst.ptr(rr) + xx);
}

template<class T>
void kronImpl(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, Stream& stream)
{
    dim3 block(32,8);
    dim3 grid (divUp (src1.cols, block.x), divUp (src1.rows, block.y));

    cudaSafeCall( cudaFuncSetCacheConfig (kkron<T>, cudaFuncCachePreferL1) );
    kkron<T><<<grid, block, 0, StreamAccessor::getStream(stream)>>>(src1, src2, dst);
    cudaSafeCall ( cudaGetLastError () );

    if (stream == 0)
         cudaSafeCall( cudaDeviceSynchronize() );
}

} // device

} // cuda

} //cv

#define DECL_KRON_IMPL_SPECS(type)\
template void cv::cuda::device::kronImpl<type>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::kronImpl<type ## 2>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::kronImpl<type ## 3>(const GpuMat&, const GpuMat&, GpuMat&, Stream&); \
template void cv::cuda::device::kronImpl<type ## 4>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

DECL_KRON_IMPL_SPECS(uchar)
DECL_KRON_IMPL_SPECS(ushort)
DECL_KRON_IMPL_SPECS(short)
DECL_KRON_IMPL_SPECS(int)
DECL_KRON_IMPL_SPECS(float)
DECL_KRON_IMPL_SPECS(double)

#undef DECL_KRON_IMPL_SPECS


template void cv::cuda::device::kronImpl<schar>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);
template void cv::cuda::device::kronImpl<char2>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);
template void cv::cuda::device::kronImpl<char3>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);
template void cv::cuda::device::kronImpl<char4>(const GpuMat&, const GpuMat&, GpuMat&, Stream&);


#if 0

namespace cv
{

namespace cuda
{

namespace device
{

namespace
{

#define IMPL_RCP_FUN(suffix)\
__device__ __forceinline__ float rcp_ ## suffix(const float& v)\
{\
    return ::__frcp_ ## suffix(v);\
}\
\
__device__ __forceinline__ float1 rcp_ ## suffix(const float1& v)\
{\
    return make_float1(::__frcp_ ## suffix(v.x));\
}\
\
__device__ __forceinline__ float2 rcp_ ## suffix(const float2& v)\
{\
    return make_float2(::__frcp_ ## suffix(v.x), ::__frcp_ ## suffix(v.y));\
}\
\
__device__ __forceinline__ float3 rcp_ ## suffix(const float3& v)\
{\
    return make_float3(::__frcp_  ## suffix(v.x), ::__frcp_  ## suffix(v.y), ::__frcp_  ## suffix(v.z));\
}\
\
__device__ __forceinline__ float4 rcp_  ## suffix(const float4& v)\
{\
    return make_float4(::__frcp_  ## suffix(v.x), ::__frcp_  ## suffix(v.y), ::__frcp_  ## suffix(v.z), ::__frcp_  ## suffix(v.w));\
}\
\
__device__ __forceinline__ double rcp_  ## suffix(const double& v)\
{\
    return ::__drcp_  ## suffix(v);\
}\
\
__device__ __forceinline__ double1 rcp_  ## suffix(const double1& v)\
{\
    return make_double1(::__drcp_  ## suffix(v.x));\
}\
\
__device__ __forceinline__ double2 rcp_  ## suffix(const double2& v)\
{\
    return make_double2(::__drcp_  ## suffix(v.x), ::__drcp_  ## suffix(v.y));\
}\
\
__device__ __forceinline__ double3 rcp_  ## suffix(const double3& v)\
{\
    return make_double3(::__drcp_  ## suffix(v.x), ::__drcp_  ## suffix(v.y), ::__drcp_  ## suffix(v.z));\
}\
\
__device__ __forceinline__ double4 rcp_  ## suffix(const double4& v)\
{\
    return make_double4(::__drcp_  ## suffix(v.x), ::__drcp_  ## suffix(v.y), ::__drcp_  ## suffix(v.z), ::__drcp_  ## suffix(v.w));\
}

IMPL_RCP_FUN(rn)
IMPL_RCP_FUN(rd)
IMPL_RCP_FUN(ru)
IMPL_RCP_FUN(rz)

#undef IMPL_RCP_FUN

#define IMPL_RCP_OP(name) \
template<class SrcType, class WrkType, class DstType>\
struct recip_ ## name ## _op : unary_function<SrcType, DstType>\
{\
    typedef SrcType source_type;\
    typedef DstType destination_type;\
\
__device__ __forceinline__ destination_type operator()(const source_type& v) const{ return saturate_cast<destination_type>(rcp_ ## name ( saturate_cast<WrkType>(v) ) );}\
};


IMPL_RCP_OP(rn)
IMPL_RCP_OP(rz)
IMPL_RCP_OP(ru)
IMPL_RCP_OP(rd)

#undef IMPL_RCP_OP

template<class SrcType, class DstType, template<class, class, class>class Op>
void recip(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

    typedef typename TypeVec<typename std::conditional<std::is_integral<typename VecTraits<SrcType>::elem_type>::value, float, double>::type, VecTraits<SrcType>::cn>::vec_type working_type;

//    static_assert ((static_cast<int>(VecTraits<SrcType>::cn) == static_cast<int>(VecTraits<DstType>::cn)) && (static_cast<int>(VecTraits<SrcType>::cn) == static_cast<int>(VecTraits<WrkType>::cn)), "Number of Cannels Must be equal");

    Op<SrcType, working_type, DstType> op;

    if(!_mask.data)
        gridTransformUnary(globPtr<SrcType>(_src), globPtr<DstType>(_dst), op, _stream);
    else
        gridTransformUnary(globPtr<SrcType>(_src), globPtr<DstType>(_dst), op, globPtr<uchar>(_mask), _stream);
}

} // anonymous

#define IMPL_CALL_FUN(name)\
template<class SrcType, class DstType>\
void recip_ ## name(const GpuMat& _src, GpuMat& _dst, Stream& _stream)\
{\
    recip<SrcType, DstType, recip_ ## name ## _op>(_src, _dst, GpuMat(), _stream);\
}\
\
template<class SrcType, class DstType> \
void masked_recip_ ## name(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)\
{\
    recip<SrcType, DstType, recip_ ## name ## _op>(_src, _dst, _mask, _stream);\
}

IMPL_CALL_FUN(rn)
IMPL_CALL_FUN(rz)
IMPL_CALL_FUN(ru)
IMPL_CALL_FUN(rd)



} // device

} // cuda

} // cv

#define SPEC_RCP(name, type)\
\
template void cv::cuda::device::recip_ ## name<type , uchar >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, uchar2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, uchar3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, uchar4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<type , schar>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, char2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, char3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, char4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<type , ushort >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, ushort2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, ushort3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, ushort4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<type , short >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, short2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, short3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, short4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<type , int >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, int2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, int3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, int4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<type , type >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, float2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, float3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, float4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<type , double >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, double2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, double3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, double4>(const GpuMat&, GpuMat&, Stream&);


#define SPEC_RCP_SCHAR(name, type)\
\
template void cv::cuda::device::recip_ ## name<s ## type , uchar >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, uchar2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, uchar3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, uchar4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<s ## type , schar>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, char2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, char3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, char4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<s ## type , ushort >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, ushort2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, ushort3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, ushort4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<s ## type , short >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, short2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, short3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, short4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<s ## type , int >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, int2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, int3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, int4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<s ## type , type >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, float2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, float3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, float4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_ ## name<s ## type , double >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 2, double2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 3, double3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_ ## name<type ## 4, double4>(const GpuMat&, GpuMat&, Stream&);

#define SPEC_MASKED_RCP(name, type)\
    \
    template void cv::cuda::device::masked_recip_ ## name<type , uchar >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, uchar2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, uchar3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, uchar4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<type , schar>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, char2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, char3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, char4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<type , ushort >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, ushort2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, ushort3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, ushort4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<type , short> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, short2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, short3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, short4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<type , int> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, int2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, int3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, int4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<type , float> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, float2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, float3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, float4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<type , double>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, double2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, double3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, double4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);

#define SPEC_MASKED_RCP_SCHAR(name, type)\
    \
    template void cv::cuda::device::masked_recip_ ## name<s ## type , uchar >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, uchar2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, uchar3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, uchar4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<s ## type , schar>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, char2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, char3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, char4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<s ## type , ushort >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, ushort2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, ushort3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, ushort4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<s ## type , short> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, short2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, short3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, short4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<s ## type , int> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, int2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, int3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, int4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<s ## type , float> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, float2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, float3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, float4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_ ## name<s ## type , double>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 2, double2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 3, double3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_ ## name<type ## 4, double4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);


SPEC_RCP(rn, uchar)
SPEC_RCP_SCHAR(rn, char)
SPEC_RCP(rn, ushort)
SPEC_RCP(rn, short)
SPEC_RCP(rn, int)
SPEC_RCP(rn, float)
SPEC_RCP(rn, double)

SPEC_RCP(rz, uchar)
SPEC_RCP_SCHAR(rz, char)
SPEC_RCP(rz, ushort)
SPEC_RCP(rz, short)
SPEC_RCP(rz, int)
SPEC_RCP(rz, float)
SPEC_RCP(rz, double)

SPEC_RCP(ru, uchar)
SPEC_RCP_SCHAR(ru, char)
SPEC_RCP(ru, ushort)
SPEC_RCP(ru, short)
SPEC_RCP(ru, int)
SPEC_RCP(ru, float)
SPEC_RCP(ru, double)

SPEC_RCP(rd, uchar)
SPEC_RCP_SCHAR(rd, char)
SPEC_RCP(rd, ushort)
SPEC_RCP(rd, short)
SPEC_RCP(rd, int)
SPEC_RCP(rd, float)
SPEC_RCP(rd, double)




SPEC_MASKED_RCP(rn, uchar)
SPEC_MASKED_RCP_SCHAR(rn, char)
SPEC_MASKED_RCP(rn, ushort)
SPEC_MASKED_RCP(rn, short)
SPEC_MASKED_RCP(rn, int)
SPEC_MASKED_RCP(rn, float)
SPEC_MASKED_RCP(rn, double)


SPEC_MASKED_RCP(rz, uchar)
SPEC_MASKED_RCP_SCHAR(rz, char)
SPEC_MASKED_RCP(rz, ushort)
SPEC_MASKED_RCP(rz, short)
SPEC_MASKED_RCP(rz, int)
SPEC_MASKED_RCP(rz, float)
SPEC_MASKED_RCP(rz, double)


SPEC_MASKED_RCP(ru, uchar)
SPEC_MASKED_RCP_SCHAR(ru, char)
SPEC_MASKED_RCP(ru, ushort)
SPEC_MASKED_RCP(ru, short)
SPEC_MASKED_RCP(ru, int)
SPEC_MASKED_RCP(ru, float)
SPEC_MASKED_RCP(ru, double)


SPEC_MASKED_RCP(rd, uchar)
SPEC_MASKED_RCP_SCHAR(rd, char)
SPEC_MASKED_RCP(rd, ushort)
SPEC_MASKED_RCP(rd, short)
SPEC_MASKED_RCP(rd, int)
SPEC_MASKED_RCP(rd, float)
SPEC_MASKED_RCP(rd, double)


#undef SPEC_RCP
#undef SPEC_RCP_SCHAR
#undef SPEC_MASKED_RCP
#undef SPEC_MASKED_RCP_SCHAR


namespace cv
{

namespace cuda
{

namespace device
{

namespace
{

__device__ __forceinline__ float recip_srt(const float& v){return ::rsqrtf(v); }
__device__ __forceinline__ float2 recip_srt(const float2& v){return make_float2(::rsqrtf(v.x), ::rsqrtf(v.y) ); }
__device__ __forceinline__ float3 recip_srt(const float3& v){return make_float3(::rsqrtf(v.x), ::rsqrtf(v.y), ::rsqrtf(v.z) ); }
__device__ __forceinline__ float4 recip_srt(const float4& v){return make_float4(::rsqrtf(v.x), ::rsqrtf(v.y), ::rsqrtf(v.z), ::rsqrtf(v.w) ); }

__device__ __forceinline__ double recip_srt(const double& v){return ::rsqrt(v); }
__device__ __forceinline__ double2 recip_srt(const double2& v){return make_double2(::rsqrt(v.x), ::rsqrt(v.y) ); }
__device__ __forceinline__ double3 recip_srt(const double3& v){return make_double3(::rsqrt(v.x), ::rsqrt(v.y), ::rsqrt(v.z) ); }
__device__ __forceinline__ double4 recip_srt(const double4& v){return make_double4(::rsqrt(v.x), ::rsqrt(v.y), ::rsqrt(v.z), ::rsqrt(v.w) ); }

template<class SrcType, class WrkType, class DstType>
struct recip_sqrt_op : unary_function<SrcType, DstType>
{
    __device__ __forceinline__ DstType operator()(const SrcType& v) const{ return saturate_cast<DstType>(recip_sqrt(saturate_cast<WrkType>(v) ) );}
};

}

template<class SrcType, class DstType>
void recip_sqrt(const GpuMat& _src, GpuMat& _dst, Stream& _stream)
{
    recip<SrcType, DstType, recip_sqrt_op>(_src, _dst, GpuMat(), _stream);
}



template<class SrcType, class DstType>
void masked_recip_sqrt(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{
    recip<SrcType, DstType, recip_sqrt_op>(_src, _dst, _mask, _stream);
}


} // device

} // cuda

} // cv

#define SPEC_SQRT_RCP(type)\
\
template void cv::cuda::device::recip_sqrt<type , uchar >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, uchar2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, uchar3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, uchar4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<type , schar>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, char2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, char3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, char4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<type , ushort >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, ushort2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, ushort3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, ushort4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<type , short >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, short2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, short3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, short4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<type , int >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, int2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, int3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, int4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<type , type >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, float2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, float3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, float4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<type , double >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, double2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, double3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, double4>(const GpuMat&, GpuMat&, Stream&);


#define SPEC_SQRT_RCP_SCHAR(type)\
\
template void cv::cuda::device::recip_sqrt<s ## type , uchar >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, uchar2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, uchar3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, uchar4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<s ## type , schar>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, char2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, char3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, char4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<s ## type , ushort >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, ushort2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, ushort3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, ushort4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<s ## type , short >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, short2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, short3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, short4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<s ## type , int >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, int2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, int3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, int4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<s ## type , type >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, float2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, float3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, float4>(const GpuMat&, GpuMat&, Stream&);\
\
template void cv::cuda::device::recip_sqrt<s ## type , double >(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 2, double2>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 3, double3>(const GpuMat&, GpuMat&, Stream&);\
template void cv::cuda::device::recip_sqrt<type ## 4, double4>(const GpuMat&, GpuMat&, Stream&);

#define SPEC_MASKED_SQRT_RCP(type)\
    \
    template void cv::cuda::device::masked_recip_sqrt<type , uchar >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, uchar2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, uchar3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, uchar4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<type , schar>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, char2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, char3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, char4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<type , ushort >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, ushort2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, ushort3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, ushort4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<type , short> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, short2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, short3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, short4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<type , int> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, int2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, int3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, int4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<type , float> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, float2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, float3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, float4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<type , double>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, double2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, double3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, double4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);

#define SPEC_MASKED_SQRT_RCP_SCHAR(type)\
    \
    template void cv::cuda::device::masked_recip_sqrt<s ## type , uchar >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, uchar2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, uchar3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, uchar4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<s ## type , schar>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, char2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, char3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, char4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<s ## type , ushort >(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, ushort2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, ushort3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, ushort4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<s ## type , short> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, short2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, short3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, short4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<s ## type , int> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, int2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, int3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, int4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<s ## type , float> (const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, float2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, float3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, float4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    \
    template void cv::cuda::device::masked_recip_sqrt<s ## type , double>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 2, double2>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 3, double3>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);\
    template void cv::cuda::device::masked_recip_sqrt<type ## 4, double4>(const GpuMat&, GpuMat&, const GpuMat&, Stream&);


SPEC_SQRT_RCP(uchar)
SPEC_SQRT_RCP_SCHAR(char)
SPEC_SQRT_RCP(ushort)
SPEC_SQRT_RCP(short)
SPEC_SQRT_RCP(int)
SPEC_SQRT_RCP(float)
SPEC_SQRT_RCP(double)

SPEC_MASKED_SQRT_RCP(uchar)
SPEC_MASKED_SQRT_RCP_SCHAR(char)
SPEC_MASKED_SQRT_RCP(ushort)
SPEC_MASKED_SQRT_RCP(short)
SPEC_MASKED_SQRT_RCP(int)
SPEC_MASKED_SQRT_RCP(float)
SPEC_MASKED_SQRT_RCP(double)

#undef SPEC_SQRT_RCP
#undef SPEC_SQRT_RCP_SCHAR
#undef SPEC_MASKED_SQRT_RCP
#undef SPEC_MASKED_RCP_SCHAR

#endif // if 0
