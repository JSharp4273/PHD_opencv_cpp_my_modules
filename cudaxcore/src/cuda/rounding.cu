#include "../precomp.hpp"

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/cudev.hpp"

#include <type_traits>

using namespace cv::cudev;

namespace cv
{

namespace cuda
{

namespace device
{

namespace
{

template<class SrcTypeRef, class SrcFunType = typename VecTraits<SrcTypeRef>::elem_type>
struct check_source_type : std::false_type
{};

template<class SrcTypeRef>
struct check_source_type<SrcTypeRef, float> : std::true_type
{};

template<class SrcTypeRef>
struct check_source_type<SrcTypeRef, double> : std::true_type
{};

template<class DstTypeRef, class DstFunType = typename VecTraits<DstTypeRef>::elem_type>
struct check_destination_type : std::false_type
{};

template<class DstTypeRef>
struct check_destination_type<DstTypeRef, int> : std::true_type
{};

template<class DstTypeRef>
struct check_destination_type<DstTypeRef, float> : std::true_type
{};

template<class DstTypeRef>
struct check_destination_type<DstTypeRef, double> : std::true_type
{};

#define IMPL_FUN_NAT_SPEC_VEC_TYPE(name, type) \
    __device__ __forceinline__ type ## 2 name (const type ## 2 & v){ return make_ ## type ## 2(::name(v.x), ::name(v.y));} \
    __device__ __forceinline__ type ## 3 name (const type ## 3 & v){ return make_ ## type ## 3(::name(v.x), ::name(v.y), ::name(v.z));} \
    __device__ __forceinline__ type ## 4 name (const type ## 4 & v){ return make_ ## type ## 4(::name(v.x), ::name(v.y), ::name(v.z), ::name(v.w));}

IMPL_FUN_NAT_SPEC_VEC_TYPE(ceil, float)
IMPL_FUN_NAT_SPEC_VEC_TYPE(ceil, double)
IMPL_FUN_NAT_SPEC_VEC_TYPE(floor, float)
IMPL_FUN_NAT_SPEC_VEC_TYPE(floor, double)
IMPL_FUN_NAT_SPEC_VEC_TYPE(round, float)
IMPL_FUN_NAT_SPEC_VEC_TYPE(round, double)

#undef IMPL_FUN_NAT_SPEC_VEC_TYPE

#define IMPL_FUN_FAST_MATH_VEC_TYPE(name, type) \
    __device__ __forceinline__ int cv ## name(const type& value)\
    {\
        int i = static_cast<int>(value); \
        return i + (i < value); \
    }\
    \
    __device__ __forceinline__ int2 cv ## name(const type ## 2& v)\
    {\
        return  make_int2(cv ## name(v.x), cv ## name(v.y));\
    }\
    \
    __device__ __forceinline__ int3 cv ## name(const type ## 3& v)\
    {\
        return  make_int3(cv ## name(v.x), cv ## name(v.y), cv ## name(v.z));\
    }\
    \
    __device__ __forceinline__ int4 cv ## name(const type ## 4& v)\
    {\
        return  make_int4(cv ## name(v.x), cv ## name(v.y), cv ## name(v.z), cv ## name(v.w));\
    }

IMPL_FUN_FAST_MATH_VEC_TYPE(Ceil, float)
IMPL_FUN_FAST_MATH_VEC_TYPE(Ceil, double)
IMPL_FUN_FAST_MATH_VEC_TYPE(Floor, float)
IMPL_FUN_FAST_MATH_VEC_TYPE(Floor, double)
IMPL_FUN_FAST_MATH_VEC_TYPE(Round, float)
IMPL_FUN_FAST_MATH_VEC_TYPE(Round, double)

#undef IMPL_FUN_FAST_MATH_VEC_TYPE

#define IMPL_OP(name, minus, maj)\
\
template<class SrcType, class DstType>\
struct op_ ## minus ## name : unary_function<SrcType, DstType>\
{\
    static_assert (static_cast<int>(VecTraits<SrcType>::cn) == static_cast<int>(VecTraits<DstType>::cn), "Number of channels mismatch between the source and destination type.");\
    static_assert (check_source_type<SrcType>::value , "The type specified as source type is no supported.");\
    static_assert (check_destination_type<SrcType>::value , "The type specified as destination type is no supported.");\
\
    __device__ __forceinline__ DstType operator()(const SrcType& src) const\
    {\
        return cv ## maj ## name(src);\
    }\
};\
\
template<class T>\
struct op_ ## minus ## name<T, T> : unary_function<T, T>\
{\
    __device__ __forceinline__ T operator()(const T& src)const\
    {\
        return minus ## name(src);\
    }\
};\
\
template<>\
struct op_ ## minus ## name<float, float> : unary_function<float, float>\
{\
    __device__ __forceinline__ float operator()(const float& src)const\
    {\
        return ::minus ## name(src);\
    }\
};\
\
template<>\
struct op_ ## minus ## name<double, double> : unary_function<double, double>\
{\
    __device__ __forceinline__ double operator()(const float& src)const\
    {\
        return ::minus ## name(src);\
    }\
};

IMPL_OP(eil, c, C)
IMPL_OP(loor, f, F)
IMPL_OP(ound, r, R)

#undef IMPL_OP

template<class SrcType, class DstType, template <class,class>class Op>
void Impl(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    Op<SrcType, DstType> op;

    gridTransformUnary(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);;
}


} // anonymous

#define IMPL_FUN(name, minus, maj)\
\
template<class SrcType, class DstType>\
void maj ## name ## Impl(const GpuMat& _src, GpuMat& _dst, Stream& _stream)\
{\
    typedef void(*function_type)(const GpuMat&, GpuMat&, Stream&);\
\
    const function_type funcs[4] = {Impl<SrcType,                                DstType,                                op_ ## minus ## name>,\
                                    Impl<typename TypeVec<SrcType, 2>::vec_type, typename TypeVec<DstType, 2>::vec_type, op_ ## minus ## name>,\
                                    Impl<typename TypeVec<SrcType, 3>::vec_type, typename TypeVec<DstType, 3>::vec_type, op_ ## minus ## name>,\
                                    Impl<typename TypeVec<SrcType, 4>::vec_type, typename TypeVec<DstType, 4>::vec_type, op_ ## minus ## name>,\
                                   };\
\
    function_type fun = funcs[_src.channels() - 1];\
\
    fun(_src, _dst, _stream);\
}

IMPL_FUN(eil, c, C)
IMPL_FUN(loor, f, F)
IMPL_FUN(ound, r, R)

#undef IMPL_FUN

} // device

} // cuda

} // cv

#define DECL_SPEC(name)\
    template void cv::cuda::device::name ## Impl<float, float>(const GpuMat& _src, GpuMat& _dst, Stream& _stream);\
    template void cv::cuda::device::name ## Impl<float, int>(const GpuMat& _src, GpuMat& _dst, Stream& _stream);\
    template void cv::cuda::device::name ## Impl<double, double>(const GpuMat& _src, GpuMat& _dst, Stream& _stream);\
    template void cv::cuda::device::name ## Impl<double, int>(const GpuMat& _src, GpuMat& _dst, Stream& _stream);

DECL_SPEC(Ceil)
DECL_SPEC(Floor)
DECL_SPEC(Round)

