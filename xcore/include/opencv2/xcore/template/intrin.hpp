#ifndef XCORE_INTRIN_HPP
#define XCORE_INTRIN_HPP

#include "opencv2/core/simd_intrinsics.hpp"


#if defined __FMA__
#undef CV_FMA3
#define CV_FMA3 1
#endif


namespace cv
{

//////////////////////////////////////////
/// More intrinsics traits + functions ///
//////////////////////////////////////////

template<typename _T> struct Type2Vec_Traits;
#define CV_INTRIN_DEF_TYPE2VEC_TRAITS(type_, vec_type_) \
    template<> struct Type2Vec_Traits<type_> \
    { \
        typedef vec_type_ vec_type; \
    }

CV_INTRIN_DEF_TYPE2VEC_TRAITS(uchar, v_uint8);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(schar, v_int8);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(ushort, v_uint16);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(short, v_int16);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(unsigned, v_uint32);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(int, v_int32);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(float, v_float32);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(uint64, v_uint64);
CV_INTRIN_DEF_TYPE2VEC_TRAITS(int64, v_int64);
#if CV_SIMD_64F
CV_INTRIN_DEF_TYPE2VEC_TRAITS(double, v_float64);
#endif

#if CV_VERSION_MAJOR < 4 || CV_VERSION_MINOR<3
template <class T>
typename Type2Vec_Traits<T>::vec_type vx_setall(const T& v);

#define IMPL_SETALL(type, fun)\
    template <> typename Type2Vec_Traits<type>::vec_type vx_setall<type>(const type& v){ return fun(v);}

IMPL_SETALL(uchar, vx_setall_u8)
IMPL_SETALL(schar, vx_setall_s8)
IMPL_SETALL(unsigned short, vx_setall_u16)
IMPL_SETALL(short, vx_setall_s16)
IMPL_SETALL(unsigned int, vx_setall_u32)
IMPL_SETALL(int, vx_setall_s32)
IMPL_SETALL(float, vx_setall_f32)
IMPL_SETALL(double, vx_setall_f64)

#undef IMPL_SETALL

#endif

template <class T>
inline typename Type2Vec_Traits<T>::vec_type vx_setzeros(){ return vx_setall<T>(static_cast<T>(0.));}


#if CV_SIMD
v_float32x4 v_fms(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c);
v_float32x4 v_nfma(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c);
v_float32x4 v_nfms(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c);
v_float32x4 v_fda(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c);
v_float32x4 v_fds(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c);
v_float32x4 v_nfda(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c);
v_float32x4 v_nfds(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c);
#endif

#if CV_SIMD128_64F
v_float64x2 v_fms(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c);
v_float64x2 v_nfma(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c);
v_float64x2 v_nfms(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c);
v_float64x2 v_fda(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c);
v_float64x2 v_fds(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c);
v_float64x2 v_nfda(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c);
v_float64x2 v_nfds(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c);
#endif

#if CV_SIMD256
v_float32x8 v_fms(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c);
v_float32x8 v_nfma(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c);
v_float32x8 v_nfms(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c);
v_float32x8 v_fda(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c);
v_float32x8 v_fds(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c);
v_float32x8 v_nfda(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c);
v_float32x8 v_nfds(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c);
#endif

#if CV_SIMD256_64F
v_float64x4 v_fms(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c);
v_float64x4 v_nfma(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c);
v_float64x4 v_nfms(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c);
v_float64x4 v_fda(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c);
v_float64x4 v_fds(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c);
v_float64x4 v_nfda(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c);
v_float64x4 v_nfds(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c);
#endif


}//cv

#endif // XCORE_INTRIN_HPP
