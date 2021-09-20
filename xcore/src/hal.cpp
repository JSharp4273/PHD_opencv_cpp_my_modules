#include "opencv2/xcore/template/hal.hpp"
#include "opencv2/xcore/template/intrin.hpp"
#include "opencv2/xcore/template/traits.hpp"
#include "opencv2/core/simd_intrinsics.hpp"
#include "opencv2/cvconfig.h"


namespace std
{

inline float fmsf(const float& a, const float& b, const float& c){ return std::fmaf(a, b, -c);}
inline float nfmaf(const float& a, const float& b, const float& c){ return std::fmaf(-a, b, c);}
inline float nfmsf(const float& a, const float& b, const float& c){ return std::fmaf(-a, b, -c);}
inline float fdaf(const float& a, const float& b, const float& c){ return (a/b)+c;}
inline float fdsf(const float& a, const float& b, const float& c){ return (a/b)-c;}
inline float nfdaf(const float& a, const float& b, const float& c){ return fdaf(-a,b,c);}
inline float nfdsf(const float& a, const float& b, const float& c){ return fdsf(-a,b,c);}


template<class T>
inline T fms(const T& a, const T& b, const T& c){ return std::fma(a, b, -c);}

template<class T>
inline T nfma(const T& a, const T& b, const T& c){ return std::fma(-a, b, c);}

template<class T>
inline T nfms(const T& a, const T& b, const T& c){ return std::fma(-a, b, -c);}

template<class T>
inline T fda(const T& a, const T& b, const T& c){ return (a/b)+c;}

template<class T>
inline T fds(const T& a, const T& b, const T& c){ return (a/b)-c;}

template<class T>
inline T nfda(const T& a, const T& b, const T& c){ return fdaf(-a,b,c);}

template<class T>
inline T nfds(const T& a, const T& b, const T& c){ return fdsf(-a,b,c);}


}//std


#define IMPL_ARITHM_SPEC_FUN_1_(fun, type, suffix)\
    template<> void fun<type>( const type* src1, size_t step1, const type* src2, size_t step2, type* dst, size_t step, int width, int height, void* ) \
{\
 fun##suffix(src1, step1, src2, step2, dst, step, width, height, nullptr);\
}

#define IMPL_ARITHM_SPEC_FUN_1(fun) \
    IMPL_ARITHM_SPEC_FUN_1_(fun, uchar, 8u)\
    IMPL_ARITHM_SPEC_FUN_1_(fun, schar, 8s)\
    IMPL_ARITHM_SPEC_FUN_1_(fun, ushort, 16u)\
    IMPL_ARITHM_SPEC_FUN_1_(fun, short, 16s)\
    IMPL_ARITHM_SPEC_FUN_1_(fun, int, 32s)\
    IMPL_ARITHM_SPEC_FUN_1_(fun, float, 32f)\
    IMPL_ARITHM_SPEC_FUN_1_(fun, double, 64f)

#define IMPL_ARITHM_SPEC_FUN_2_(fun, type, suffix)\
    template<> void fun<type>( const type* src1, size_t step1, const type* src2, size_t step2, type* dst, size_t step, int width, int height, void* smth) \
{\
 fun##suffix(src1, step1, src2, step2, dst, step, width, height, smth);\
}

#define IMPL_ARITHM_SPEC_FUN_2(fun) \
    IMPL_ARITHM_SPEC_FUN_2_(fun, uchar, 8u)\
    IMPL_ARITHM_SPEC_FUN_2_(fun, schar, 8s)\
    IMPL_ARITHM_SPEC_FUN_2_(fun, ushort, 16u)\
    IMPL_ARITHM_SPEC_FUN_2_(fun, short, 16s)\
    IMPL_ARITHM_SPEC_FUN_2_(fun, int, 32s)\
    IMPL_ARITHM_SPEC_FUN_2_(fun, float, 32f)\
    IMPL_ARITHM_SPEC_FUN_2_(fun, double, 64f)

#define IMPL_ARITHM_SPEC_FUN_CMP_(fun, type, suffix)\
    template<> void fun<type>( const type* src1, size_t step1, const type* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* smth) \
{\
 fun##suffix(src1, step1, src2, step2, dst, step, width, height, smth);\
}

#define IMPL_ARITHM_SPEC_FUN_CMP(fun) \
    IMPL_ARITHM_SPEC_FUN_CMP_(fun, uchar, 8u)\
    IMPL_ARITHM_SPEC_FUN_CMP_(fun, schar, 8s)\
    IMPL_ARITHM_SPEC_FUN_CMP_(fun, ushort, 16u)\
    IMPL_ARITHM_SPEC_FUN_CMP_(fun, short, 16s)\
    IMPL_ARITHM_SPEC_FUN_CMP_(fun, int, 32s)\
    IMPL_ARITHM_SPEC_FUN_CMP_(fun, float, 32f)\
    IMPL_ARITHM_SPEC_FUN_CMP_(fun, double, 64f)

#define IMPL_ROUNDING_(name, type1, type2, suffix)\
    template<> void name(const type1* src, size_t step1, type2* dst, size_t step2, int width, int height){ name ## suffix(src, step1, dst, step2, width, height);}

#define IMPL_ROUNDING\
    IMPL_ROUNDING_(ceil, float, float, 32f) \
    IMPL_ROUNDING_(ceil, float, int, 32f) \
    IMPL_ROUNDING_(ceil, double, double, 64f) \
    IMPL_ROUNDING_(ceil, double, int, 32f) \
    IMPL_ROUNDING_(floor, float, float, 32f) \
    IMPL_ROUNDING_(floor, float, int, 32f) \
    IMPL_ROUNDING_(floor, double, double, 64f) \
    IMPL_ROUNDING_(floor, double, int, 32f) \
    IMPL_ROUNDING_(round, float, float, 32f) \
    IMPL_ROUNDING_(round, float, int, 32f) \
    IMPL_ROUNDING_(round, double, double, 64f) \
    IMPL_ROUNDING_(round, double, int, 32f)

namespace cv
{

namespace hal
{

IMPL_ARITHM_SPEC_FUN_1(add)

IMPL_ARITHM_SPEC_FUN_1(sub)

IMPL_ARITHM_SPEC_FUN_1(max)

IMPL_ARITHM_SPEC_FUN_1(min)

IMPL_ARITHM_SPEC_FUN_1(absdiff)

IMPL_ARITHM_SPEC_FUN_CMP(cmp)

IMPL_ARITHM_SPEC_FUN_2(mul)

IMPL_ARITHM_SPEC_FUN_2(div)

IMPL_ARITHM_SPEC_FUN_2(recip)














void ceil32f(const float* src, size_t step1, float* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(float);
    const size_t dst_row_step = step2 / sizeof(float);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(float);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const float* it_src = src;
        float* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_float32 v_src = vx_load(it_src);
#ifdef CV_AVX
            v_src.val = _mm256_ceil_ps(v_src.val);
#elif defined(CV_SSE4_1)
            v_src.val = _mm_ceil_ps(v_src.val);
#else
            v_src = v_cvt_f32(v_ceil(v_src));
#endif
            v_store(it_dst, v_src);
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = saturate_cast<float>(cvCeil(*it_dst));
    }
}

void ceil32f(const float* src, size_t step1, int* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(float);
    const size_t dst_row_step = step2 / sizeof(int);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(float);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const float* it_src = src;
        int* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
            v_store(it_dst, v_ceil( vx_load(it_src) ) );
#endif

        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = cvCeil(*it_dst);
    }
}


void ceil64f(const double* src, size_t step1, double* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(double);
    const size_t dst_row_step = step2 / sizeof(double);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(double);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const double* it_src = src;
        double* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_float64 v_src = vx_load(it_src);
#ifdef CV_AVX
            v_src.val = _mm256_ceil_pd(v_src.val);
#elif defined(CV_SSE4_1)
            v_src.val = _mm_ceil_pd(v_src.val);
#else
            v_src = v_cvt_f64(v_ceil(v_src));
#endif
            v_store(it_dst, v_src);
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = saturate_cast<double>(cvCeil(*it_dst));
    }
}

void ceil64f(const double* src, size_t step1, int* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(double);
    const size_t dst_row_step = step2 / sizeof(int);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(double);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const double* it_src = src;
        int* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_int32 v_a = v_ceil( vx_load(it_src) );
            v_int32 v_b = v_ceil( vx_load(it_src + inc/2) );

            v_store(it_dst, v_combine_low(v_a, v_b) );
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = cvCeil(*it_dst);
    }
}







void floor32f(const float* src, size_t step1, float* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(float);
    const size_t dst_row_step = step2 / sizeof(float);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(float);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const float* it_src = src;
        float* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_float32 v_src = vx_load(it_src);
            // The only reason why some intrinsics code is written, is to avoid a un-necessary type convertion.
#ifdef CV_AVX
            v_src.val = _mm256_floor_ps(v_src.val);
#elif defined(CV_SSE4_1)
            v_src.val = _mm_floor_ps(v_src.val);
#else
            v_src = v_cvt_f32(v_ceil(v_src));
#endif
            v_store(it_dst, v_src);
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = saturate_cast<float>(cvFloor(*it_src));
    }
}

void floor32f(const float* src, size_t step1, int* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(float);
    const size_t dst_row_step = step2 / sizeof(int);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(float);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const float* it_src = src;
        int* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
            v_store(it_dst, v_floor( vx_load(it_src) ) );
#endif

        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = cvFloor(*it_src);
    }
}

void floor64f(const double* src, size_t step1, double* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(double);
    const size_t dst_row_step = step2 / sizeof(double);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(double);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const double* it_src = src;
        double* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_float64 v_src = vx_load(it_src);
#ifdef CV_AVX
            v_src.val = _mm256_floor_pd(v_src.val);
#elif defined(CV_SSE4_1)
            v_src.val = _mm_floor_pd(v_src.val);
#else
            v_src = v_cvt_f64(v_floor(v_src));
#endif
            v_store(it_dst, v_src);
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = saturate_cast<double>(cvFloor(*it_src));
    }
}

void floor64f(const double* src, size_t step1, int* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(double);
    const size_t dst_row_step = step2 / sizeof(int);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(double);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const double* it_src = src;
        int* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_int32 v_a = v_floor( vx_load(it_src) );
            v_int32 v_b = v_floor( vx_load(it_src + inc/2) );

            v_store(it_dst, v_combine_low(v_a, v_b) );
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = cvFloor(*it_src);
    }
}




void round32f(const float* src, size_t step1, float* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(float);
    const size_t dst_row_step = step2 / sizeof(float);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(float);
    const int vec_width = width - (width%inc);
#endif

    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const float* it_src = src;
        float* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_float32 v_src = vx_load(it_src);
#ifdef CV_AVX
            v_src.val = _mm256_round_ps(v_src.val, _MM_FROUND_NINT);
#elif defined(CV_SSE4_1)
            v_src.val = _mm_round_ps(v_src.val, _MM_FROUND_NINT);
#else
            v_src = v_cvt_f32(v_round(v_src));
#endif
            v_store(it_dst, v_src);
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = saturate_cast<float>(cvRound(*it_src));
    }
}

void round32f(const float* src, size_t step1, int* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(float);
    const size_t dst_row_step = step2 / sizeof(int);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(float);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const float* it_src = src;
        int* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
            v_store(it_dst, v_round( vx_load(it_src) ) );
#endif

        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = cvRound(*it_src);
    }
}

void round64f(const double* src, size_t step1, double* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(double);
    const size_t dst_row_step = step2 / sizeof(double);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(double);
    const int vec_width = width - (width%inc);
#endif
    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const double* it_src = src;
        double* it_dst = dst;

#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_float64 v_src = vx_load(it_src);
#ifdef CV_AVX
            v_src.val = _mm256_round_pd(v_src.val, _MM_FROUND_NINT);
#elif defined(CV_SSE4_1)
            v_src.val = _mm_round_pd(v_src.val, _MM_FROUND_NINT);
#else
            v_src = v_cvt_f64(v_round(v_src));
#endif
            v_store(it_dst, v_src);
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = saturate_cast<double>(cvRound(*it_src));
    }
}


void round64f(const double* src, size_t step1, int* dst, size_t step2, int width, int height)
{
    const size_t src_row_step = step1 / sizeof(double);
    const size_t dst_row_step = step2 / sizeof(int);

#ifdef CV_ENABLE_INTRINSICS
    const int inc = CV_SIMD_WIDTH/sizeof(inc);
    const int vec_width = width - (width%inc);
#endif

    for(int r=0, c=0; r<height; r++, c=0, src+=src_row_step, dst+=dst_row_step)
    {
        const double* it_src = src;
        int* it_dst = dst;
#ifdef CV_ENABLE_INTRINSICS
        for(;c<vec_width; c+=inc, it_src+=inc, it_dst+=inc)
        {
            v_int32 v_a = v_round(vx_load(it_src));
            v_int32 v_b = v_round(vx_load(it_src + inc/2));

            v_store(it_dst, v_combine_low(v_a, v_b) );
        }
#endif
        for(;c<width; c++, it_src++, it_dst++)
            *it_dst = cvRound(*it_src);
    }
}

#if 0

namespace
{

template<class T1, class Tvec>
struct OpFMA
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_fma(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return std::fma(_a, _b, _c);
    }

};


template<class T1, class Tvec>
struct OpFMS
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_fms(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return std::fma(_a, _b, -_c);
    }

};

template<class T1, class Tvec>
struct OpNFMA
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_nfma(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return std::fma(-_a, _b, _c);
    }

};

template<class T1, class Tvec>
struct OpNFMS
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_nfms(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return std::nfms(-_a, _b, -_c);
    }

};

template<class T1, class Tvec>
struct OpFDA
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_fda(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return (_a / _b) + _c;
    }

};


template<class T1, class Tvec>
struct OpFDS
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_fds(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return (_a / _b) - _c;
    }

};

template<class T1, class Tvec>
struct OpNFDA
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_nfda(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return (-_a / _b) + _c;
    }

};

template<class T1, class Tvec>
struct OpNFDS
{

    static Tvec r(const Tvec& _a, const Tvec& _b, const Tvec& _c)
    {
        return v_nfds(_a, _b, _c);
    }

    static T1 r(const T1& _a, const T1& _b, const T1& _c)
    {
        return (-_a / _b) - _c;
    }

};


template< template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct tri_loader
{
    typedef OP<T1, Tvec> op;

    static inline void l(const T1* src1, const T1* src2, const T1* src3, T1* dst)
    {
        Tvec a = vx_load(src1);
        Tvec b = vx_load(src2);
        Tvec c = vx_load(src3);
        v_store(dst, op::r(a, b, c));
    }

    static inline void la(const T1* src1, const T1* src2, const T1* src3, T1* dst)
    {
        Tvec a = vx_load_aligned(src1);
        Tvec b = vx_load_aligned(src2);
        Tvec c = vx_load_aligned(src3);
        v_store_aligned(dst, op::r(a, b, c));
    }

//    static inline void lasnc(const T1* src1, const T1* src2, const T1* src3, T1* dst)
//    {
//        Tvec a = vx_load_aligned_nocache(src1);
//        Tvec b = vx_load_aligned(src2);
//        Tvec c = vx_load_aligned(src3);
//        v_store_aligned_nocache(dst, op::r(a, b, c));
//    }


    static inline void l64(const T1* src1, const T1* src2, const T1* src3, T1* dst)
    {
        Tvec a = vx_load_low(src1), b = vx_load_low(src2), c = vx_load_low(src3);
        v_store_low(dst, op::r(a, b, c));
    }
};


template<typename T1, typename T2>
static inline bool is_aligned(const T1* src1, const T1* src2, const T1* src3, const T2* dst)
{ return (((size_t)src1|(size_t)src2|(size_t)src3|(size_t)dst) & (CV_SIMD_WIDTH - 1)) == 0; }


template<template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
void tri_loop(const T1* src1, size_t step1, const T1* src2, size_t step2, const T1* src3, size_t step3, T1* dst, size_t step, int width, int height)
{
    typedef OP<T1, Tvec> op;
#if CV_SIMD
    typedef tri_loader<OP, T1, Tvec> ldr;
    enum {wide_step = Tvec::nlanes};
    #if !CV_NEON && CV_SIMD_WIDTH == 16
        enum {wide_step_l = wide_step * 2};
    #else
        enum {wide_step_l = wide_step};
    #endif
#endif // CV_SIMD

    step1 /= sizeof(T1);
    step2 /= sizeof(T1);
    step3 /= sizeof(T1);
    step  /= sizeof(T1);

    for (; height--; src1 += step1, src2 += step2, src3 += step3, dst += step)
    {
        int x = 0;

    #if CV_SIMD
        #if !CV_NEON && !CV_MSA
        if (is_aligned(src1, src2, src3, dst))
        {
            for (; x <= width - wide_step_l; x += wide_step_l)
            {
                ldr::la(src1 + x, src2 + x, src3 + x, dst + x);
                #if CV_SIMD_WIDTH == 16
                ldr::la(src1 + x + wide_step, src2 + x + wide_step, dst + x + wide_step);
                #endif
            }
        }
        else
        #endif
            for (; x <= width - wide_step_l; x += wide_step_l)
            {
                ldr::l(src1 + x, src2 + x, src3 + x, dst + x);
                #if !CV_NEON && CV_SIMD_WIDTH == 16
                ldr::l(src1 + x + wide_step, src2 + x + wide_step, dst + x + wide_step);
                #endif
            }

        #if CV_SIMD_WIDTH == 16
        for (; x <= width - 8/(int)sizeof(T1); x += 8/(int)sizeof(T1))
        {
            ldr::l64(src1 + x, src2 + x, src3 + x, dst + x);
        }
        #endif
    #endif // CV_SIMD

    #if CV_ENABLE_UNROLLED || CV_SIMD_WIDTH > 16
        for (; x <= width - 4; x += 4)
        {
            T1 t0 = op::r(src1[x], src2[x], src3[x]);
            T1 t1 = op::r(src1[x + 1], src2[x + 1], src3[x + 1]);
            dst[x] = t0; dst[x + 1] = t1;

            t0 = op::r(src1[x + 2], src2[x + 2], src3[x + 2]);
            t1 = op::r(src1[x + 3], src2[x + 3], src3[x + 3]);
            dst[x + 2] = t0; dst[x + 3] = t1;
        }
    #endif

        for (; x < width; x++)
            dst[x] = op::r(src1[x], src2[x], src3[x]);
    }

    vx_cleanup();
}

#define DECL_FMX_TEMPLATE_(name, opname)\
template<class T>\
void name ## _ ( const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height)\
{\
    tri_loop<opname, T, typename Type2Vec_Traits<T>::vec_type>(a, step1, b, step2, c, step3, d, step4, width, height);\
}

DECL_FMX_TEMPLATE_(fma, OpFMA)
DECL_FMX_TEMPLATE_(fms, OpFMS)
DECL_FMX_TEMPLATE_(nfma, OpNFMA)
DECL_FMX_TEMPLATE_(nfms, OpNFMS)
DECL_FMX_TEMPLATE_(fda, OpFDA)
DECL_FMX_TEMPLATE_(fds, OpFDS)
DECL_FMX_TEMPLATE_(nfda, OpNFDA)
DECL_FMX_TEMPLATE_(nfds, OpNFDS)

} //anonymous

void fma32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    fma_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void fms32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    fms_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfma32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    nfma_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfms32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    nfms_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void fda32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    fda_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void fds32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    fds_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfda32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    nfda_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfds32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height)
{
    nfds_(a, step1, b, step2, c, step3, d, step4, width, height);
}


//

void fma64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    fma_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void fms64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    fms_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfma64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    nfma_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfms64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    nfms_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void fda64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    fda_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void fds64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    fds_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfda64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    nfda_(a, step1, b, step2, c, step3, d, step4, width, height);
}

void nfds64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height)
{
    nfds_(a, step1, b, step2, c, step3, d, step4, width, height);
}

#define IMPL_FMX_SPECS(name)\
    template<> void name<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height){ name ## _(a, step1, b, step2, c, step3, d, step4, width, height);} \
    template<> void name<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height){ name ## _(a, step1, b, step2, c, step3, d, step4, width, height);}

IMPL_FMX_SPECS(fma)
IMPL_FMX_SPECS(fms)
IMPL_FMX_SPECS(nfma)
IMPL_FMX_SPECS(nfms)

IMPL_FMX_SPECS(fda)
IMPL_FMX_SPECS(fds)
IMPL_FMX_SPECS(nfda)
IMPL_FMX_SPECS(nfds)



#endif





} // hal

} // cv
