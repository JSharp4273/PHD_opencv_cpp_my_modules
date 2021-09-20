#include "opencv2/xcore/template/intrin.hpp"

namespace cv
{



#if !CV_FMA3

#if CV_SIMD128
v_float32x4 v_fms(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    static v_float32x4 v_minus_one = v_setall_f32(-1.f);

    return v_fma(_a, _b, v_minus_one * _c);
}
v_float32x4 v_nfma(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    static v_float32x4 minus_one = v_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, _c);
}
v_float32x4 v_nfms(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    static v_float32x4 minus_one = v_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, minus_one * _c);
}
v_float32x4 v_fda(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    static const v_float32x4 ones = v_setall_f32(1.f);
    return v_fma(_a, ones / _b, _c);
}
v_float32x4 v_fds(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    static const v_float32x4 ones = v_setall_f32(1.f);
    static const v_float32x4 minus_ones = v_setall_f32(-1.f);
    return v_fma(_a, ones / _b, minus_ones * _c);
}
v_float32x4 v_nfda(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    static const v_float32x4 ones = v_setall_f32(1.f);
    static const v_float32x4 minus_ones = v_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, _c);
}
v_float32x4 v_nfds(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    static const v_float32x4 ones = v_setall_f32(1.f);
    static const v_float32x4 minus_ones = v_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, minus_ones * _c);
}
#endif

#if CV_SIMD256
v_float32x8 v_fms(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    static v_float32x8 minus_one = v256_setall_f32(-1.f);

    return v_fma(_a, _b, minus_one * _c);
}
v_float32x8 v_nfma(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    static v_float32x8 minus_one = v256_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, _c);
}
v_float32x8 v_nfms(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    static v_float32x8 minus_one = v256_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, minus_one * _c);
}
v_float32x8 v_fda(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    static const v_float32x8 ones = v256_setall_f32(1.f);
    return v_fma(_a, ones / _b, _c);
}
v_float32x8 v_fds(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    static const v_float32x8 ones = v256_setall_f32(1.f);
    static const v_float32x8 minus_ones = v256_setall_f32(-1.f);
    return v_fma(_a, ones / _b, minus_ones * _c);
}
v_float32x8 v_nfda(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    static const v_float32x8 ones = v256_setall_f32(1.f);
    static const v_float32x8 minus_ones = v256_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, _c);
}
v_float32x8 v_nfds(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    static const v_float32x8 ones = v256_setall_f32(1.f);
    static const v_float32x8 minus_ones = v256_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, minus_ones * _c);
}
#endif



//

#if CV_SIMD128_64F
v_float64x2 v_fms(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    static v_float64x2 v_minus_one = v_setall_f32(-1.f);

    return v_fma(_a, _b, v_minus_one * _c);
}
v_float64x2 v_nfma(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    static v_float64x2 minus_one = v_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, _c);
}
v_float64x2 v_nfms(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    static v_float64x2 minus_one = v_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, minus_one * _c);
}
v_float64x2 v_fda(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    static const v_float64x2 ones = v_setall_f32(1.f);
    return v_fma(_a, ones / _b, _c);
}
v_float64x2 v_fds(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    static const v_float64x2 ones = v_setall_f32(1.f);
    static const v_float64x2 minus_ones = v_setall_f32(-1.f);
    return v_fma(_a, ones / _b, minus_ones * _c);
}
v_float64x2 v_nfda(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    static const v_float64x2 ones = v_setall_f32(1.f);
    static const v_float64x2 minus_ones = v_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, _c);
}
v_float64x2 v_nfds(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    static const v_float64x2 ones = v_setall_f32(1.f);
    static const v_float64x2 minus_ones = v_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, minus_ones * _c);
}
#endif

#if CV_SIMD256_64F
v_float64x4 v_fms(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    static v_float64x4 minus_one = v256_setall_f32(-1.f);

    return v_fma(_a, _b, minus_one * _c);
}
v_float64x4 v_nfma(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    static v_float64x4 minus_one = v256_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, _c);
}
v_float64x4 v_nfms(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    static v_float64x4 minus_one = v256_setall_f32(-1.f);

    return v_fma(minus_one * _a, _b, minus_one * _c);
}
v_float64x4 v_fda(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    static const v_float64x4 ones = v256_setall_f32(1.f);
    return v_fma(_a, ones / _b, _c);
}
v_float64x4 v_fds(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    static const v_float64x4 ones = v256_setall_f32(1.f);
    static const v_float64x4 minus_ones = v256_setall_f32(-1.f);
    return v_fma(_a, ones / _b, minus_ones * _c);
}
v_float64x4 v_nfda(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    static const v_float64x4 ones = v256_setall_f32(1.f);
    static const v_float64x4 minus_ones = v256_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, _c);
}
v_float64x4 v_nfds(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    static const v_float64x4 ones = v256_setall_f32(1.f);
    static const v_float64x4 minus_ones = v256_setall_f32(-1.f);
    return v_fma(minus_ones * _a, ones / _b, minus_ones * _c);
}
#endif


#else

#if CV_SIMD128
v_float32x4 v_fms(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    v_float32x4 ret;

    ret.val = _mm_fmsub_ps(_a.val, _b.val, _c.val);

    return ret;
}
v_float32x4 v_nfma(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    v_float32x4 ret;

    static const __m128 _na = _mm_set1_ps(-1.f);

    ret.val = _mm_fmadd_ps(_mm_mul_ps(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float32x4 v_nfms(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    v_float32x4 ret;

    static const __m128 _na = _mm_set1_ps(-1.f);

    ret.val = _mm_fmsub_ps(_mm_mul_ps(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float32x4 v_fda(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    v_float32x4 ret;

    ret.val = _mm_fmadd_ps(_a.val, _mm_rcp_ps(_b.val), _c.val);

    return ret;
}
v_float32x4 v_fds(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    v_float32x4 ret;

    ret.val = _mm_fmsub_ps(_a.val, _mm_rcp_ps(_b.val), _c.val);

    return ret;
}
v_float32x4 v_nfda(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    v_float32x4 ret;

    static const __m128 _neg = _mm_set1_ps(-1.f);

    ret.val = _mm_fmadd_ps(_mm_mul_ps(_a.val, _neg), _mm_rcp_ps(_b.val), _c.val);

    return ret;
}
v_float32x4 v_nfds(const v_float32x4& _a, const v_float32x4& _b, const v_float32x4& _c)
{
    v_float32x4 ret;

    static const __m128 _neg = _mm_set1_ps(-1.f);

    ret.val = _mm_fmsub_ps(_mm_mul_ps(_a.val, _neg), _mm_rcp_ps(_b.val), _c.val);

    return ret;
}
#endif

#if CV_SIMD256
v_float32x8 v_fms(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    v_float32x8 ret;

    ret.val = _mm256_fmsub_ps(_a.val, _b.val, _c.val);

    return ret;
}
v_float32x8 v_nfma(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    v_float32x8 ret;

    static const __m256 _na = _mm256_set1_ps(-1.f);

    ret.val = _mm256_fmadd_ps(_mm256_mul_ps(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float32x8 v_nfms(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    v_float32x8 ret;

    static const __m256 _na = _mm256_set1_ps(-1.f);

    ret.val = _mm256_fmsub_ps(_mm256_mul_ps(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float32x8 v_fda(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    v_float32x8 ret;

    ret.val = _mm256_fmadd_ps(_a.val, _mm256_rcp_ps(_b.val), _c.val);

    return ret;
}
v_float32x8 v_fds(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    v_float32x8 ret;

    ret.val = _mm256_fmsub_ps(_a.val, _mm256_rcp_ps(_b.val), _c.val);

    return ret;
}
v_float32x8 v_nfda(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    v_float32x8 ret;

    static const __m256 _neg = _mm256_set1_ps(-1.f);

    ret.val = _mm256_fmadd_ps(_mm256_mul_ps(_a.val, _neg), _mm256_rcp_ps(_b.val), _c.val);

    return ret;
}
v_float32x8 v_nfds(const v_float32x8& _a, const v_float32x8& _b, const v_float32x8& _c)
{
    v_float32x8 ret;

    static const __m256 _neg = _mm256_set1_ps(-1.f);

    ret.val = _mm256_fmsub_ps(_mm256_mul_ps(_a.val, _neg), _mm256_rcp_ps(_b.val), _c.val);

    return ret;
}
#endif




#if CV_SIMD128_64F
v_float64x2 v_fms(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    v_float64x2 ret;

    ret.val = _mm_fmsub_pd(_a.val, _b.val, _c.val);

    return ret;
}
v_float64x2 v_nfma(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    v_float64x2 ret;

    static const __m128d _na = _mm_set1_pd(-1.f);

    ret.val = _mm_fmadd_pd(_mm_mul_pd(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float64x2 v_nfms(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    v_float64x2 ret;

    static const __m128d _na = _mm_set1_pd(-1.f);

    ret.val = _mm_fmsub_pd(_mm_mul_pd(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float64x2 v_fda(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    v_float64x2 ret;

    static const __m128d ones = _mm_set1_pd(1.);

    ret.val = _mm_fmadd_pd(_a.val, _mm_div_pd(ones, _b.val), _c.val);

    return ret;
}
v_float64x2 v_fds(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    v_float64x2 ret;

    static const __m128d ones = _mm_set1_pd(1.);

    ret.val = _mm_fmsub_pd(_a.val, _mm_div_pd(ones, _b.val), _c.val);

    return ret;
}
v_float64x2 v_nfda(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    v_float64x2 ret;

    static const __m128d neg = _mm_set1_pd(-1.f);
    static const __m128d ones = _mm_set1_pd(1.);

    ret.val = _mm_fmadd_pd(_mm_mul_pd(_a.val, neg), _mm_div_pd(ones, _b.val), _c.val);

    return ret;
}
v_float64x2 v_nfds(const v_float64x2& _a, const v_float64x2& _b, const v_float64x2& _c)
{
    v_float64x2 ret;

    static const __m128d neg = _mm_set1_pd(-1.f);
    static const __m128d ones = _mm_set1_pd(1.);

    ret.val = _mm_fmsub_pd(_mm_mul_pd(_a.val, neg), _mm_div_pd(ones, _b.val), _c.val);

    return ret;
}
#endif

#if CV_SIMD256_64F
v_float64x4 v_fms(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    v_float64x4 ret;

    ret.val = _mm256_fmsub_pd(_a.val, _b.val, _c.val);

    return ret;
}
v_float64x4 v_nfma(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    v_float64x4 ret;

    static const __m256d _na = _mm256_set1_pd(-1.f);

    ret.val = _mm256_fmadd_pd(_mm256_mul_pd(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float64x4 v_nfms(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    v_float64x4 ret;

    static const __m256d _na = _mm256_set1_pd(-1.f);

    ret.val = _mm256_fmsub_pd(_mm256_mul_pd(_a.val, _na), _b.val, _c.val);

    return ret;
}
v_float64x4 v_fda(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    v_float64x4 ret;

    static const __m256d ones = _mm256_set1_pd(1.);

    ret.val = _mm256_fmadd_pd(_a.val, _mm256_div_pd(ones, _b.val), _c.val);

    return ret;
}
v_float64x4 v_fds(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    v_float64x4 ret;

    static const __m256d ones = _mm256_set1_pd(1.);

    ret.val = _mm256_fmsub_pd(_a.val, _mm256_div_pd(ones, _b.val), _c.val);

    return ret;
}
v_float64x4 v_nfda(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    v_float64x4 ret;

    static const __m256d neg = _mm256_set1_pd(-1.f);
    static const __m256d ones = _mm256_set1_pd(1.);

    ret.val = _mm256_fmadd_pd(_mm256_mul_pd(_a.val, neg), _mm256_div_pd(ones, _b.val), _c.val);

    return ret;
}
v_float64x4 v_nfds(const v_float64x4& _a, const v_float64x4& _b, const v_float64x4& _c)
{
    v_float64x4 ret;

    static const __m256d neg = _mm256_set1_pd(-1.f);
    static const __m256d ones = _mm256_set1_pd(1.);

    ret.val = _mm256_fmsub_pd(_mm256_mul_pd(_a.val, neg), _mm256_div_pd(ones, _b.val), _c.val);

    return ret;
}
#endif

#endif

} //cv
