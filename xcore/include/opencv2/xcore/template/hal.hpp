#ifndef XCORE_HAL_HPP
#define XCORE_HAL_HPP

#include "opencv2/core/hal/hal.hpp"

namespace cv
{

namespace hal
{

#define DECL_SPEC_ARITHM_FUN_(fun, type)\
    template<> void fun<type>( const type* src1, size_t step1, const type* src2, size_t step2, type* dst, size_t step, int width, int height, void* smth);

#define DECL_SPEC_ARITHM_FUN(fun)\
    DECL_SPEC_ARITHM_FUN_(fun, uchar)\
    DECL_SPEC_ARITHM_FUN_(fun, schar)\
    DECL_SPEC_ARITHM_FUN_(fun, ushort)\
    DECL_SPEC_ARITHM_FUN_(fun, short)\
    DECL_SPEC_ARITHM_FUN_(fun, int)\
    DECL_SPEC_ARITHM_FUN_(fun, float)\
    DECL_SPEC_ARITHM_FUN_(fun, double)


#define DECL_SPEC_CMP_FUN_(fun, type)\
    template<> void fun<type>( const type* src1, size_t step1, const type* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* smth);

#define DECL_SPEC_CMP_FUN(fun)\
    DECL_SPEC_CMP_FUN_(fun, uchar)\
    DECL_SPEC_CMP_FUN_(fun, schar)\
    DECL_SPEC_CMP_FUN_(fun, ushort)\
    DECL_SPEC_CMP_FUN_(fun, short)\
    DECL_SPEC_CMP_FUN_(fun, int)\
    DECL_SPEC_CMP_FUN_(fun, float)\
    DECL_SPEC_CMP_FUN_(fun, double)


template<class T>
void add( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void*);

DECL_SPEC_ARITHM_FUN(add)

template<class T>
void sub( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void*);

DECL_SPEC_ARITHM_FUN(sub)

template<class T>
void max( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void*);

DECL_SPEC_ARITHM_FUN(max)

template<class T>
void min( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void*);

DECL_SPEC_ARITHM_FUN(min)

template<class T>
void absdiff( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void*);

DECL_SPEC_ARITHM_FUN(absdiff)

template<class T>
void cmp( const T* src1, size_t step1, const T* src2, size_t step2, uchar* dst, size_t step, int width, int height, void* _cmpop);

DECL_SPEC_CMP_FUN(cmp)

template<class T>
void mul( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void* scale);

DECL_SPEC_ARITHM_FUN(mul)

template<class T>
void div( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void* scale);

DECL_SPEC_ARITHM_FUN(div)

template<class T>
void recip( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void* scale);

DECL_SPEC_ARITHM_FUN(recip)

template<class T>
void addWeighted( const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height, void* _scalars );

DECL_SPEC_ARITHM_FUN(addWeighted)

#undef DECL_SPEC_ARITHM_FUN
#undef DECL_SPEC_ARITHM_FUN_


void ceil32f(const float* src, size_t step1, float* dst, size_t step2, int width, int height);
void ceil32f(const float* src, size_t step1, int* dst, size_t step2, int width, int height);

void ceil64f(const double* src, size_t step1, double* dst, size_t step2, int width, int height);
void ceil64f(const double* src, size_t step1, int* dst, size_t step2, int width, int height);


template<class S, class D>
void ceil(const S* src, size_t step1, D* dst, size_t step2, int width, int height);

template<>
void ceil<float, float>(const float* src, size_t step1, float* dst, size_t step2, int width, int height);

template<>
void ceil<float, int>(const float* src, size_t step1, int* dst, size_t step2, int width, int height);

template<>
void ceil<double, double>(const double* src, size_t step1, double* dst, size_t step2, int width, int height);

template<>
void ceil<double, int>(const double* src, size_t step1, int* dst, size_t step2, int width, int height);



void floor32f(const float* src, size_t step1, float* dst, size_t step2, int width, int height);
void floor32f(const float* src, size_t step1, int* dst, size_t step2, int width, int height);

void floor64f(const double* src, size_t step1, double* dst, size_t step2, int width, int height);
void floor64f(const double* src, size_t step1, int* dst, size_t step2, int width, int height);


template<class S, class D>
void floor(const S* src, size_t step1, D* dst, size_t step2, int width, int height);

template<>
void floor<float, float>(const float* src, size_t step1, float* dst, size_t step2, int width, int height);

template<>
void floor<float, int>(const float* src, size_t step1, int* dst, size_t step2, int width, int height);

template<>
void floor<double, double>(const double* src, size_t step1, double* dst, size_t step2, int width, int height);

template<>
void floor<double, int>(const double* src, size_t step1, int* dst, size_t step2, int width, int height);



void round32f(const float* src, size_t step1, float* dst, size_t step2, int width, int height);
void round32f(const float* src, size_t step1, int* dst, size_t step2, int width, int height);

void round64f(const double* src, size_t step1, double* dst, size_t step2, int width, int height);
void round64f(const double* src, size_t step1, int* dst, size_t step2, int width, int height);


template<class S, class D>
void round(const S* src, size_t step1, D* dst, size_t step2, int width, int height);

template<>
void round<float, float>(const float* src, size_t step1, float* dst, size_t step2, int width, int height);

template<>
void round<float, int>(const float* src, size_t step1, int* dst, size_t step2, int width, int height);

template<>
void round<double, double>(const double* src, size_t step1, double* dst, size_t step2, int width, int height);

template<>
void round<double, int>(const double* src, size_t step1, int* dst, size_t step2, int width, int height);


#if 0 // development in progress
void fma32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
void fms32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
void nfma32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
void nfms32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
void fda32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
void fds32f( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
void nfda32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
void nfds32f(const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);

void fma64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
void fms64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
void nfma64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
void nfms64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
void fda64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
void fds64f( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
void nfda64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
void nfds64f(const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);

template<class T>
void fma( const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void fma<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void fma<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);

template<class T>
void fms( const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void fms<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void fms<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);


template<class T>
void nfma(const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void nfma<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void nfma<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);


template<class T>
void nfms(const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void nfms<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void nfms<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);


template<class T>
void fda( const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void fda<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void fda<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);


template<class T>
void fds( const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void fds<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void fds<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);


template<class T>
void nfda(const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void nfda<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void nfda<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);


template<class T>
void nfds(const T* a, size_t step1, const T* b, size_t step2, const T* c, size_t step3, T* d, size_t step4, int width, int height);

template <>
void nfds<float>( const float* a, size_t step1, const float* b, size_t step2, const float* c, size_t step3, float* d, size_t step4, int width, int height);
template <>
void nfds<double>( const double* a, size_t step1, const double* b, size_t step2, const double* c, size_t step3, double* d, size_t step4, int width, int height);
#endif

} // hal

}// cv

#endif // XCORE_HAL_HPP
