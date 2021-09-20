#ifndef CUDEV_FUNCTIONAL_EXT_HPP
#define CUDEV_FUNCTIONAL_EXT_HPP

#if 0

// Inspired by cudev functions
#include <opencv2/cudev/functional/functional.hpp>

namespace cv
{
namespace cudev
{

template <typename _Arg1, typename _Arg2, typename _Arg3, typename _Result>
struct ternary_function
{
    typedef _Arg1   first_argument_type;
    typedef _Arg2   second_argument_type;
    typedef _Arg3   third_argument_type;
    typedef _Result result_type;
};

template <typename _Arg1, typename _Arg2, typename _Arg3, typename _Arg4, typename _Result>
struct quaternary_function
{
    typedef _Arg1   first_argument_type;
    typedef _Arg2   second_argument_type;
    typedef _Arg3   third_argument_type;
    typedef _Arg4   fourth_argument_type;
    typedef _Result result_type;
};

template <typename _Arg1, typename _Arg2, typename _Arg3, typename _Arg4, typename _Arg5, typename _Result>
struct quinary_function
{
    typedef _Arg1   first_argument_type;
    typedef _Arg2   second_argument_type;
    typedef _Arg3   third_argument_type;
    typedef _Arg4   fourth_argument_type;
    typedef _Arg5   fifth_argument_type;
    typedef _Result result_type;
};

template <typename _Arg1, typename _Arg2, typename _Arg3, typename _Arg4, typename _Arg5, typename _Arg6, typename _Result>
struct senary_function
{
    typedef _Arg1   first_argument_type;
    typedef _Arg2   second_argument_type;
    typedef _Arg3   third_argument_type;
    typedef _Arg4   fourth_argument_type;
    typedef _Arg5   fifth_argument_type;
    typedef _Arg6   sixth_argument_type;
    typedef _Result result_type;
};



// Rename in order to do not interface with the definition in : opencv2/cudev/functional/functional.hpp
#define CV_CUDEV_UNARY_FUNCTION_INST_EXT(name, func) \
    template <typename T> struct name ## _func : unary_function<T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a) const \
        { \
            return name(a); \
        } \
    }; \
    template <> struct name ## _func<uchar> : unary_function<uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<schar> : unary_function<schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<ushort> : unary_function<ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<short> : unary_function<short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<uint> : unary_function<uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<int> : unary_function<int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<float> : unary_function<float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a) const \
        { \
            return func ## f(a); \
        } \
    }; \
    template <> struct name ## _func<double> : unary_function<double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a) const \
        { \
            return func(a); \
        } \
    };


// This trick is nice :).
//#undef CV_CUDEV_UNARY_FUNCTION_INST_EXT


#define CV_CUDEV_BINARY_FUNCTION_INST_EXT(name, func) \
    template <typename T> struct name ## _func : binary_function<T, T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b) const \
        { \
            return name(a, b); \
        } \
    }; \
    template <> struct name ## _func<uchar> : binary_function<uchar, uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a, uchar b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<schar> : binary_function<schar, schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a, schar b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<ushort> : binary_function<ushort, ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a, ushort b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<short> : binary_function<short, short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a, short b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<uint> : binary_function<uint, uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a, uint b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<int> : binary_function<int, int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a, int b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<float> : binary_function<float, float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a, float b) const \
        { \
            return func ## f(a, b); \
        } \
    }; \
    template <> struct name ## _func<double> : binary_function<double, double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a, double b) const \
        { \
            return func(a, b); \
        } \
    };


#define CV_CUDEV_TERNARY_FUNCTION_INST(name, func) \
    template <typename T> struct name ## _func : ternary_function<T, T, T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b, typename TypeTraits<T>::parameter_type c) const \
        { \
            return name(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<uchar> : ternary_function<uchar, uchar, uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a, uchar b, uchar c) const \
        { \
            return func ## f(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<schar> : ternary_function<schar, schar, schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a, schar b, schar c) const \
        { \
            return func ## f(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<ushort> : ternary_function<ushort, ushort, ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a, ushort b, ushort c) const \
        { \
            return func ## f(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<short> : ternary_function<short, short, short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a, short b, short c) const \
        { \
            return func ## f(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<uint> : ternary_function<uint, uint, uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a, uint b, uint c) const \
        { \
            return func ## f(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<int> : ternary_function<int, int, int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a, int b, int c) const \
        { \
            return func ## f(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<float> : ternary_function<float, float, float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a, float b, float c) const \
        { \
            return func ## f(a, b, c); \
        } \
    }; \
    template <> struct name ## _func<double> : ternary_function<double, double, double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a, double b,double c) const \
        { \
            return func(a, b, c); \
        } \
    };


#define CV_CUDEV_QUATERNARY_FUNCTION_INST(name, func) \
    template <typename T> struct name ## _func : quaternary_function<T, T, T, T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b, typename TypeTraits<T>::parameter_type c, typename TypeTraits<T>::parameter_type d) const \
        { \
            return name(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<uchar> : quaternary_function<uchar, uchar, uchar, uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a, uchar b, uchar c, uchar d) const \
        { \
            return func ## f(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<schar> : quaternary_function<schar, schar, schar, schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a, schar b, schar c, schar d) const \
        { \
            return func ## f(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<ushort> : quaternary_function<ushort, ushort, ushort, ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a, ushort b, ushort c, ushort d) const \
        { \
            return func ## f(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<short> : quaternary_function<short, short, short, short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a, short b, short c, short d) const \
        { \
            return func ## f(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<uint> : quaternary_function<uint, uint, uint, uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a, uint b, uint c, uint d) const \
        { \
            return func ## f(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<int> : quaternary_function<int, int, int, int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a, int b, int c, int d) const \
        { \
            return func ## f(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<float> : quaternary_function<float, float, float, float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a, float b, float c, float d) const \
        { \
            return func ## f(a, b, c, d); \
        } \
    }; \
    template <> struct name ## _func<double> : quaternary_function<double, double, double, double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a, double b,double c, double d) const \
        { \
            return func(a, b, c, d); \
        } \
    };


#define CV_CUDEV_QUINARY_FUNCTION_INST(name, func) \
    template <typename T> struct name ## _func : quinary_function<T, T, T, T, T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b, typename TypeTraits<T>::parameter_type c, typename TypeTraits<T>::parameter_type d, typename TypeTraits<T>::parameter_type e) const \
        { \
            return name(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<uchar> : quinary_function<uchar, uchar, uchar, uchar, uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a, uchar b, uchar c, uchar d, uchar e) const \
        { \
            return func ## f(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<schar> : quinary_function<schar, schar, schar, schar, schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a, schar b, schar c, schar d, schar e) const \
        { \
            return func ## f(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<ushort> : quinary_function<ushort, ushort, ushort, ushort, ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a, ushort b, ushort c, ushort d, ushort e) const \
        { \
            return func ## f(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<short> : quinary_function<short, short, short, short, short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a, short b, short c, short d, short e) const \
        { \
            return func ## f(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<uint> : quinary_function<uint, uint, uint, uint, uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a, uint b, uint c, uint d, uint e) const \
        { \
            return func ## f(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<int> : quinary_function<int, int, int, int, int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a, int b, int c, int d, int e) const \
        { \
            return func ## f(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<float> : quinary_function<float, float, float, float, float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a, float b, float c, float d, float e) const \
        { \
            return func ## f(a, b, c, d, e); \
        } \
    }; \
    template <> struct name ## _func<double> : quinary_function<double, double, double, double, double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a, double b,double c, double d, double e) const \
        { \
            return func(a, b, c, d, e); \
        } \
    };


#define CV_CUDEV_SENARY_FUNCTION_INST(name, func) \
    template <typename T> struct name ## _func : senary_function<T, T, T, T, T, T, typename functional_detail::FloatType<T>::type> \
    { \
        __device__ __forceinline__ typename functional_detail::FloatType<T>::type operator ()(typename TypeTraits<T>::parameter_type a, typename TypeTraits<T>::parameter_type b, typename TypeTraits<T>::parameter_type c, typename TypeTraits<T>::parameter_type d, typename TypeTraits<T>::parameter_type e, typename TypeTraits<T>::parameter_type f) const \
        { \
            return name(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<uchar> : senary_function<uchar, uchar, uchar, uchar, uchar, uchar, float> \
    { \
        __device__ __forceinline__ float operator ()(uchar a, uchar b, uchar c, uchar d, uchar e, uchar f) const \
        { \
            return func ## f(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<schar> : senary_function<schar, schar, schar, schar, schar, schar, float> \
    { \
        __device__ __forceinline__ float operator ()(schar a, schar b, schar c, schar d, schar e, schar f) const \
        { \
            return func ## f(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<ushort> : senary_function<ushort, ushort, ushort, ushort, ushort, ushort, float> \
    { \
        __device__ __forceinline__ float operator ()(ushort a, ushort b, ushort c, ushort d, ushort e, ushort f) const \
        { \
            return func ## f(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<short> : senary_function<short, short, short, short, short, short, float> \
    { \
        __device__ __forceinline__ float operator ()(short a, short b, short c, short d, short e, short f) const \
        { \
            return func ## f(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<uint> : senary_function<uint, uint, uint, uint, uint, uint, float> \
    { \
        __device__ __forceinline__ float operator ()(uint a, uint b, uint c, uint d, uint e, uint f) const \
        { \
            return func ## f(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<int> : senary_function<int, int, int, int, int, int, float> \
    { \
        __device__ __forceinline__ float operator ()(int a, int b, int c, int d, int e, int f) const \
        { \
            return func ## f(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<float> : senary_function<float, float, float, float, float, float, float> \
    { \
        __device__ __forceinline__ float operator ()(float a, float b, float c, float d, float e, float f) const \
        { \
            return func ## f(a, b, c, d, e, f); \
        } \
    }; \
    template <> struct name ## _func<double> : senary_function<double, double, double, double, double, double, double> \
    { \
        __device__ __forceinline__ double operator ()(double a, double b,double c, double d, double e, double f) const \
        { \
            return func(a, b, c, d, e, f); \
        } \
    };


} // cudev


} // cv

#endif

#endif // CUDEV_FUNCTIONAL_EXT_HPP
