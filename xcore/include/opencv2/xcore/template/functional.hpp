#ifndef XCORE_FUNCTIONAL_HPP
#define XCORE_FUNCTIONAL_HPP

#include "opencv2/core.hpp"
#include "opencv2/cvconfig.h"

namespace cv
{
///
/// \brief check_UMat_output_ : try to identify if one or more OutputArray argument is a UMat
/// \return
///
inline bool check_UMat_output_(){ return false;}

template<class T, class... Args>
inline bool check_UMat_output_(T _arg, Args... args){ return false | check_UMat_output_(args...);}

template<class... Args>
inline bool check_UMat_output_(InputArray _arg, Args... args){ return false | check_UMat_output_(args...);}

template<class... Args>
inline bool check_UMat_output_(OutputArray _arg, Args... args){ return _arg.isUMat() | check_UMat_output_(args...);}

template<class... Args>
inline bool check_UMat_output_(InputOutputArray _arg, Args... args){ return _arg.isUMat() | check_UMat_output_(args...);}

inline bool check_UMat_output_(InputArray ){ return false;}
inline bool check_UMat_output_(OutputArray _arg){ return _arg.isUMat();}
inline bool check_UMat_output_(InputOutputArray _arg){ return _arg.isUMat();}
template<class... Args>
inline bool check_UMat_output(Args... args){ return check_UMat_output_(args...);}

///
/// \brief ocl_func : a functor is given as static template argument of this function.
/// The function must take a template argument which represents, the container type (Mat, UMat,...) of the data.
/// The list of argument for the functor is provided as argument of this function.
/// If one of the output argument is a UMat then the function call the given functor with a UMat as type argmument
/// If the function object execute whithout sending any exception the function return true.
/// If there is not UMat among the list of the Output argument or if during the exectution an exception is raised
/// then the function return false.
///
template<template<class>class type,class ... Args>
inline bool ocl_func(Args&... _Args)
{
#ifdef HAVE_OPENCL
 if(!check_UMat_output(_Args...))
     return false;
 bool ret(true);
 try
 {
     type<cv::UMat> func;

    func(_Args...);
 }
 catch(...)
 {
     ret = false;
 }
 return ret;
#else
    return false;
#endif
}

///
/// \brief func : a functor is given as static template argument of this function.
/// The function must take a template argument which represents, the container type (Mat, UMat,...) of the data.
/// The list of argument for the functor is provided as argument of this function.
/// This function will call an instance of the provided functor or a Mat argument.
///
template<template<class>class type,class ... Args>
inline void func_no_cl(Args&... args)
{
    type<cv::Mat> func;
    func(args...);
}


///
/// \brief func : a functor is given as static template argument of this function.
/// The function must take a template argument which represents, the container type (Mat, UMat,...) of the data.
/// The list of argument for the functor is provided as argument of this function.
/// If OpenCV was compiled with OpenCL and an output argument is a UMat then an attempt to use the provided function
/// with UMat argument will be made.
/// If this attempt fail the function will call an instance of the given functor for a Mat argument.
///
template<template<class>class type,class ... Args>
inline void func(Args&... args)
{
#ifdef HAVE_OPENCL
    if(ocl_func<type>(args...))
        return;
#endif
    func_no_cl<type>(args...); //no cl ... or failed cl
}

/*
 *
 * template<class T>
 * struct fake_fma_impl
 * {
 * void operator()(InputArray& _a, InputArray& _b, InputArray& _c, OutputArray& _dst) const
 * {
 * T tmp;
 * multiply(_a, _b, tmp);
 * add(tmp, _c, _dst);
 * }
 * };
 *
 * void fake_fma(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst)
 * {
 * func<fake_fma_impl>(_a, _b, _c, _dst);
 * }
 *
 */

// The goal of these functions is to try to use the TAPI as efficiently as possible, while simplifying the development of algorithm.
// The functor approach allows to make specialisation when needed.



} // cv

#endif // XCORE_FUNCTIONAL_HPP
