#ifndef XCORE_TRAITS_HPP
#define XCORE_TRAITS_HPP

#include "opencv2/core.hpp"

namespace cv
{

template<class T>
struct type2flag
{
    enum{flag=CV_MAKETYPE(DataType<typename DataType<T>::channel_type>::depth, DataType<T>::channels)};
};

template<int depth>
struct flag2FundamentalType;

template<>
struct flag2FundamentalType<CV_8U>{ typedef uchar value_type; };

template<>
struct flag2FundamentalType<CV_8S>{ typedef schar value_type; };

template<>
struct flag2FundamentalType<CV_16U>{ typedef ushort value_type; };

template<>
struct flag2FundamentalType<CV_16S>{ typedef short value_type; };

template<>
struct flag2FundamentalType<CV_32S>{ typedef int value_type; };

template<>
struct flag2FundamentalType<CV_32F>{ typedef float value_type; };

template<>
struct flag2FundamentalType<CV_64F>{ typedef double value_type; };



template<int type, int depth=CV_MAT_DEPTH(type), int cn=CV_MAT_CN(type)>
struct flag2type
{
    typedef Vec<typename flag2FundamentalType<depth>::value_type, cn> value_type;
};

template<int type, int depth>
struct flag2type<type, depth, 1>
{
    typedef typename flag2FundamentalType<depth>::value_type value_type;
};



} // cv

#endif // XCORE_TRAITS_HPP
