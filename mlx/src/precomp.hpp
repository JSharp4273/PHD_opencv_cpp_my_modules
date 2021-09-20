#ifndef MLXPRECOMP_HPP
#define MLX_PRECOMP_HPP

#pragma once

#include <numeric>
#include <locale>
#include <functional>

#include "opencv2/core.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/hdf.hpp"
#include "opencv2/cvconfig.h"

#include "opencv2/xcore.hpp"
#include "opencv2/xcore/template/arguments_io.hpp"
#include "opencv2/xcore/template/hal.hpp"
#include "opencv2/xcore/template/intrin.hpp"


namespace cv
{

template<class T>
void centreToTheMeanAxis0(const T& X, const T& mu_X, T& dst, const int& dtype=-1);

template<>
void centreToTheMeanAxis0<Mat>(const Mat& X, const Mat& mu_X, Mat& dst, const int& dtype);

template<>
void centreToTheMeanAxis0<UMat>(const UMat& X, const UMat& mu_X, UMat& dst, const int& dtype);



template<class T>
void centreToTheMeanAxis0(const T& X, T& dst, const int& dtype=-1);

template<>
void centreToTheMeanAxis0<Mat>(const Mat& X, Mat& dst, const int& dtype);

template<>
void centreToTheMeanAxis0<UMat>(const UMat& X, UMat& dst, const int& dtype);


template <class T>
struct h5_ds_t
{
    T* ds;
    String ds_name;
};

typedef h5_ds_t<Mat> read_h5_ds_t;
typedef h5_ds_t<const Mat> write_h5_ds_t;


} //cv

#endif // PRECOMP_HPP
