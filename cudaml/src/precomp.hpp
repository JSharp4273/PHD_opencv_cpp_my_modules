#ifndef CUDAML_PRECOMP_HPP
#define CUDAML_PRECOMP_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/hdf.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/cvconfig.h"

#include "opencv2/cudaxcore/linalg.hpp"
#include "opencv2/cudaxcore.hpp"

#include "opencv2/xcore/template/arguments_io.hpp"

//#include "linalg.hpp"

//#ifdef _DEBUG
//#include <iostream>
//using namespace std;
//#endif

//namespace af
//{

//class array;

//} // af

namespace cv
{

namespace cuda
{

/////
///// \brief RowWise2ColWise : copy data from a row-major management, to a column-major one.
///// \param src_data : source in memory
///// \param src_step : step of the data source in bytes (i.e. cols * sizeof(element) or (i.e. cols * channels * sizeof(element) )
///// \param dst_data : destination in memory
///// \param dst_step : step of the data destination in bytes (i.e. rows * sizeof(element) or (i.e. rows * channels * sizeof(element) )
///// \param rows : number of rows
///// \param cols : number of column
///// \param type : opencv type flag
///// \param stream : Stream of the asynchronous version.
/////
//void RowWise2ColWise(const void* src_data, size_t src_step, void* dst_data, size_t dst_step, int rows, int cols, int type, Stream& stream = Stream::Null());

/////
///// \brief ColWise2RowWise : copy data from a column-major management, to a row-major one.
///// \param src_data : source in memory
///// \param src_step : step of the data source in bytes (i.e. rows * sizeof(element) or (i.e. rows * channels * sizeof(element) )
///// \param dst_data : destination in memory
///// \param dst_step : step of the data destination in bytes (i.e. cols * sizeof(element) or (i.e. cols * channels * sizeof(element) )
///// \param rows : number of rows
///// \param cols : number of column
///// \param type : opencv type flag
///// \param stream : Stream of the asynchronous version.
/////
//void ColWise2RowWise(const void* src_data, size_t src_step, void* dst_data, size_t dst_step, int rows, int cols, int type, Stream& stream = Stream::Null());

//namespace arrayfire
//{

/////
///// \brief GpuMat2Array : copy an OpenCV's GpuMat to an ArrayFire array
///// \param _src : data to copy
///// \param _dst : data copied
///// \param stream : Stream of the asynchronous version.
/////
//void GpuMat2Array(const GpuMat& _src, af::array& _dst, Stream& stream = Stream::Null());

/////
///// \brief Array2GpuMat : copy an ArrayFire array to an OpenCV's GpuMat
///// \param _src : data to copy
///// \param _dst : data copied
///// \param stream : Stream of the asynchronous version.
/////
//void Array2GpuMat(const af::array& _src, GpuMat& _dst, Stream& stream = Stream::Null());

//} // arrayfire





#define IMPL_VOID_ACCESSOR(name, variable) \
    virtual void get##name(OutputArray _dst, Stream& stream) const CV_OVERRIDE\
{\
    CV_Assert(_dst.isMat() || _dst.isUMat() || _dst.isGpuMat());\
    \
    if(_dst.isGpuMat())\
        variable.copyTo(_dst, stream);\
    else \
        variable.download(_dst);\
} \
    virtual void set##name(InputArray _dst, Stream& ) CV_OVERRIDE\
{\
    CV_Assert(_dst.isMat() || _dst.isUMat() || _dst.isGpuMat() || _dst.kind() == _InputArray::EXPR );\
    \
    if(_dst.isGpuMat())\
    variable = _dst.getGpuMat();\
    else\
    variable.upload(_dst);\
}

///
/// \brief get_scalar : Take a GpuMat with size 1 x 1 and return the scalar element
/// \param m
/// \return
///
inline double get_scalar(const GpuMat& m){if(m.empty()) return 0.;  Mat mm(m); mm.convertTo(mm, CV_64F); return mm.at<double>(0);}


///
/// \brief centreToTheMeanAxis0 : subtract mu_X vector to every rows of X
/// \param X : Matrix M x N
/// \param mu_X : Vector 1 x N
/// \param dst : Matrix M x N
/// \param dtype : optional depth of the output array
/// \param stream : Stream of the asynchronous version.
///
void centreToTheMeanAxis0(const GpuMat& X, const GpuMat& mu_X, GpuMat& dst, const int& dtype=-1, Stream& stream = Stream::Null());



}// cuda

}// cv

#endif // PRECOMP_HPP
