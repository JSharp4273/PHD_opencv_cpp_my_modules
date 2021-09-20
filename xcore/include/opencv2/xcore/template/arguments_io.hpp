#ifndef XCORE_ARGUMENTS_IO_HPP
#define XCORE_ARGUMENTS_IO_HPP

#include "opencv2/core.hpp"
#include "opencv2/cvconfig.h"

namespace cv
{

////////////////////////////////////////
/// Input Output function            ///
////////////////////////////////////////


bool isMatrix(InputArray& _src);
bool isVectorOfMatrix(InputArray& _src);
bool isScalar(InputArray& _src);

bool isMatrix(InputOutputArray& _src);
bool isVectorOfMatrix(InputOutputArray& _src);
bool isScalar(InputOutputArray& _src);

bool isMatrix(OutputArray& _src);
bool isVectorOfMatrix(OutputArray& _src);
bool isScalar(OutputArray& _src);


template<class T>
T getInput(InputArray& _src, const bool& copy=true);

template<class T>
T getInput(InputOutputArray& _src, const bool& copy=true);

template<class T>
void getInput(InputArray& _src, T& _dst, const bool& copy=true);

template<class T>
void getInput(InputOutputArray& _src, T& _dst, const bool& copy=true);



#define DECL_SPEC_GETINPUT(type)\
    template<>\
    type getInput<type>(InputArray& _src, const bool& copy); \
    \
    template<>\
    type getInput<type>(InputOutputArray& _src, const bool& copy); \
    \
    template<>\
    void getInput<type>(InputArray& _src, type& _dst, const bool& copy);\
    \
    template<>\
    void getInput<type>(InputOutputArray& _src, type& _dst, const bool& copy);

DECL_SPEC_GETINPUT(Mat)
DECL_SPEC_GETINPUT(UMat)
#ifdef HAVE_CUDA
DECL_SPEC_GETINPUT(cuda::GpuMat)
#endif
DECL_SPEC_GETINPUT(std::vector<Mat>)
DECL_SPEC_GETINPUT(std::vector<UMat>)
#ifdef HAVE_CUDA
DECL_SPEC_GETINPUT(std::vector<cuda::GpuMat>)
#endif

#undef DECL_SPEC_GETINPUT

template<class T>
void setOutput(OutputArray& _dst, const T& arg);

template<class T>
void setOutput(InputOutputArray& _dst, const T& arg);

#define DECL_SPEC_SETOUTPUT(type)\
    template<>\
    void setOutput<type>(OutputArray& _dst, const type& arg);\
    \
    template<>\
    void setOutput<type>(InputOutputArray& _dst, const type& arg);

DECL_SPEC_SETOUTPUT(Mat)
DECL_SPEC_SETOUTPUT(UMat)
#ifdef HAVE_CUDA
DECL_SPEC_SETOUTPUT(cuda::GpuMat)
#endif
DECL_SPEC_SETOUTPUT(std::vector<Mat>)
DECL_SPEC_SETOUTPUT(std::vector<UMat>)
#ifdef HAVE_CUDA
DECL_SPEC_SETOUTPUT(std::vector<cuda::GpuMat>)
#endif

#undef DECL_SPEC_GETINPUT


Scalar getScalar(InputArray& _src);
Scalar getScalar(InputOutputArray& _src);

Scalar fromAnythingToScalar(InputArray _src);


Mat toMat(InputArray _src);
UMat toUMat(InputArray _src);
#ifdef HAVE_CUDA
cuda::GpuMat toGpuMat(InputArray _src);
#endif
std::vector<Mat> toMatVector(InputArray _src);
std::vector<UMat> toUMatVector(InputArray _src);
#ifdef HAVE_CUDA
std::vector<cuda::GpuMat> toGpuMatVector(InputArray _src);
#endif


void toMat(InputArray _src, Mat& ret);
void toUMat(InputArray _src, UMat& ret);
#ifdef HAVE_CUDA
void toGpuMat(InputArray _src, cuda::GpuMat& ret);
#endif
void toMatVector(InputArray _src, std::vector<Mat>& ret);
void toUMatVector(InputArray _src, std::vector<UMat>& ret);
#ifdef HAVE_CUDA
void toGpuMatVector(InputArray _src, std::vector<cuda::GpuMat>& ret);
#endif

}// cv

#endif // XCORE_ARGUMENTS_IO_HPP
