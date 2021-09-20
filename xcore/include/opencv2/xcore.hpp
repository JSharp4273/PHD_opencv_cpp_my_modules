#ifndef XCORE_HPP
#define XCORE_HPP

#pragma once

#include "opencv2/cvconfig.h"

#include "opencv2/core.hpp"
#ifdef HAVE_CUDA
#include "opencv2/core/cuda.hpp"
#endif

#include "opencv2/xcore/linalg.hpp"

#ifdef HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif



namespace cv
{

////////////////////////////////////////
/// More functions                   ///
////////////////////////////////////////

///
/// \brief getMinMax : return the minimum and maximum values of the given matrix.
/// \param _src : matrix to process
/// \return Vec2d where the first element is the minimum value and the last element is the maximum value.
///
///
Vec2d getMinMax(InputArray _src);

#if 0 // development in progress
///
/// \brief fma : fused multiply addition. return a x b + c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fma(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);
///
/// \brief fms : fused multiply substraction. return a x b - c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fms(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);
///
/// \brief fma : fused multiply addition. return -(a x b) + c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfma(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);
///
/// \brief fma : fused multiply addition. return -(a x b) - c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfms(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);


///
/// \brief fma : fused division addition. return a / b + c, by computing a x reciprocal(b) + c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fda(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);
///
/// \brief fds : fused division substraction. return a / b - c, by computing a x reciprocal(b) - c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fds(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);
///
/// \brief fda : fused division addition. return -(a x b) + c, by computing -(a x reciprocal(b) ) + c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfda(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);
///
/// \brief fda : fused division addition. return -(a x b) - c, by computing -(a x reciprocal(b) ) - c using the appropriate hardware optimisation if available.
/// \param _a : Matrix of any type, with any number of channels.
/// \param _b : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _c : Matrix of any type, with any number of channels. Must have the same size as _a.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfds(InputArray _a, InputArray _b, InputArray _c, OutputArray _dst, int dtype=-1);
#endif


////////////////////////////////////////
/// More parallel loop options       ///
////////////////////////////////////////

///
/// \brief The ParallelLoopBodyInt class
///
/// Allows to write to code for only one element of a range.
///

class ParallelLoopBodyInt : public virtual ParallelLoopBody
{
public:

    ParallelLoopBodyInt() = default;
    virtual ~ParallelLoopBodyInt() = default;

    virtual void operator()(const Range& range)const CV_OVERRIDE
    {
        for(int r=range.start; r<range.end; r++)
            this->operator()(r);

    }

    virtual void operator()(const int& idx)const = 0;
};

///
/// \brief The ParallelLoopBodyLambdaWrapperInt class
///
/// Same as previously but for addapted for lambda functions
///
class ParallelLoopBodyLambdaWrapperInt : public virtual ParallelLoopBodyInt
{

public:

    inline ParallelLoopBodyLambdaWrapperInt(const std::function<void(const int&)>& _fun):
        fun(_fun)
    {}

    virtual ~ParallelLoopBodyLambdaWrapperInt() = default;

    virtual void operator()(const int& index)const
    {
        this->fun(index);
    }

private:

    const std::function<void(const int&)>& fun;

};

///
/// \brief parallel_for_ : overloads of the parallel_for_ function to works with the new classes.
/// \param range, (or start, end) : range of values, which will be splited into chuncks.
/// \param functor : functor or function to apply either on each chuncks, or on each element.
/// \param nstripes : number of chunks to splits the range into. i.e. a range between [0,80],
///                   with a stripes of 8, means that the data are splitted in 8 chuncks.
///                   It does not set the number of thread to use.
void parallel_for_(const Range& range, std::function<void(const int&)> functor, double nstripes=-1.);
void parallel_for_(const int& start, const int& end, const ParallelLoopBody& fun);
void parallel_for_(const int& start, const int& end, const std::function<void(const Range&)>& fun);
void parallel_for_(const int& start, const int& end, const std::function<void(const int&)>& fun);

///
/// \brief highPrioriyParallelFor : when compiled with TBB, act same as parallel_for_ but with high_priority threads.
/// \param range : range of values, which will be splited into chuncks.
/// \param fun : functor or function to apply either on each chuncks, or on each element.
///
void highPrioriyParallelFor(const Range& range, const ParallelLoopBody& fun);
void highPrioriyParallelFor(const Range& range, const std::function<void(const Range&)>& fun);
void highPrioriyParallelFor(const Range& range, const std::function<void(const int&)>& fun);


#ifdef HAVE_TBB
template<class Body>
inline void highPrioriyParallelFor(const tbb::blocked_range<int>& range, const Body& body)
{
    tbb::task_group_context tgc;

    tgc.set_priority(tbb::priority_high);

    tbb::parallel_for(range,body);
}

tbb::blocked_range<int> convertRange(const Range& range);

template<class Body>
inline void highPrioriyParallelFor(const int& start, const int& end, const Body& body)
{
    tbb::task_group_context tgc;

    tgc.set_priority(tbb::priority_high);

    tbb::parallel_for(start, end, body);
}
#endif


///
/// \brief ceil : Rounds floating-point number to the nearest integer not smaller than the original, element-wisely.
/// \param _src : matrix of data to process, can have any type of number of channel, but must be 2D
/// \param _dst : matrix of the same size as the _src matrix.
/// \param dtype : optional depth of the output array
///
CV_EXPORTS_W void ceil(InputArray _src, OutputArray _dst, int dtype=-1);

///
/// \brief floor : Rounds floating-point number to the nearest integer not larger than the original.
/// \param _src : matrix of data to process, can have any type of number of channel, but must be 2D
/// \param _dst : matrix of the same size as the _src matrix.
/// \param dtype : optional depth of the output array
///
CV_EXPORTS_W void floor(InputArray _src, OutputArray _dst, int dtype=-1);

///
/// \brief round : Rounds floating-point number to the nearest integer, element-wisely.
/// \param _src : matrix of data to process, can have any type of number of channel, but must be 2D
/// \param _dst : matrix of the same size as the _src matrix.
/// \param dtype : optional depth of the output array
///
CV_EXPORTS_W void round(InputArray _src, OutputArray _dst, int dtype=-1);

///
/// \brief kron : compute the Kronecker product of two arrays.
/// \param _src1 : first input array, can have any depth but must have a single channel.
/// \param _src2 : must have the same, must have the same depth as _src1, with a single channel.
/// \param _dst : matrix of size [_src1.rows x _src2.rows, _src1.cols x _src2.cols].
///
CV_EXPORTS_W void kron(InputArray _src1, InputArray _src2, OutputArray _dst);

} // cv

#endif // XCORE_HPP
