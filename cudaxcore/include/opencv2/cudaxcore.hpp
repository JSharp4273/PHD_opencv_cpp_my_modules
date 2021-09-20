#ifndef CUDAXCORE_HPP
#define CUDAXCORE_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaxcore/linalg.hpp"

namespace cv
{

namespace cuda
{
///
/// \brief The RNGAsync class : Random number generator.
///
/// It encapsulate both the state as well as the generator to use.
/// It has only method to fill matrix.
/// It is able to manage, uniform, normal and log-normal distributions.
///
class RNGAsync
{
public:

    enum {
        UNIFORM    = 0,
        NORMAL     = 1,        
        LOG_NORMAL = 2,
#if 0        
        POISSON    = 3
#endif
    };

    enum generator_id_t
    {
        PSEUDO_DEFAULT,
        PSEUDO_XORWOW,
        PSEUDO_MRG32K3A,
        PSEUDO_MTGP32,
        PSEUDO_MT19937,
        PSEUDO_PHILOX4_32_10,
        QUASI_DEFAULT,
        QUASI_SOBOL32,
        QUASI_SCRAMBLED_SOBOL32,
        QUASI_SOBOL64,
        QUASI_SCRAMBLED_SOBOL64
    };

    ///
    /// \brief RNGAsync constructors
    ///
    RNGAsync();
    /// \overload
    /// \param _generator : random number generator to use.
    RNGAsync(generator_id_t _generator);
    /// \overload
    /// \param state 64-bit value used to initialize the RNGAsync.
    RNGAsync(const uint64& _state);
    /// \overload
    /// \param state 64-bit value used to initialize the RNGAsync.
    /// \param _generator : random number generator to use.
    RNGAsync(const uint64& _state, generator_id_t _generator);

    virtual ~RNGAsync() = default;

    ///
    /// \brief fill : fill a matrix with random number.
    /// Currently all the generator generate floating point number.
    /// If the input matrix has interger type the floating numbers will be cast to the type.
    ///
    /// \param _src : Matrix of any tpye of number of channels.
    /// \param distType : distribution type. Can be RNGAsync::UNIFORM, RNGAsync::NORMAL, RNGAsync::LOG_NORMAL
    /// \param _a : first distribution parameter; in case of the uniform
    /// distribution, this is an inclusive lower boundary, in case of the normal or log_normal
    /// distribution, this is a mean value.
    /// \param _b : second distribution parameter; in case of the uniform
    /// distribution, this is a non-inclusive upper boundary, in case of the
    /// normal distribution, this is a standard deviation (diagonal of the
    /// standard deviation matrix or the full standard deviation matrix).
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP void fill(InputOutputArray _src, const int& distType, InputArray _a, InputArray _b, Stream& _stream = Stream::Null());

    uint64 state;
    generator_id_t generator_id;
};


/** @brief Returns the default random number generator.

The function cv::theRNGAsync returns the default random number generator. For each thread, there is a
separate random number generator, so you can use the function safely in multi-thread environments.
If you just need to get a single random number using this generator or initialize an array, you can
use randu or randn instead. But if you are going to generate many random numbers inside a loop, it
is much faster to use this function to retrieve the generator and then use RNG::operator _Tp() .
@sa RNG, randu, randn
*/
CV_EXPORTS RNGAsync& theRNGAsync();

/** @brief Sets state of default random number generator.

The function cv::setRNGAsyncSeed sets state of default random number generator to custom value.
@param seed new state for default random number generator
@sa RNG, randu, randn
*/
CV_EXPORTS_W void setRNGAsyncSeed(int seed);

///
/// \brief clip : clip the values within the specified range.
///               Every value lower than the _lower_threshold will be set to the _lower_threshold value.
///               Every value higher than the _higher_threshold will be set to the _higher_threshold value.
/// \param _src : Matrix to process. Can have any type but no more than four channels.
/// \param _lower_threshold : lower value to set
/// \param _higher_threshold : lower higher to set
/// \param _dst : destination Matrix.
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void clip(InputArray _src, InputArray _lower_threshold, InputArray _higher_threshold, OutputArray _dst, Stream& stream = Stream::Null());

///
/// \brief mean : return the mean of the given argument. Note that this function does not use cv::mean.
/// \param _src : Matrix to process. Can have any depth except CV_64F.
/// \return
///
CV_EXPORTS_W Scalar mean(InputArray _src);

///
/// \brief ceil : Rounds floating-point number to the nearest integer not smaller than the original, element-wisely.
/// \param _src : matrix of data to process, can have any type of number of channel, but must be 2D
/// \param _dst : matrix of the same size as the _src matrix.
/// \param dtype : optional depth of the output array
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void ceil(InputArray _src, OutputArray _dst, int dtype=-1, Stream& stream = Stream::Null());

///
/// \brief floor : Rounds floating-point number to the nearest integer not larger than the original.
/// \param _src : matrix of data to process, can have any type of number of channel, but must be 2D
/// \param _dst : matrix of the same size as the _src matrix.
/// \param dtype : optional depth of the output array
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void floor(InputArray _src, OutputArray _dst, int dtype=-1, Stream& stream = Stream::Null());

///
/// \brief round : Rounds floating-point number to the nearest integer, element-wisely.
/// \param _src : matrix of data to process, can have any type of number of channel, but must be 2D
/// \param _dst : matrix of the same size as the _src matrix.
/// \param dtype : optional depth of the output array
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void round(InputArray _src, OutputArray _dst, int dtype=-1, Stream& stream = Stream::Null());

///
/// \brief kron : compute the Kronecker product of two arrays.
/// \param _src1 : first input array, can have any depth but must have a single channel.
/// \param _src2 : must have the same, must have the same depth as _src1, with a single channel.
/// \param _dst : matrix of size [_src1.rows x _src2.rows, _src1.cols x _src2.cols].
/// \param stream : Stream of the asynchronous version.
///
CV_EXPORTS_W void kron(InputArray _src1, InputArray _src2, OutputArray _dst, Stream& stream = Stream::Null());


#if 0 // Development In Progress

// for more informations regarding rounding please check: https://en.wikipedia.org/wiki/Rounding
enum
{
  ROUND_TOWARD_ZERO,
    ROUND_DOWN,
    ROUND_UP,
    ROUND_TO_NEAREST_EVEN
};

///
/// \brief reciprocal : compute the reciprocal of the source argument. i.e. 1 / _src.
///
/// \param _src : GpuMat any size, and up to 4 channels.
/// \param _dst : GpuMat same size as the input argument, and if dtype argument is not set, same type as _src. Type specify by dtype otherwise.
/// \param round_flag : rounding operation to compute on the data.
/// \param _mask : optional, mask of the argument to compute.
/// \param dtype : optional, type of output argument. If not specify output argument will have the same as the source argument.
/// \param stream : Stream of the asynchronous version.
///
void reciprocal(InputArray _src, OutputArray _dst, int round_flag=ROUND_TO_NEAREST_EVEN, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());

///
/// \brief reciprocal : compute the reciprocal of the source argument. i.e. 1 / sqrt(_src)
/// \param _src : GpuMat any size, and up to 4 channels.
/// \param _dst : GpuMat same size as the input argument, and if dtype argument is not set, same type as _src. Type specify by dtype otherwise.
/// \param _mask : optional, mask of the argument to compute.
/// \param dtype : optional, type of output argument. If not specify output argument will have the same as the source argument.
/// \param stream : Stream of the asynchronous version.
///
void reciprocal_sqrt(InputArray _src, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());


///
/// \brief fma : fused multiply addition. return a x b + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fma(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fms : fused multiply substraction. return a x b - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fms(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fma : fused multiply addition. return -(a x b) + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfma(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fma : fused multiply addition. return -(a x b) - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfms(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());


///
/// \brief fma : fused division addition. return a / b + c, by computing a x reciprocal(b) + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fda(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fds : fused division substraction. return a / b - c, by computing a x reciprocal(b) - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void fds(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fda : fused division addition. return -(a x b) + c, by computing -(a x reciprocal(b) ) + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfda(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fda : fused division addition. return -(a x b) - c, by computing -(a x reciprocal(b) ) - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void nfds(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());


//
//
//
//


///
/// \brief fma : fused multiply addition. return a x b + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wfma(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fms : fused multiply substraction. return a x b - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wfms(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fma : fused multiply addition. return -(a x b) + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wnfma(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fma : fused multiply addition. return -(a x b) - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wnfms(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());


///
/// \brief fma : fused division addition. return a / b + c, by computing a x reciprocal(b) + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wfda(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fds : fused division substraction. return a / b - c, by computing a x reciprocal(b) - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wfds(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fda : fused division addition. return -(a x b) + c, by computing -(a x reciprocal(b) ) + c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wnfda(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());
///
/// \brief fda : fused division addition. return -(a x b) - c, by computing -(a x reciprocal(b) ) - c using the appropriate hardware optimisation if available.
/// \param _src1 : Matrix of any type, with any number of channels.
/// \param _src2 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _src3 : Matrix of any type, with any number of channels. Must have the same size as _src1.
/// \param _dst : Matrix same size and number of channels as a.
/// \param dtype : depth of the output matrix when if at least one of the input matrix does not have the same type as the other.
///
void wnfds(InputArray _w1, InputArray _src1,InputArray _w2,  InputArray _src2,InputArray _w3,  InputArray _src3, OutputArray _dst, InputArray _mask = noArray(), int dtype=-1, Stream& stream = Stream::Null());

#endif

} // cuda

} // cv

#endif // CUDAXCORE_HPP
