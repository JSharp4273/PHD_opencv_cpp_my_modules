#include "opencv2/cudaxcore.hpp"
#include "../precomp.hpp"

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/cudaxcore/utils.hpp"

#include <curand.h>


#define GENERATE_UNIFORM 0
#define GENERATE_NORMAL 1
#define GENERATE_LOG_NORMAL 2
#if 0
#define GENERATE_POISSON 3
#endif

namespace cv
{

namespace cuda
{

namespace
{
#if 0
template<class T>
__global__ void cvtUInt32To_kernel(const PtrStep<unsigned> src, PtrStepSz<T> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    dst(y,x) = device::saturate_cast<T>(src(y, x));
}

template<class T>
void cvtUInt32To(const unsigned* src_ptr, const size_t& src_step, T* dst_ptr, const size_t& dst_step, const int& rows, const int& cols, cudaStream_t stream)
{
    dim3 block(32,8);
    dim3 grid (divUp (cols, block.x), divUp (rows, block.y));

    cudaSafeCall( cudaFuncSetCacheConfig (cvtUInt32To_kernel<T>, cudaFuncCachePreferL1) );
    cvtUInt32To_kernel<<<block, grid, 0, stream>>>(PtrStep<unsigned>(const_cast<unsigned*>(src_ptr), src_step), PtrStepSz<T>(rows, cols, dst_ptr, src_step));
    cudaSafeCall ( cudaGetLastError () );

    if (stream == 0)
         cudaSafeCall( cudaDeviceSynchronize() );
}
#endif

template<int d, class T>
void generate(curandGenerator_t&, GpuMat&, const Scalar&, const Scalar&, const size_t&, const uint64&, Stream&);

template<>
void generate<GENERATE_UNIFORM, float>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _min, const Scalar& _max, const size_t& size, const uint64& state, Stream& _stream)
{    
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));

    curandSafeCall(curandGenerateUniform(generator, _src.ptr<float>(), size));

    multiply(_src, _max-_min, _src, 1., CV_32F, _stream);
    add(_src, _min, _src, noArray(), CV_32F, _stream);
}

template<>
void generate<GENERATE_UNIFORM, double>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _min, const Scalar& _max, const size_t& size, const uint64& state, Stream& _stream)
{
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));

    curandSafeCall(curandGenerateUniformDouble(generator, _src.ptr<double>(), size));

    multiply(_src, _max-_min, _src, 1., CV_64F, _stream);
    add(_src, _min, _src, noArray(), CV_64F, _stream);
}


template<>
void generate<GENERATE_NORMAL, float>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _mean, const Scalar& _std, const size_t& size, const uint64& state, Stream& _stream)
{
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));
    curandSafeCall(curandSetStream(generator, StreamAccessor::getStream(_stream)));

    float mean(0.f), stddev(1.f);

    if(_src.channels()==1)
    {
        mean = static_cast<float>(_mean(0));
        stddev = static_cast<float>(_std(0));
    }

    curandSafeCall(curandGenerateNormal(generator, _src.ptr<float>(), size, mean, stddev) );

    if(_src.channels()>1)
    {
        multiply(_src, _std, _src, 1., CV_32F, _stream);
        add(_src, _mean, _src, noArray(), CV_32F, _stream);
    }
}

template<>
void generate<GENERATE_NORMAL, double>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _mean, const Scalar& _std, const size_t& size, const uint64& state, Stream& _stream)
{
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));
    curandSafeCall(curandSetStream(generator, StreamAccessor::getStream(_stream)));

    double mean(0.), stddev(1.);

    if(_src.channels()==1)
    {
        mean = _mean(0);
        stddev = _std(0);
    }

    curandSafeCall(curandGenerateNormalDouble(generator, _src.ptr<double>(), size, mean, stddev) );

    if(_src.channels()>1)
    {
        multiply(_src, _std, _src, 1., CV_64F, _stream);
        add(_src, _mean, _src, noArray(), CV_64F, _stream);
    }
}


template<>
void generate<GENERATE_LOG_NORMAL, float>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _mean, const Scalar& _std, const size_t& size, const uint64& state, Stream& _stream)
{
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));
    curandSafeCall(curandSetStream(generator, StreamAccessor::getStream(_stream)));

    float mean(0.f), stddev(1.f);

    if(_src.channels()==1)
    {
        mean = static_cast<float>(_mean(0));
        stddev = static_cast<float>(_std(0));
    }

    curandSafeCall(curandGenerateLogNormal(generator, _src.ptr<float>(), size, mean, stddev) );

    if(_src.channels()>1)
    {
        multiply(_src, _std, _src, 1., CV_32F, _stream);
        add(_src, _mean, _src, noArray(), CV_32F, _stream);
    }
}

template<>
void generate<GENERATE_LOG_NORMAL, double>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _mean, const Scalar& _std, const size_t& size, const uint64& state, Stream& _stream)
{
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));
    curandSafeCall(curandSetStream(generator, StreamAccessor::getStream(_stream)));

    double mean(0.), stddev(1.);

    if(_src.channels()==1)
    {
        mean = _mean(0);
        stddev = _std(0);
    }

    curandSafeCall(curandGenerateLogNormalDouble(generator, _src.ptr<double>(), size, mean, stddev) );

    if(_src.channels()>1)
    {
        multiply(_src, _std, _src, 1., CV_64F, _stream);
        add(_src, _mean, _src, noArray(), CV_64F, _stream);
    }
}

#if 0
template<>
void generate<GENERATE_POISSON, float>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _lambda, const Scalar& , const size_t& size, const uint64& state, Stream& _stream)
{
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));
    curandSafeCall(curandSetStream(generator, StreamAccessor::getStream(_stream)));


    unsigned* d_tmp(nullptr);

    // 1) number of elements to allocate in bytes.
    size_t nb_elems = sizeof(float) == sizeof(unsigned) ? size : (size / sizeof(float)) * sizeof(unsigned);

    cudaMalloc(&d_tmp, nb_elems);

    // 2) number of elements to generate.
    nb_elems /= sizeof(unsigned);


    curandSafeCall(curandGeneratePoisson(generator, d_tmp, nb_elems, _lambda(0) ) );

    cvtUInt32To<float>(d_tmp, static_cast<size_t>(_src.cols * sizeof(unsigned)), _src.ptr<float>(), _src.step, _src.rows, _src.cols, StreamAccessor::getStream(_stream));

    cudaFree(d_tmp);

}

template<>
void generate<GENERATE_POISSON, double>(curandGenerator_t& generator, GpuMat& _src, const Scalar& _lambda, const Scalar& , const size_t& size, const uint64& state, Stream& _stream)
{
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, state));
    curandSafeCall(curandSetStream(generator, StreamAccessor::getStream(_stream)));

        unsigned* d_tmp(nullptr);

        // 1) number of elements to allocate in bytes.
        size_t nb_elems = (size / sizeof(double)) * sizeof(unsigned);

        cudaMalloc(&d_tmp, nb_elems);

        // 2) number of elements to generate.
        nb_elems /= sizeof(unsigned);

        curandSafeCall(curandGeneratePoisson(generator, d_tmp, nb_elems, _lambda(0) ) );

        cvtUInt32To<double>(d_tmp, _src.cols * sizeof(unsigned), _src.ptr<double>(), _src.step, _src.rows, _src.cols, StreamAccessor::getStream(_stream));

        cudaFree(d_tmp);
}
#endif


} // anonymous


void RNGAsync::fill(InputOutputArray _src, const int& distType, InputArray _a, InputArray _b, Stream& _stream)
{

    typedef void (*function_type)(curandGenerator_t&, GpuMat&, const Scalar&, const Scalar&, const size_t&, const uint64&, Stream&);

    static const function_type funcs[4][2] = {{generate<GENERATE_UNIFORM,   float>, generate<GENERATE_UNIFORM,    double>},
                                             {generate<GENERATE_NORMAL,     float>, generate<GENERATE_NORMAL,     double>},
                                             {generate<GENERATE_LOG_NORMAL, float>, generate<GENERATE_LOG_NORMAL, double>},
#if 0
                                             {generate<GENERATE_POISSON,    float>, generate<GENERATE_POISSON,    double>}
#endif
                                             };

    CV_Assert(_src.isGpuMat() && !_src.empty());

    GpuMat src = _src.depth() >= CV_32F ? _src.getGpuMat() : GpuMat(_src.size(), CV_32FC(_src.channels()));

    size_t total_amount_of_memory_to_process = src.rows * (src.step/src.elemSize1());

    const int sdepth = src.depth();
    const int wdepth = std::max(sdepth, CV_32F);

    if(sdepth != wdepth)
    {
        GpuMat tmp = src;
        src.release();
        tmp.convertTo(src, wdepth);
    }


    curandGenerator_t generator;

    switch (this->generator_id)
    {
    case PSEUDO_DEFAULT:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        break;

    case PSEUDO_XORWOW:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
        break;

    case PSEUDO_MRG32K3A:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MRG32K3A));
        break;

    case PSEUDO_MTGP32:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
        break;

    case PSEUDO_PHILOX4_32_10:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        break;

    case QUASI_DEFAULT:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_QUASI_DEFAULT));
        break;

    case QUASI_SOBOL32:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SOBOL32));
        break;

    case QUASI_SCRAMBLED_SOBOL32:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
        break;

    case QUASI_SOBOL64:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SOBOL64));
        break;

    case QUASI_SCRAMBLED_SOBOL64:
        curandSafeCall(curandCreateGenerator(&generator, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64));
        break;

    default:
        CV_Error(Error::StsBadArg, "The type of generator specified, is either wrong, or not supported yet.");
        break;
    }

//    curandSafeCall(curandCreateGenerator(&generator, static_cast<curandRngType>(this->generator_id)));
    curandSafeCall(curandSetPseudoRandomGeneratorSeed(generator, this->state));


    function_type fun = funcs[distType][src.depth()-CV_32F];

    fun(generator, src, getScalar(_a),  getScalar(_b), total_amount_of_memory_to_process, this->state, _stream);


    curandDestroyGenerator(generator);

    if(sdepth != wdepth)
        src.convertTo(_src, _src.depth());
    else
        src.copyTo(_src);
}



} // cuda

} // cv
