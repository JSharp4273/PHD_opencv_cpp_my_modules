#include "opencv2/cudaxcore.hpp"
#include "opencv2/cudaxcore/utils.hpp"
#include "precomp.hpp"


#ifndef HAVE_CURAND
#error "curand is required"
#else


namespace cv
{

namespace cuda
{

namespace
{

} // anonymous

RNGAsync::RNGAsync():
    state(0xffffffff),
    generator_id(PSEUDO_DEFAULT)
{}

RNGAsync::RNGAsync(generator_id_t _generator):
    state(0xffffffff),
    generator_id(_generator)
{}

RNGAsync::RNGAsync(const uint64& _state):
    state(_state ? 0xffffffff : _state),
    generator_id(PSEUDO_DEFAULT)
{}

RNGAsync::RNGAsync(const uint64& _state, generator_id_t _generator):
    state(_state ? 0xffffffff : _state),
    generator_id(_generator)
{}

namespace
{

RNGAsync _rng;

} // anonymous

RNGAsync& theRNGAsync(){ return _rng;}

void setRNGAsyncSeed(int seed)
{
    _rng = RNGAsync(seed);
}





Scalar mean(InputArray _src)
{
    double total = static_cast<double>(_src.total());

    Scalar tmp = cuda::sum(_src);

    for(double& ti : tmp.val)
        ti/=total;

    return tmp;
}

} // cuda

} // cv

#endif
