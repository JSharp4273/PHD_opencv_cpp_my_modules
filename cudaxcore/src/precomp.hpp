#ifndef CUDAXCORE_PRECOMP_HPP
#define CUDAXCORE_PRECOMP_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"

#include "opencv2/xcore/template/arguments_io.hpp"

#include "opencv2/cvconfig.h"

#include <type_traits>

namespace cv
{

namespace cuda
{

template <class T>
struct isVectorType : std::true_type{};

template<>
struct isVectorType<uchar> : std::false_type{};

template<>
struct isVectorType<schar> : std::false_type{};

template<>
struct isVectorType<ushort> : std::false_type{};

template<>
struct isVectorType<short> : std::false_type{};

template<>
struct isVectorType<int> : std::false_type{};

template<>
struct isVectorType<unsigned> : std::false_type{};

template<>
struct isVectorType<float> : std::false_type{};

template<>
struct isVectorType<double> : std::false_type{};

} // cuda

} // cv

#endif // CUDAXCORE_PRECOMP_HPP
