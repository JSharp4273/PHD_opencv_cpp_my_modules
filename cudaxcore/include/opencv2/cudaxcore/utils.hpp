#ifndef CUDAXCORE_UTILS_HPP
#define CUDAXCORE_UTILS_HPP

#include <curand.h>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda/common.hpp"

const char *curandGetErrorString(curandStatus_t error);

namespace cv { namespace cuda {
    static inline void checkCurandError(curandStatus_t err, const char* file, const int line, const char* func)
    {
        if (CURAND_STATUS_SUCCESS != err)
            cv::error(cv::Error::GpuApiCallError, curandGetErrorString(err), func, file, line);
    }
}}

#ifndef curandSafeCall
    #define curandSafeCall(expr)  cv::cuda::checkCurandError(expr, __FILE__, __LINE__, CV_Func)
#endif


namespace cv
{

namespace cuda
{

///
/// \brief RowWise2ColWise : copy data from a row-major management, to a column-major one.
/// \param src_data : source in memory
/// \param src_step : step of the data source in bytes (i.e. cols * sizeof(element) or (i.e. cols * channels * sizeof(element) )
/// \param dst_data : destination in memory
/// \param dst_step : step of the data destination in bytes (i.e. rows * sizeof(element) or (i.e. rows * channels * sizeof(element) )
/// \param rows : number of rows
/// \param cols : number of column
/// \param type : opencv type flag
/// \param stream : Stream of the asynchronous version.
///
void RowWise2ColWise(const void* src_data, size_t src_step, void* dst_data, size_t dst_step, int rows, int cols, int type, Stream& stream = Stream::Null());

///
/// \brief ColWise2RowWise : copy data from a column-major management, to a row-major one.
/// \param src_data : source in memory
/// \param src_step : step of the data source in bytes (i.e. rows * sizeof(element) or (i.e. rows * channels * sizeof(element) )
/// \param dst_data : destination in memory
/// \param dst_step : step of the data destination in bytes (i.e. cols * sizeof(element) or (i.e. cols * channels * sizeof(element) )
/// \param rows : number of rows
/// \param cols : number of column
/// \param type : opencv type flag
/// \param stream : Stream of the asynchronous version.
///
void ColWise2RowWise(const void* src_data, size_t src_step, void* dst_data, size_t dst_step, int rows, int cols, int type, Stream& stream = Stream::Null());


} // cuda

} // cv


#endif // CUDAXCORE_UTILS_HPP
