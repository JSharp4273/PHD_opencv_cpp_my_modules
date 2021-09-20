#include "opencv2/cudaxcore/utils.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

const char *curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

namespace cv
{

namespace cuda
{

namespace device
{

template<class T>
void CM2RW_caller(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int elem_size, cudaStream_t stream);

template<class T>
void RM2CW_caller(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int elem_size, cudaStream_t stream);

} // device

void RowWise2ColWise(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int type, Stream& stream)
{
    typedef void(*function_type)(const void*, size_t, void*, size_t, int, int, int, cudaStream_t);

    static const function_type funcs[] = {device::RM2CW_caller<uchar>,
                                          device::RM2CW_caller<schar>,
                                          device::RM2CW_caller<ushort>,
                                          device::RM2CW_caller<short>,
                                          device::RM2CW_caller<int>,
                                          device::RM2CW_caller<float>,
                                          device::RM2CW_caller<double>};

    int depth(CV_MAT_DEPTH(type));

    function_type fun = funcs[depth];

    fun(src_ptr, src_step, dst_ptr, dst_step, rows, cols, CV_ELEM_SIZE(type), StreamAccessor::getStream(stream));
}

void ColWise2RowWise(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int type, Stream& stream)
{
    typedef void(*function_type)(const void*, size_t, void*, size_t, int, int, int, cudaStream_t);

    static const function_type funcs[] = {device::CM2RW_caller<uchar>,
                                          device::CM2RW_caller<schar>,
                                          device::CM2RW_caller<ushort>,
                                          device::CM2RW_caller<short>,
                                          device::CM2RW_caller<int>,
                                          device::CM2RW_caller<float>,
                                          device::CM2RW_caller<double>};

    int depth(CV_MAT_DEPTH(type));

    function_type fun = funcs[depth];

    fun(src_ptr, src_step, dst_ptr, dst_step, rows, cols, CV_ELEM_SIZE(type), StreamAccessor::getStream(stream));
}

}// cuda

} // cv
