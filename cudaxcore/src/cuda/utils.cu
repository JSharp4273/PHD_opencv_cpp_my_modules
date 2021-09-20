#include "opencv2/cudaxcore/utils.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"

namespace cv
{

namespace cuda
{

namespace device
{

namespace
{

template<class T>
__global__ void RW2CW_kernel(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int elem_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= cols || y >= rows)
        return;

    const T* src = reinterpret_cast<const T*>(src_ptr + y * src_step) + x;
    T* dst = reinterpret_cast<T*>(dst_ptr + x * dst_step) + y;

    memcpy(dst, src, elem_size);

//    printf("dst_value: %f\n",*dst);
}

template<class T>
__global__ void CW2RW_kernel(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int elem_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= cols || y >= rows)
        return;

    const T* src = reinterpret_cast<const T*>(src_ptr + x * src_step) + y;
    T* dst = reinterpret_cast<T*>(dst_ptr + y * dst_step) + x;

    memcpy(dst, src, elem_size);
}

} // anonymous

template<class T>
void RM2CW_caller(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int elem_size, cudaStream_t stream)
{
    dim3 block(32,8);
    dim3 grid (divUp (cols, block.x), divUp (rows, block.y));

    cudaSafeCall( cudaFuncSetCacheConfig (RW2CW_kernel<T>, cudaFuncCachePreferL1) );
    RW2CW_kernel<T><<<grid, block, 0, stream>>>(src_ptr, src_step, dst_ptr, dst_step, rows, cols, elem_size);
    cudaSafeCall ( cudaGetLastError () );

    if (stream == 0)
         cudaSafeCall( cudaDeviceSynchronize() );
}

template<class T>
void CM2RW_caller(const void* src_ptr, size_t src_step, void* dst_ptr, size_t dst_step, int rows, int cols, int elem_size, cudaStream_t stream)
{
    dim3 block(32,8);
    dim3 grid (divUp (cols, block.x), divUp (rows, block.y));

    cudaSafeCall( cudaFuncSetCacheConfig (CW2RW_kernel<T>, cudaFuncCachePreferL1) );
    CW2RW_kernel<T><<<grid, block, 0, stream>>>(src_ptr, src_step, dst_ptr, dst_step, rows, cols, elem_size);
    cudaSafeCall ( cudaGetLastError () );

    if (stream == 0)
         cudaSafeCall( cudaDeviceSynchronize() );
}

} // device

} // cuda

} // cv

#define DECL_SPECS(type)\
    template void cv::cuda::device::RM2CW_caller<type>(const void*, size_t, void*, size_t, int, int, int, cudaStream_t); \
    template void cv::cuda::device::CM2RW_caller<type>(const void*, size_t, void*, size_t, int, int, int, cudaStream_t);

DECL_SPECS(uchar)
DECL_SPECS(schar)
DECL_SPECS(ushort)
DECL_SPECS(short)
DECL_SPECS(int)
DECL_SPECS(float)
DECL_SPECS(double)

#undef DECL_SPECS
