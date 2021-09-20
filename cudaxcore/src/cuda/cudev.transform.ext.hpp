#ifndef CUDEV_TRANSFORM_EXT_HPP
#define CUDEV_TRANSFORM_EXT_HPP

#if 0

#include "opencv2/cudev/common.hpp"
#include "opencv2/cudev/ptr2d/gpumat.hpp"
#include "opencv2/cudev/ptr2d/glob.hpp"
#include "opencv2/cudev/ptr2d/traits.hpp"
#include "opencv2/cudev/grid/transform.hpp"
#include "opencv2/cudev/grid/detail/transform.hpp"

namespace cv
{

namespace cudev
{

namespace grid_transform_detail
{

template<int cn>
struct OpUnroller_ext : OpUnroller<cn>
{};

template<>
struct OpUnroller_ext<1> : OpUnroller<1>
{
    template <typename T1, typename T2,typename T3, typename D, class TerOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, D& dst, const TerOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x,src3.x);
    }

    template <typename T1, typename T2,typename T3, typename T4, typename D, class QuaOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, D& dst, const QuaOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x,src3.x, src4.x);
    }

    template <typename T1, typename T2,typename T3, typename T4, typename T5, typename D, class QuiOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, D& dst, const QuiOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x,src3.x, src4.x, src5.x);
    }

    template <typename T1, typename T2,typename T3, typename T4, typename T5, typename T6, typename D, class SenOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, const T6& src6, D& dst, const SenOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x,src3.x, src4.x, src5.x, src6.x);
    }
};

template<>
struct OpUnroller_ext<2> : OpUnroller<2>
{
    template <typename T1, typename T2,typename T3, typename D, class TerOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, D& dst, const TerOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename D, class QuaOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, D& dst, const QuaOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5, typename D, class QuiOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, D& dst, const QuiOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x, src5.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y, src5.y);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename D, class SenOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, const T6& src6, D& dst, const SenOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x, src5.x, src6.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y, src5.y, src6.y);
    }

};


template<>
struct OpUnroller_ext<3> : OpUnroller<3>
{
    template <typename T1, typename T2,typename T3, typename D, class TerOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, D& dst, const TerOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename D, class QuaOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, D& dst, const QuaOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z, src4.z);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5, typename D, class QuiOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, D& dst, const QuiOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x, src5.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y, src5.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z, src4.z, src5.z);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename D, class SenOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, const T6& src6, D& dst, const SenOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x, src5.x, src6.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y, src5.y, src6.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z, src4.z, src5.z, src6.z);
    }

};


template<>
struct OpUnroller_ext<4> : OpUnroller<4>
{
    template <typename T1, typename T2,typename T3, typename D, class TerOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, D& dst, const TerOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z);
        if (mask(y, x_shifted + 3))
            dst.w = op(src1.w, src2.w, src3.w);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename D, class QuaOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, D& dst, const QuaOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z, src4.z);
        if (mask(y, x_shifted + 3))
            dst.w = op(src1.w, src2.w, src3.w, src4.w);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5, typename D, class QuiOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, D& dst, const QuiOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x, src5.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y, src5.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z, src4.z, src5.z);
        if (mask(y, x_shifted + 3))
            dst.w = op(src1.w, src2.w, src3.w, src4.w, src5.w);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename D, class SenOp, class MaskPtr>
    __device__ __forceinline__ static void unroll(const T1& src1, const T2& src2,const T3& src3, const T4& src4, const T5& src5, const T6& src6, D& dst, const SenOp& op, const MaskPtr& mask, int x_shifted, int y)
    {
        if (mask(y, x_shifted))
            dst.x = op(src1.x, src2.x, src3.x, src4.x, src5.x, src6.x);
        if (mask(y, x_shifted + 1))
            dst.y = op(src1.y, src2.y, src3.y, src4.y, src5.y, src6.y);
        if (mask(y, x_shifted + 2))
            dst.z = op(src1.z, src2.z, src3.z, src4.z, src5.z, src6.z);
        if (mask(y, x_shifted + 3))
            dst.w = op(src1.w, src2.w, src3.w, src4.w, src5.w, src6.w);
    }

};


//

template <class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp, class MaskPtr>
__global__ void transformSimple(const SrcPtr1 src1, const SrcPtr2 src2, const SrcPtr3 src3, GlobPtr<DstType> dst, const TerOp op, const MaskPtr mask, const int rows, const int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows || !mask(y, x))
        return;

    dst(y, x) = saturate_cast<DstType>(op(src1(y, x), src2(y, x),src3(y, x)));
}

template <class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, typename DstType, class QuaOp, class MaskPtr>
__global__ void transformSimple(const SrcPtr1 src1, const SrcPtr2 src2, const SrcPtr3 src3, const SrcPtr4 src4, GlobPtr<DstType> dst, const QuaOp op, const MaskPtr mask, const int rows, const int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows || !mask(y, x))
        return;

    dst(y, x) = saturate_cast<DstType>(op(src1(y, x), src2(y, x), src3(y, x), src4(y, x)));
}

template <class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class QuiOp, class MaskPtr>
__global__ void transformSimple(const SrcPtr1 src1, const SrcPtr2 src2, const SrcPtr3 src3, const SrcPtr4 src4, const SrcPtr5 src5, GlobPtr<DstType> dst, const QuiOp op, const MaskPtr mask, const int rows, const int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows || !mask(y, x))
        return;

    dst(y, x) = saturate_cast<DstType>(op(src1(y, x), src2(y, x), src3(y, x), src4(y, x), src5(y, x) ) );
}

template <class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class SenOp, class MaskPtr>
__global__ void transformSimple(const SrcPtr1 src1, const SrcPtr2 src2, const SrcPtr3 src3, const SrcPtr4 src4, const SrcPtr5 src5, const SrcPtr6 src6, GlobPtr<DstType> dst, const SenOp op, const MaskPtr mask, const int rows, const int cols)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows || !mask(y, x))
        return;

    dst(y, x) = saturate_cast<DstType>(op(src1(y, x), src2(y, x), src3(y, x), src4(y, x), src5(y, x), src6(y, x) ) );
}

template <int SHIFT, typename SrcType1, typename SrcType2,typename SrcType3, typename DstType, class TerOp, class MaskPtr>
__global__ void transformSmart(const GlobPtr<SrcType1> src1_, const GlobPtr<SrcType2> src2_,const GlobPtr<SrcType3> src3_, GlobPtr<DstType> dst_, const TerOp op, const MaskPtr mask, const int rows, const int cols)
{
    typedef typename MakeVec<SrcType1, SHIFT>::type read_type1;
    typedef typename MakeVec<SrcType2, SHIFT>::type read_type2;
    typedef typename MakeVec<SrcType3, SHIFT>::type read_type3;
    typedef typename MakeVec<DstType, SHIFT>::type write_type;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_shifted = x * SHIFT;

    if (y < rows)
    {
        const SrcType1* src1 = src1_.row(y);
        const SrcType2* src2 = src2_.row(y);
        const SrcType3* src3 = src3_.row(y);

        DstType* dst = dst_.row(y);

        if (x_shifted + SHIFT - 1 < cols)
        {
            const read_type1 src1_n_el = ((const read_type1*)src1)[x];
            const read_type2 src2_n_el = ((const read_type2*)src2)[x];
            const read_type3 src3_n_el = ((const read_type3*)src3)[x];

            write_type dst_n_el = ((const write_type*)dst)[x];

            OpUnroller_ext<SHIFT>::unroll(src1_n_el, src2_n_el,src3_n_el, dst_n_el, op, mask, x_shifted, y);

            ((write_type*)dst)[x] = dst_n_el;
        }
        else
        {
            for (int real_x = x_shifted; real_x < cols; ++real_x)
            {
                if (mask(y, real_x))
                    dst[real_x] = op(src1[real_x], src2[real_x], src3[real_x]);
            }
        }
    }
}

template <int SHIFT, typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename DstType, class QuaOp, class MaskPtr>
__global__ void transformSmart(const GlobPtr<SrcType1> src1_, const GlobPtr<SrcType2> src2_, const GlobPtr<SrcType3> src3_, const GlobPtr<SrcType4> src4_, GlobPtr<DstType> dst_, const QuaOp op, const MaskPtr mask, const int rows, const int cols)
{
    typedef typename MakeVec<SrcType1, SHIFT>::type read_type1;
    typedef typename MakeVec<SrcType2, SHIFT>::type read_type2;
    typedef typename MakeVec<SrcType3, SHIFT>::type read_type3;
    typedef typename MakeVec<SrcType4, SHIFT>::type read_type4;
    typedef typename MakeVec<DstType, SHIFT>::type write_type;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_shifted = x * SHIFT;

    if (y < rows)
    {
        const SrcType1* src1 = src1_.row(y);
        const SrcType2* src2 = src2_.row(y);
        const SrcType3* src3 = src3_.row(y);
        const SrcType4* src4 = src4_.row(y);

        DstType* dst = dst_.row(y);

        if (x_shifted + SHIFT - 1 < cols)
        {
            const read_type1 src1_n_el = ((const read_type1*)src1)[x];
            const read_type2 src2_n_el = ((const read_type2*)src2)[x];
            const read_type3 src3_n_el = ((const read_type3*)src3)[x];
            const read_type4 src4_n_el = ((const read_type4*)src4)[x];

            write_type dst_n_el = ((const write_type*)dst)[x];

            OpUnroller_ext<SHIFT>::unroll(src1_n_el, src2_n_el, src3_n_el, src4_n_el, dst_n_el, op, mask, x_shifted, y);

            ((write_type*)dst)[x] = dst_n_el;
        }
        else
        {
            for (int real_x = x_shifted; real_x < cols; ++real_x)
            {
                if (mask(y, real_x))
                    dst[real_x] = op(src1[real_x], src2[real_x], src3[real_x], src4[real_x]);
            }
        }
    }
}

template <int SHIFT, typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename SrcType5, typename DstType, class QuiOp, class MaskPtr>
__global__ void transformSmart(const GlobPtr<SrcType1> src1_, const GlobPtr<SrcType2> src2_, const GlobPtr<SrcType3> src3_, const GlobPtr<SrcType4> src4_, const GlobPtr<SrcType5> src5_, GlobPtr<DstType> dst_, const QuiOp op, const MaskPtr mask, const int rows, const int cols)
{
    typedef typename MakeVec<SrcType1, SHIFT>::type read_type1;
    typedef typename MakeVec<SrcType2, SHIFT>::type read_type2;
    typedef typename MakeVec<SrcType3, SHIFT>::type read_type3;
    typedef typename MakeVec<SrcType4, SHIFT>::type read_type4;
    typedef typename MakeVec<SrcType5, SHIFT>::type read_type5;
    typedef typename MakeVec<DstType, SHIFT>::type write_type;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_shifted = x * SHIFT;

    if (y < rows)
    {
        const SrcType1* src1 = src1_.row(y);
        const SrcType2* src2 = src2_.row(y);
        const SrcType3* src3 = src3_.row(y);
        const SrcType4* src4 = src4_.row(y);
        const SrcType5* src5 = src5_.row(y);

        DstType* dst = dst_.row(y);

        if (x_shifted + SHIFT - 1 < cols)
        {
            const read_type1 src1_n_el = ((const read_type1*)src1)[x];
            const read_type2 src2_n_el = ((const read_type2*)src2)[x];
            const read_type3 src3_n_el = ((const read_type3*)src3)[x];
            const read_type4 src4_n_el = ((const read_type4*)src4)[x];
            const read_type5 src5_n_el = ((const read_type4*)src5)[x];

            write_type dst_n_el = ((const write_type*)dst)[x];

            OpUnroller_ext<SHIFT>::unroll(src1_n_el, src2_n_el, src3_n_el, src4_n_el, src5_n_el, dst_n_el, op, mask, x_shifted, y);

            ((write_type*)dst)[x] = dst_n_el;
        }
        else
        {
            for (int real_x = x_shifted; real_x < cols; ++real_x)
            {
                if (mask(y, real_x))
                    dst[real_x] = op(src1[real_x], src2[real_x], src3[real_x], src4[real_x], src5[real_x]);
            }
        }
    }
}

template <int SHIFT, typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename SrcType5, typename SrcType6, typename DstType, class QuiOp, class MaskPtr>
__global__ void transformSmart(const GlobPtr<SrcType1> src1_, const GlobPtr<SrcType2> src2_, const GlobPtr<SrcType3> src3_, const GlobPtr<SrcType4> src4_, const GlobPtr<SrcType5> src5_, const GlobPtr<SrcType6> src6_, GlobPtr<DstType> dst_, const QuiOp op, const MaskPtr mask, const int rows, const int cols)
{
    typedef typename MakeVec<SrcType1, SHIFT>::type read_type1;
    typedef typename MakeVec<SrcType2, SHIFT>::type read_type2;
    typedef typename MakeVec<SrcType3, SHIFT>::type read_type3;
    typedef typename MakeVec<SrcType4, SHIFT>::type read_type4;
    typedef typename MakeVec<SrcType5, SHIFT>::type read_type5;
    typedef typename MakeVec<SrcType6, SHIFT>::type read_type6;
    typedef typename MakeVec<DstType, SHIFT>::type write_type;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x_shifted = x * SHIFT;

    if (y < rows)
    {
        const SrcType1* src1 = src1_.row(y);
        const SrcType2* src2 = src2_.row(y);
        const SrcType3* src3 = src3_.row(y);
        const SrcType4* src4 = src4_.row(y);
        const SrcType5* src5 = src5_.row(y);
        const SrcType6* src6 = src6_.row(y);

        DstType* dst = dst_.row(y);

        if (x_shifted + SHIFT - 1 < cols)
        {
            const read_type1 src1_n_el = ((const read_type1*)src1)[x];
            const read_type2 src2_n_el = ((const read_type2*)src2)[x];
            const read_type3 src3_n_el = ((const read_type3*)src3)[x];
            const read_type4 src4_n_el = ((const read_type4*)src4)[x];
            const read_type5 src5_n_el = ((const read_type5*)src5)[x];
            const read_type6 src6_n_el = ((const read_type6*)src6)[x];

            write_type dst_n_el = ((const write_type*)dst)[x];

            OpUnroller_ext<SHIFT>::unroll(src1_n_el, src2_n_el, src3_n_el, src4_n_el, src5_n_el, dst_n_el, op, mask, x_shifted, y);

            ((write_type*)dst)[x] = dst_n_el;
        }
        else
        {
            for (int real_x = x_shifted; real_x < cols; ++real_x)
            {
                if (mask(y, real_x))
                    dst[real_x] = op(src1[real_x], src2[real_x], src3[real_x], src4[real_x], src5[real_x], src6[real_x]);
            }
        }
    }
}

template <bool UseSmart, class Policy>
struct TransformDispatcher_ext : TransformDispatcher<UseSmart,Policy>
{

};

template <class Policy>
struct TransformDispatcher_ext<false, Policy> : TransformDispatcher<false,Policy>
{

    template <class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp, class MaskPtr>
    __host__ static void call(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, const GlobPtr<DstType>& dst, const TerOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        transformSimple<<<grid, block, 0, stream>>>(src1, src2, src3, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, typename DstType, class QuaOp, class MaskPtr>
    __host__ static void call(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const GlobPtr<DstType>& dst, const QuaOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        transformSimple<<<grid, block, 0, stream>>>(src1, src2, src3, src4, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class QuiOp, class MaskPtr>
    __host__ static void call(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const GlobPtr<DstType>& dst, const QuiOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        transformSimple<<<grid, block, 0, stream>>>(src1, src2, src3, src4, src5, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class SenOp, class MaskPtr>
    __host__ static void call(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, const GlobPtr<DstType>& dst, const SenOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

        transformSimple<<<grid, block, 0, stream>>>(src1, src2, src3, src4, src5, src6, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
};

template <class Policy>
struct TransformDispatcher_ext<true, Policy> : TransformDispatcher<true,Policy>
{

    template <typename SrcType1, typename SrcType2,typename SrcType3, typename DstType, class BinOp, class MaskPtr>
    __host__ static void call(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2,const GlobPtr<SrcType3>& src3, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        if (Policy::shift == 1 ||
            !isAligned(src1.data, Policy::shift * sizeof(SrcType1)) || !isAligned(src1.step, Policy::shift * sizeof(SrcType1)) ||
            !isAligned(src2.data, Policy::shift * sizeof(SrcType2)) || !isAligned(src2.step, Policy::shift * sizeof(SrcType2)) ||
            !isAligned(src3.data, Policy::shift * sizeof(SrcType3)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType3)) ||
            !isAligned(dst.data,  Policy::shift * sizeof(DstType))  || !isAligned(dst.step,  Policy::shift * sizeof(DstType)))
        {
            TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, dst, op, mask, rows, cols, stream);
            return;
        }

        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x * Policy::shift), divUp(rows, block.y));

        transformSmart<Policy::shift><<<grid, block, 0, stream>>>(src1, src2,src3, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename DstType, class BinOp, class MaskPtr>
    __host__ static void call(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<SrcType3>& src3, const GlobPtr<SrcType4>& src4, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        if (Policy::shift == 1 ||
            !isAligned(src1.data, Policy::shift * sizeof(SrcType1)) || !isAligned(src1.step, Policy::shift * sizeof(SrcType1)) ||
            !isAligned(src2.data, Policy::shift * sizeof(SrcType2)) || !isAligned(src2.step, Policy::shift * sizeof(SrcType2)) ||
            !isAligned(src3.data, Policy::shift * sizeof(SrcType3)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType3)) ||
            !isAligned(src4.data, Policy::shift * sizeof(SrcType4)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType4)) ||
            !isAligned(dst.data,  Policy::shift * sizeof(DstType))  || !isAligned(dst.step,  Policy::shift * sizeof(DstType)))
        {
            TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, src4, dst, op, mask, rows, cols, stream);
            return;
        }

        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x * Policy::shift), divUp(rows, block.y));

        transformSmart<Policy::shift><<<grid, block, 0, stream>>>(src1, src2, src3, src4, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename SrcType5, typename DstType, class BinOp, class MaskPtr>
    __host__ static void call(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<SrcType3>& src3, const GlobPtr<SrcType4>& src4, const GlobPtr<SrcType5>& src5, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        if (Policy::shift == 1 ||
            !isAligned(src1.data, Policy::shift * sizeof(SrcType1)) || !isAligned(src1.step, Policy::shift * sizeof(SrcType1)) ||
            !isAligned(src2.data, Policy::shift * sizeof(SrcType2)) || !isAligned(src2.step, Policy::shift * sizeof(SrcType2)) ||
            !isAligned(src3.data, Policy::shift * sizeof(SrcType3)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType3)) ||
            !isAligned(src4.data, Policy::shift * sizeof(SrcType4)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType4)) ||
            !isAligned(src5.data, Policy::shift * sizeof(SrcType5)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType5)) ||
            !isAligned(dst.data,  Policy::shift * sizeof(DstType))  || !isAligned(dst.step,  Policy::shift * sizeof(DstType)))
        {
            TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, src4, src5, dst, op, mask, rows, cols, stream);
            return;
        }

        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x * Policy::shift), divUp(rows, block.y));

        transformSmart<Policy::shift><<<grid, block, 0, stream>>>(src1, src2, src3, src4, src5, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }

    template <typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename SrcType5, typename SrcType6, typename DstType, class BinOp, class MaskPtr>
    __host__ static void call(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<SrcType3>& src3, const GlobPtr<SrcType4>& src4, const GlobPtr<SrcType5>& src5, const GlobPtr<SrcType6>& src6, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
    {
        if (Policy::shift == 1 ||
            !isAligned(src1.data, Policy::shift * sizeof(SrcType1)) || !isAligned(src1.step, Policy::shift * sizeof(SrcType1)) ||
            !isAligned(src2.data, Policy::shift * sizeof(SrcType2)) || !isAligned(src2.step, Policy::shift * sizeof(SrcType2)) ||
            !isAligned(src3.data, Policy::shift * sizeof(SrcType3)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType3)) ||
            !isAligned(src4.data, Policy::shift * sizeof(SrcType4)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType4)) ||
            !isAligned(src5.data, Policy::shift * sizeof(SrcType5)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType5)) ||
            !isAligned(src6.data, Policy::shift * sizeof(SrcType6)) || !isAligned(src3.step, Policy::shift * sizeof(SrcType6)) ||
            !isAligned(dst.data,  Policy::shift * sizeof(DstType))  || !isAligned(dst.step,  Policy::shift * sizeof(DstType)))
        {
            TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, src4, src5, src6, dst, op, mask, rows, cols, stream);
            return;
        }

        const dim3 block(Policy::block_size_x, Policy::block_size_y);
        const dim3 grid(divUp(cols, block.x * Policy::shift), divUp(rows, block.y));

        transformSmart<Policy::shift><<<grid, block, 0, stream>>>(src1, src2, src3, src4, src5, src6, dst, op, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( cudaGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( cudaDeviceSynchronize() );
    }
};



template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_ternary(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, dst, op, mask, rows, cols, stream);
}

template <class Policy, typename SrcType1, typename SrcType2,typename SrcType3, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_ternary(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2,const GlobPtr<SrcType3>& src3, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<VecTraits<SrcType1>::cn == 1 && VecTraits<SrcType2>::cn == 1 && VecTraits<SrcType3>::cn == 1 && VecTraits<DstType>::cn == 1 && Policy::shift != 1, Policy>::call(src1, src2, src3, dst, op, mask, rows, cols, stream);
}



template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_quaternary(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, const SrcPtr4& src4, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, src4, dst, op, mask, rows, cols, stream);
}

template <class Policy, typename SrcType1, typename SrcType2, typename SrcType3, typename SrcType4, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_quaternary(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<SrcType3>& src3, const GlobPtr<SrcType4>& src4, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<VecTraits<SrcType1>::cn == 1 && VecTraits<SrcType2>::cn == 1 && VecTraits<SrcType3>::cn == 1 && VecTraits<DstType>::cn == 1 && Policy::shift != 1, Policy>::call(src1, src2, src3, src4, dst, op, mask, rows, cols, stream);
}



template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_quinary(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, src4, src5, dst, op, mask, rows, cols, stream);
}

template <class Policy, typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename SrcType5, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_quinary(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<SrcType3>& src3, const GlobPtr<SrcType4>& src4, const GlobPtr<SrcType5>& src5, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<VecTraits<SrcType1>::cn == 1 && VecTraits<SrcType2>::cn == 1 && VecTraits<SrcType3>::cn == 1 && VecTraits<DstType>::cn == 1 && Policy::shift != 1, Policy>::call(src1, src2, src3, src4, src5, dst, op, mask, rows, cols, stream);
}



template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_senary(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<false, Policy>::call(src1, src2, src3, src4, src5, dst, op, mask, rows, cols, stream);
}

template <class Policy, typename SrcType1, typename SrcType2,typename SrcType3, typename SrcType4, typename SrcType5, class SrcType6, typename DstType, class BinOp, class MaskPtr>
__host__ void transform_senary(const GlobPtr<SrcType1>& src1, const GlobPtr<SrcType2>& src2, const GlobPtr<SrcType3>& src3, const GlobPtr<SrcType4>& src4, const GlobPtr<SrcType5>& src5, const GlobPtr<SrcType6>& src6, const GlobPtr<DstType>& dst, const BinOp& op, const MaskPtr& mask, int rows, int cols, cudaStream_t stream)
{
    TransformDispatcher_ext<VecTraits<SrcType1>::cn == 1 && VecTraits<SrcType2>::cn == 1 && VecTraits<SrcType3>::cn == 1 && VecTraits<DstType>::cn == 1 && Policy::shift != 1, Policy>::call(src1, src2, src3, src4, src5, src6, dst, op, mask, rows, cols, stream);
}


} // grid_transform_detail

// TERNARY

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformTernary_(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_ternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformTernary_(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_ternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp>
__host__ void gridTransformTernary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_ternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp>
__host__ void gridTransformTernary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );

    grid_transform_detail::transform_ternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}


template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformTernary(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTernary_<DefaultTransformPolicy>(src1, src2, src3, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformTernary(const SrcPtr1& src1, const SrcPtr2& src2,const SrcPtr3& src3, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformTernary_<DefaultTransformPolicy>(src1, src2, src3, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp>
__host__ void gridTransformTernary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformTernary_<DefaultTransformPolicy>(src1, src2, src3, dst, op, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, typename DstType, class TerOp>
__host__ void gridTransformTernary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformTernary_<DefaultTransformPolicy>(src1, src2, src3, dst, op, stream);
}

// QUATERNARY

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuaternary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuaternary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, typename DstType, class TerOp>
__host__ void gridTransformQuaternary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, typename DstType, class TerOp>
__host__ void gridTransformQuaternary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}


template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuaternary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformQuaternary_<DefaultTransformPolicy>(src1, src2, src3, src4, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuaternary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformQuaternary_<DefaultTransformPolicy>(src1, src2, src3, src4, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, typename DstType, class TerOp>
__host__ void gridTransformQuaternary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformQuaternary_<DefaultTransformPolicy>(src1, src2, src3, src4, dst, op, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, typename DstType, class TerOp>
__host__ void gridTransformQuaternary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformQuaternary_<DefaultTransformPolicy>(src1, src2, src3, src4, dst, op, stream);
}

// QINARY


template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuirnary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(src5), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuirnary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(src5), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp>
__host__ void gridTransformQuirnary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(src5), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp>
__host__ void gridTransformQuirnary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(src5), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}


template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuirnary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformQuirnary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformQuirnary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr4& src5, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformQuirnary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp>
__host__ void gridTransformQuirnary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformQuirnary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, dst, op, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp>
__host__ void gridTransformQuirnary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformQuirnary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, dst, op, stream);
}

// SENARY


template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformSenary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr5& src6, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );
    CV_Assert( getRows(src6) == rows && getCols(src6) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformSenary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );
    CV_Assert( getRows(mask) == rows && getCols(mask) == cols );
    CV_Assert( getRows(src6) == rows && getCols(src6) == cols );

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, shrinkPtr(mask), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class TerOp>
__host__ void gridTransformSenary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );
    CV_Assert( getRows(src6) == rows && getCols(src6) == cols );

    dst.create(rows, cols);

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class TerOp>
__host__ void gridTransformSenary_(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    const int rows = getRows(src1);
    const int cols = getCols(src1);

    CV_Assert( getRows(dst) == rows && getCols(dst) == cols );
    CV_Assert( getRows(src2) == rows && getCols(src2) == cols );
    CV_Assert( getRows(src3) == rows && getCols(src3) == cols );
    CV_Assert( getRows(src4) == rows && getCols(src4) == cols );
    CV_Assert( getRows(src5) == rows && getCols(src5) == cols );
    CV_Assert( getRows(src6) == rows && getCols(src6) == cols );

    grid_transform_detail::transform_quaternary<Policy>(shrinkPtr(src1), shrinkPtr(src2), shrinkPtr(src3), shrinkPtr(src4), shrinkPtr(dst), op, WithOutMask(), rows, cols, StreamAccessor::getStream(stream));
}


template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformSenary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, GpuMat_<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformSenary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, src6, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class TerOp, class MaskPtr>
__host__ void gridTransformSenary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, const GlobPtrSz<DstType>& dst, const TerOp& op, const MaskPtr& mask, Stream& stream = Stream::Null())
{
    gridTransformSenary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, src6, dst, op, mask, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2, class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class TerOp>
__host__ void gridTransformSenary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, GpuMat_<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformSenary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, src6, dst, op, stream);
}

template <class Policy, class SrcPtr1, class SrcPtr2,class SrcPtr3, class SrcPtr4, class SrcPtr5, class SrcPtr6, typename DstType, class TerOp>
__host__ void gridTransformSenary(const SrcPtr1& src1, const SrcPtr2& src2, const SrcPtr3& src3, const SrcPtr4& src4, const SrcPtr5& src5, const SrcPtr6& src6, const GlobPtrSz<DstType>& dst, const TerOp& op, Stream& stream = Stream::Null())
{
    gridTransformSenary_<DefaultTransformPolicy>(src1, src2, src3, src4, src5, src6, dst, op, stream);
}

} // cudev

} // cv

#endif

#endif // CUDEV_TRANSFORM_EXT_HPP
