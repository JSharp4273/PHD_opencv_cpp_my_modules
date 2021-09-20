#include "opencv2/cudaxcore.hpp"
#include "precomp.hpp"

namespace cv
{

namespace cuda
{

#ifndef HAVE_CUDA
void clip(InputArray _src, InputArray _l, InputArray _h, Stream& stream)
{
    CV_Error(Error::StsError, "Cuda modules have not been compiled");
}
#else

namespace device
{
template<class T>
void clipImpl(const GpuMat& _src, const Scalar& _l, const Scalar& _u, GpuMat& _dst, Stream& stream);
} // device


void clip(InputArray _src, InputArray _l, InputArray _h, OutputArray _dst, Stream& stream)
{
    CV_Assert(_src.isGpuMat() && (_l.isMatx() && (_l.total() <= 4)) && (_h.isMatx() && (_h.total() <= 4)) );

    using device::clipImpl;

    typedef void (*function_type)(const GpuMat&, const Scalar&, const Scalar&, GpuMat&, Stream&);

    static const function_type funcs[7][4] = {
        {clipImpl<uchar>, clipImpl<uchar3>, clipImpl<uchar3>, clipImpl<uchar4>},
        {clipImpl<schar>, clipImpl<char3>, clipImpl<char3>, clipImpl<char4>},
        {clipImpl<ushort>, clipImpl<ushort3>, clipImpl<ushort3>, clipImpl<ushort4>},
        {clipImpl<short>, clipImpl<short3>, clipImpl<short3>, clipImpl<short4>},
        {clipImpl<int>, clipImpl<int3>, clipImpl<int3>, clipImpl<int4>},
        {clipImpl<float>, clipImpl<float3>, clipImpl<float3>, clipImpl<float4>},
        {clipImpl<double>, clipImpl<double3>, clipImpl<double3>, clipImpl<double4>}
    };

    function_type fun = funcs[_src.depth()][_src.channels()];

    fun(_src.getGpuMat(), getScalar(_l), getScalar(_h), _dst.getGpuMatRef(), stream);

    /////////////// Possible alternative implementation (un verified) ///////////////

//    GpuMat src = _src.getGpuMat(), lt, ht, nlt, nht, all_mask;

//    const int stype = src.type();
//    const int sdepth = CV_MAT_DEPTH(stype);
//    const int cn = CV_MAT_CN(stype);
//    const int wdepth = std::max(sdepth, CV_33F);
////    const int wtype = CV_MAKETYPE(wdepth, cn);

//    if(_src.channels() == 1)
//    {
//        // find the postions to replace.
//        compare(src, _l, lt, CMP_LT, stream);
//        compare(src, _h, ht, CMP_GT, stream);

//        // find the positions to keep.
//        bitwise_not(lt, nlt, noArray(), stream);
//        bitwise_not(ht, nht, noArray(), stream);

//        // erase the elements with a value lower than _l and higher than _h.
//        bitwise_xor(nlt, nht, all_mask, noArray(), stream);
//        if(sdepth>CV_8U)
//        {
//            bitwise_and(all_mask, 1., all_mask, noArray(), stream);
//            multiply(src, all_mask, src, 1., sdepth, stream);
//        }
//        else
//            bitwise_and(src, all_mask, src, noArray(), stream);

//        // replace
//        bitwise_xor(src, _l, src, lt, stream);
//        bitwise_xor(src, _h, src, ht, stream);

//    }
//    else
//    {
//        Scalar __l(getScalar(_l)), __h(getScalar(_h));

//        std::vector<GpuMat> cns;
//        split(src, cns, stream);

//        for(size_t i=0; i<cns.size(); i++)
//        {

//            src = cns.at(i);

//            // find the postions to replace.
//            compare(src, __l(i), lt, CMP_LT, stream);
//            compare(src, __h(i), ht, CMP_GT, stream);

//            // find the positions to keep.
//            bitwise_not(lt, nlt, noArray(), stream);
//            bitwise_not(ht, nht, noArray(), stream);

//            // erase the elements with a value lower than _l and higher than _h.
//            bitwise_xor(nlt, nht, all_mask, noArray(), stream);
//            if(sdepth>CV_8U)
//            {
//                bitwise_and(all_mask, 1., all_mask, noArray(), stream);
//                multiply(src, all_mask, src, 1., sdepth, stream);
//            }
//            else
//                bitwise_and(src, all_mask, src, noArray(), stream);

//            // replace
//            bitwise_xor(src, __l(i), src, lt, stream);
//            bitwise_xor(src, __h(i), src, ht, stream);

//            if(cns.at(i).data != src.data)
//                cns.at(i) = src;

//        }
//    }

}


namespace device
{

template<class T>
void kronImpl(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

}// anonymous


void kron(InputArray _src1, InputArray _src3, OutputArray _dst, Stream& stream)
{
    typedef void(*function_type)(const GpuMat&, const GpuMat&, GpuMat&, Stream&);

    static const function_type funcs[7][4] = {
        {device::kronImpl<uchar >, device::kronImpl<uchar3 >, device::kronImpl<uchar3 >, device::kronImpl<uchar4 >},
        {device::kronImpl<schar >, device::kronImpl<char3  >, device::kronImpl<char3  >, device::kronImpl<char4  >},
        {device::kronImpl<ushort>, device::kronImpl<ushort3>, device::kronImpl<ushort3>, device::kronImpl<ushort4>},
        {device::kronImpl<short >, device::kronImpl<short3 >, device::kronImpl<short3 >, device::kronImpl<short4 >},
        {device::kronImpl<int   >, device::kronImpl<int3   >, device::kronImpl<int3   >, device::kronImpl<int4   >},
        {device::kronImpl<float >, device::kronImpl<float3 >, device::kronImpl<float3 >, device::kronImpl<float4 >},
        {device::kronImpl<double>, device::kronImpl<double3>, device::kronImpl<double3>, device::kronImpl<double4>},
    };

    CV_Assert(_src1.type() == _src3.type() && _src1.channels() <= 4 && _src1.isGpuMat() && _src3.isGpuMat() && _dst.isGpuMat());

    GpuMat src1(_src1.getGpuMat()), src3(_src3.getGpuMat());
    GpuMat dst(_src1.rows() * _src3.rows(), _src1.cols() * _src3.cols(), _src1.type(), Scalar::all(0.));

    function_type fun = funcs[src1.depth()][src1.channels()-1];

    fun(src1, src3, dst, stream);


    dst.copyTo(_dst, stream);
}

#if 0

namespace device
{

template<class SrcType, class DstType>
void recip_rd(const GpuMat& _src, GpuMat& _dst, Stream& _stream);

template<class SrcType, class DstType>
void recip_ru(const GpuMat& _src, GpuMat& _dst, Stream& _stream);

template<class SrcType, class DstType>
void recip_rn(const GpuMat& _src, GpuMat& _dst, Stream& _stream);

template<class SrcType, class DstType>
void recip_rz(const GpuMat& _src, GpuMat& _dst, Stream& _stream);



template<class SrcType, class DstType>
void masked_recip_rd(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream);

template<class SrcType, class DstType>
void masked_recip_ru(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream);

template<class SrcType, class DstType>
void masked_recip_rn(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream);

template<class SrcType, class DstType>
void masked_recip_rz(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream);


} // device

void reciprocal(InputArray _src, OutputArray _dst, int flag_norm, int dtype, InputArray _mask, Stream& _stream)
{
    CV_Assert(_src.isGpuMat() && _dst.isGpuMat() && (_mask.empty() || (_mask.isGpuMat() && (_mask.size() == _src.size()) && (_mask.type() == CV_8UC1) ) ));

    GpuMat src = _src.getGpuMat();
    GpuMat dst;



    int stype = src.type();
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int wdepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruction_needed(false);

    if(cn>4)
    {
        reconstruction_needed = true;

        GpuMat tmp;

        if(!src.isContinuous())
        {
            src.copyTo(tmp, _stream);
            src.release();
            src = tmp;
        }

        tmp = src.reshape(1);
        src = tmp;
    }

    dst.create(src.size(), !reconstruction_needed ? wtype : wdepth);

    if(_mask.empty())
    {

        typedef void(*function_type)(const GpuMat&, GpuMat&, Stream&);

        static const function_type funcs[7][7][4][4] = {
            {

                {
                    {device::recip_rz<uchar  , uchar  >, device::recip_rd<uchar  , uchar  >, device::recip_ru<uchar  , uchar  >, device::recip_rn<uchar  , uchar  >},
                    {device::recip_rz<uchar2 , uchar2 >, device::recip_rd<uchar2 , uchar2 >, device::recip_ru<uchar2 , uchar2 >, device::recip_rn<uchar2 , uchar2 >},
                    {device::recip_rz<uchar3 , uchar3 >, device::recip_rd<uchar3 , uchar3 >, device::recip_ru<uchar3 , uchar3 >, device::recip_rn<uchar3 , uchar3 >},
                    {device::recip_rz<uchar4 , uchar4 >, device::recip_rd<uchar4 , uchar4 >, device::recip_ru<uchar4 , uchar4 >, device::recip_rn<uchar4 , uchar4 >}
                },
                {
                    {device::recip_rz<uchar  , schar >, device::recip_rd<uchar  , schar >, device::recip_ru<uchar  , schar >, device::recip_rn<uchar  , schar >},
                    {device::recip_rz<uchar2 , char2 >, device::recip_rd<uchar2 , char2 >, device::recip_ru<uchar2 , char2 >, device::recip_rn<uchar2 , char2 >},
                    {device::recip_rz<uchar3 , char3 >, device::recip_rd<uchar3 , char3 >, device::recip_ru<uchar3 , char3 >, device::recip_rn<uchar3 , char3 >},
                    {device::recip_rz<uchar4 , char4 >, device::recip_rd<uchar4 , char4 >, device::recip_ru<uchar4 , char4 >, device::recip_rn<uchar4 , char4 >}
                },
                {
                    {device::recip_rz<uchar  , ushort  >, device::recip_rd<uchar  , ushort  >, device::recip_ru<uchar  , ushort  >, device::recip_rn<uchar  , ushort  >},
                    {device::recip_rz<uchar2 , ushort2 >, device::recip_rd<uchar2 , ushort2 >, device::recip_ru<uchar2 , ushort2 >, device::recip_rn<uchar2 , ushort2 >},
                    {device::recip_rz<uchar3 , ushort3 >, device::recip_rd<uchar3 , ushort3 >, device::recip_ru<uchar3 , ushort3 >, device::recip_rn<uchar3 , ushort3 >},
                    {device::recip_rz<uchar4 , ushort4 >, device::recip_rd<uchar4 , ushort4 >, device::recip_ru<uchar4 , ushort4 >, device::recip_rn<uchar4 , ushort4 >}
                },
                {
                    {device::recip_rz<uchar  , short  >, device::recip_rd<uchar  , short  >, device::recip_ru<uchar  , short  >, device::recip_rn<uchar  , short  >},
                    {device::recip_rz<uchar2 , short2 >, device::recip_rd<uchar2 , short2 >, device::recip_ru<uchar2 , short2 >, device::recip_rn<uchar2 , short2 >},
                    {device::recip_rz<uchar3 , short3 >, device::recip_rd<uchar3 , short3 >, device::recip_ru<uchar3 , short3 >, device::recip_rn<uchar3 , short3 >},
                    {device::recip_rz<uchar4 , short4 >, device::recip_rd<uchar4 , short4 >, device::recip_ru<uchar4 , short4 >, device::recip_rn<uchar4 , short4 >}
                },
                {
                    {device::recip_rz<uchar  , int  >, device::recip_rd<uchar  , int  >, device::recip_ru<uchar ,  int  >, device::recip_rn<uchar  , int  >},
                    {device::recip_rz<uchar2 , int2 >, device::recip_rd<uchar2 , int2 >, device::recip_ru<uchar2 , int2 >, device::recip_rn<uchar2 , int2 >},
                    {device::recip_rz<uchar3 , int3 >, device::recip_rd<uchar3 , int3 >, device::recip_ru<uchar3 , int3 >, device::recip_rn<uchar3 , int3 >},
                    {device::recip_rz<uchar4 , int4 >, device::recip_rd<uchar4 , int4 >, device::recip_ru<uchar4 , int4 >, device::recip_rn<uchar4 , int4 >}
                },
                {
                    {device::recip_rz<uchar  , float  >, device::recip_rd<uchar  , float  >, device::recip_ru<uchar  , float  >, device::recip_rn<uchar  , float  >},
                    {device::recip_rz<uchar2 , float2 >, device::recip_rd<uchar2 , float2 >, device::recip_ru<uchar2 , float2 >, device::recip_rn<uchar2 , float2 >},
                    {device::recip_rz<uchar3 , float3 >, device::recip_rd<uchar3 , float3 >, device::recip_ru<uchar3 , float3 >, device::recip_rn<uchar3 , float3 >},
                    {device::recip_rz<uchar4 , float4 >, device::recip_rd<uchar4 , float4 >, device::recip_ru<uchar4 , float4 >, device::recip_rn<uchar4 , float4 >}
                },
                {
                    {device::recip_rz<uchar  , double  >, device::recip_rd<uchar  , double  >, device::recip_ru<uchar , double  >, device::recip_rn< uchar  , double  >},
                    {device::recip_rz<uchar2 , double2 >, device::recip_rd<uchar2 , double2 >, device::recip_ru<uchar2 , double2 >, device::recip_rn<uchar2 , double2 >},
                    {device::recip_rz<uchar3 , double3 >, device::recip_rd<uchar3 , double3 >, device::recip_ru<uchar3 , double3 >, device::recip_rn<uchar3 , double3 >},
                    {device::recip_rz<uchar4 , double4 >, device::recip_rd<uchar4 , double4 >, device::recip_ru<uchar4 , double4 >, device::recip_rn<uchar4 , double4 >}
                }
            },
            {

                {
                    {device::recip_rz<schar , uchar >, device::recip_rd<schar  , uchar >, device::recip_ru<schar  , uchar  >, device::recip_rn<schar , uchar  >},
                    {device::recip_rz<char2 , uchar2 >, device::recip_rd<char2 , uchar2 >, device::recip_ru<char2 , uchar2 >, device::recip_rn<char2 , uchar2 >},
                    {device::recip_rz<char3 , uchar3 >, device::recip_rd<char3 , uchar3 >, device::recip_ru<char3 , uchar3 >, device::recip_rn<char3 , uchar3 >},
                    {device::recip_rz<char4 , uchar4 >, device::recip_rd<char4 , uchar4 >, device::recip_ru<char4 , uchar4 >, device::recip_rn<char4 , uchar4 >}
                },
                {
                    {device::recip_rz<schar , schar >, device::recip_rd<schar , schar >, device::recip_ru<schar , schar >, device::recip_rn<schar , schar >},
                    {device::recip_rz<char2 , char2 >, device::recip_rd<char2 , char2 >, device::recip_ru<char2 , char2 >, device::recip_rn<char2 , char2 >},
                    {device::recip_rz<char3 , char3 >, device::recip_rd<char3 , char3 >, device::recip_ru<char3 , char3 >, device::recip_rn<char3 , char3 >},
                    {device::recip_rz<char4 , char4 >, device::recip_rd<char4 , char4 >, device::recip_ru<char4 , char4 >, device::recip_rn<char4 , char4 >}
                },
                {
                    {device::recip_rz<schar , ushort  >, device::recip_rd<schar , ushort  >, device::recip_ru<schar,  ushort  >, device::recip_rn<schar , ushort  >},
                    {device::recip_rz<char2 , ushort2 >, device::recip_rd<char2 , ushort2 >, device::recip_ru<char2 , ushort2 >, device::recip_rn<char2 , ushort2 >},
                    {device::recip_rz<char3 , ushort3 >, device::recip_rd<char3 , ushort3 >, device::recip_ru<char3 , ushort3 >, device::recip_rn<char3 , ushort3 >},
                    {device::recip_rz<char4 , ushort4 >, device::recip_rd<char4 , ushort4 >, device::recip_ru<char4 , ushort4 >, device::recip_rn<char4 , ushort4 >}
                },
                {
                    {device::recip_rz<schar , short  >, device::recip_rd<schar , short  >, device::recip_ru<schar , short  >, device::recip_rn<schar , short  >},
                    {device::recip_rz<char2 , short2 >, device::recip_rd<char2 , short2 >, device::recip_ru<char2 , short2 >, device::recip_rn<char2 , short2 >},
                    {device::recip_rz<char3 , short3 >, device::recip_rd<char3 , short3 >, device::recip_ru<char3 , short3 >, device::recip_rn<char3 , short3 >},
                    {device::recip_rz<char4 , short4 >, device::recip_rd<char4 , short4 >, device::recip_ru<char4 , short4 >, device::recip_rn<char4 , short4 >}
                },
                {
                    {device::recip_rz<schar , int  >, device::recip_rd<schar , int  >, device::recip_ru<schar , int  >, device::recip_rn<schar , int  >},
                    {device::recip_rz<char2 , int2 >, device::recip_rd<char2 , int2 >, device::recip_ru<char2 , int2 >, device::recip_rn<char2 , int2 >},
                    {device::recip_rz<char3 , int3 >, device::recip_rd<char3 , int3 >, device::recip_ru<char3 , int3 >, device::recip_rn<char3 , int3 >},
                    {device::recip_rz<char4 , int4 >, device::recip_rd<char4 , int4 >, device::recip_ru<char4 , int4 >, device::recip_rn<char4 , int4 >}
                },
                {
                    {device::recip_rz<schar , float  >, device::recip_rd<schar , float  >, device::recip_ru<schar , float  >, device::recip_rn<schar , float  >},
                    {device::recip_rz<char2 , float2 >, device::recip_rd<char2 , float2 >, device::recip_ru<char2 , float2 >, device::recip_rn<char2 , float2 >},
                    {device::recip_rz<char3 , float3 >, device::recip_rd<char3 , float3 >, device::recip_ru<char3 , float3 >, device::recip_rn<char3 , float3 >},
                    {device::recip_rz<char4 , float4 >, device::recip_rd<char4 , float4 >, device::recip_ru<char4 , float4 >, device::recip_rn<char4 , float4 >}
                },
                {
                    {device::recip_rz<schar , double  >, device::recip_rd<schar , double  >, device::recip_ru<schar , double  >, device::recip_rn<schar, double   >},
                    {device::recip_rz<char2 , double2 >, device::recip_rd<char2 , double2 >, device::recip_ru<char2 , double2 >, device::recip_rn<char2 , double2 >},
                    {device::recip_rz<char3 , double3 >, device::recip_rd<char3 , double3 >, device::recip_ru<char3 , double3 >, device::recip_rn<char3 , double3 >},
                    {device::recip_rz<char4 , double4 >, device::recip_rd<char4 , double4 >, device::recip_ru<char4 , double4 >, device::recip_rn<char4 , double4 >}
                }
            },
            {

                {
                    {device::recip_rz<ushort , uchar  >, device::recip_rd<ushort , uchar  >, device::recip_ru<ushort , uchar  >, device::recip_rn<ushort , uchar  >},
                    {device::recip_rz<ushort2, uchar2 >, device::recip_rd<ushort3, uchar3 >, device::recip_ru<ushort2, uchar2 >, device::recip_rn<ushort2, uchar2 >},
                    {device::recip_rz<ushort3, uchar3 >, device::recip_rd<ushort3, uchar3 >, device::recip_ru<ushort3, uchar3 >, device::recip_rn<ushort3, uchar3 >},
                    {device::recip_rz<ushort4, uchar4 >, device::recip_rd<ushort4, uchar4 >, device::recip_ru<ushort4, uchar4 >, device::recip_rn<ushort4, uchar4 >}
                },
                {
                    {device::recip_rz<ushort , schar >, device::recip_rd<ushort , schar >, device::recip_ru<ushort , schar >, device::recip_rn<ushort , schar >},
                    {device::recip_rz<ushort2, char2 >, device::recip_rd<ushort2, char2 >, device::recip_ru<ushort2, char2 >, device::recip_rn<ushort2, char2 >},
                    {device::recip_rz<ushort3, char3 >, device::recip_rd<ushort3, char3 >, device::recip_ru<ushort3, char3 >, device::recip_rn<ushort3, char3 >},
                    {device::recip_rz<ushort4, char4 >, device::recip_rd<ushort4, char4 >, device::recip_ru<ushort4, char4 >, device::recip_rn<ushort4, char4 >}
                },
                {
                    {device::recip_rz<ushort , ushort  >, device::recip_rd<ushort , ushort  >, device::recip_ru<ushort , ushort  >, device::recip_rn<ushort , ushort  >},
                    {device::recip_rz<ushort2, ushort2 >, device::recip_rd<ushort2, ushort2 >, device::recip_ru<ushort2, ushort2 >, device::recip_rn<ushort2, ushort2 >},
                    {device::recip_rz<ushort3, ushort3 >, device::recip_rd<ushort3, ushort3 >, device::recip_ru<ushort3, ushort3 >, device::recip_rn<ushort3, ushort3 >},
                    {device::recip_rz<ushort4, ushort4 >, device::recip_rd<ushort4, ushort4 >, device::recip_ru<ushort4, ushort4 >, device::recip_rn<ushort4, ushort4 >}
                },
                {
                    {device::recip_rz<ushort , short  >, device::recip_rd<ushort , short  >, device::recip_ru<ushort , short  >, device::recip_rn<ushort , short  >},
                    {device::recip_rz<ushort2, short2 >, device::recip_rd<ushort2, short2 >, device::recip_ru<ushort2, short2 >, device::recip_rn<ushort2, short2 >},
                    {device::recip_rz<ushort3, short3 >, device::recip_rd<ushort3, short3 >, device::recip_ru<ushort3, short3 >, device::recip_rn<ushort3, short3 >},
                    {device::recip_rz<ushort4, short4 >, device::recip_rd<ushort4, short4 >, device::recip_ru<ushort4, short4 >, device::recip_rn<ushort4, short4 >}
                },
                {
                    {device::recip_rz<ushort , int  >, device::recip_rd<ushort , int  >, device::recip_ru<ushort , int  >, device::recip_rn<ushort , int  >},
                    {device::recip_rz<ushort2, int2 >, device::recip_rd<ushort2, int2 >, device::recip_ru<ushort2, int2 >, device::recip_rn<ushort2, int2 >},
                    {device::recip_rz<ushort3, int3 >, device::recip_rd<ushort3, int3 >, device::recip_ru<ushort3, int3 >, device::recip_rn<ushort3, int3 >},
                    {device::recip_rz<ushort4, int4 >, device::recip_rd<ushort4, int4 >, device::recip_ru<ushort4, int4 >, device::recip_rn<ushort4, int4 >}
                },
                {
                    {device::recip_rz<ushort , float  >, device::recip_rd<ushort , float  >, device::recip_ru<ushort , float  >, device::recip_rn<ushort , float  >},
                    {device::recip_rz<ushort2, float2 >, device::recip_rd<ushort2, float2 >, device::recip_ru<ushort2, float2 >, device::recip_rn<ushort2, float2 >},
                    {device::recip_rz<ushort3, float3 >, device::recip_rd<ushort3, float3 >, device::recip_ru<ushort3, float3 >, device::recip_rn<ushort3, float3 >},
                    {device::recip_rz<ushort4, float4 >, device::recip_rd<ushort4, float4 >, device::recip_ru<ushort4, float4 >, device::recip_rn<ushort4, float4 >}
                },
                {
                    {device::recip_rz<ushort , double  >, device::recip_rd<ushort , double  >, device::recip_ru<ushort , double  >, device::recip_rn<ushort , double  >},
                    {device::recip_rz<ushort2, double2 >, device::recip_rd<ushort2, double2 >, device::recip_ru<ushort2, double2 >, device::recip_rn<ushort2, double2 >},
                    {device::recip_rz<ushort3, double3 >, device::recip_rd<ushort3, double3 >, device::recip_ru<ushort3, double3 >, device::recip_rn<ushort3, double3 >},
                    {device::recip_rz<ushort4, double4 >, device::recip_rd<ushort4, double4 >, device::recip_ru<ushort4, double4 >, device::recip_rn<ushort4, double4 >}
                }
            },
            {

                {
                    {device::recip_rz<short , uchar  >, device::recip_rd<short , uchar  >, device::recip_ru<short , uchar  >, device::recip_rn<short , uchar  >},
                    {device::recip_rz<short2, uchar2 >, device::recip_rd<short2, uchar2 >, device::recip_ru<short2, uchar2 >, device::recip_rn<short2, uchar2 >},
                    {device::recip_rz<short3, uchar3 >, device::recip_rd<short3, uchar3 >, device::recip_ru<short3, uchar3 >, device::recip_rn<short3, uchar3 >},
                    {device::recip_rz<short4, uchar4 >, device::recip_rd<short4, uchar4 >, device::recip_ru<short4, uchar4 >, device::recip_rn<short4, uchar4 >}
                },
                {
                    {device::recip_rz<short , schar >, device::recip_rd<short , schar   >, device::recip_ru<short , schar >, device::recip_rn<short , schar >},
                    {device::recip_rz<short2, char2 >, device::recip_rd<short2, char2 >, device::recip_ru<short2, char2 >, device::recip_rn<short2, char2 >},
                    {device::recip_rz<short3, char3 >, device::recip_rd<short3, char3 >, device::recip_ru<short3, char3 >, device::recip_rn<short3, char3 >},
                    {device::recip_rz<short4, char4 >, device::recip_rd<short4, char4 >, device::recip_ru<short4, char4 >, device::recip_rn<short4, char4 >}
                },
                {
                    {device::recip_rz<short , ushort  >, device::recip_rd<short , ushort  >, device::recip_ru<short , ushort  >, device::recip_rn<short , ushort  >},
                    {device::recip_rz<short2, ushort2 >, device::recip_rd<short2, ushort2 >, device::recip_ru<short2, ushort2 >, device::recip_rn<short2, ushort2 >},
                    {device::recip_rz<short3, ushort3 >, device::recip_rd<short3, ushort3 >, device::recip_ru<short3, ushort3 >, device::recip_rn<short3, ushort3 >},
                    {device::recip_rz<short4, ushort4 >, device::recip_rd<short4, ushort4 >, device::recip_ru<short4, ushort4 >, device::recip_rn<short4, ushort4 >}
                },
                {
                    {device::recip_rz<short , short  >, device::recip_rd<short , short  >, device::recip_ru<short , short  >, device::recip_rn<short , short  >},
                    {device::recip_rz<short2, short2 >, device::recip_rd<short2, short2 >, device::recip_ru<short2, short2 >, device::recip_rn<short2, short2 >},
                    {device::recip_rz<short3, short3 >, device::recip_rd<short3, short3 >, device::recip_ru<short3, short3 >, device::recip_rn<short3, short3 >},
                    {device::recip_rz<short4, short4 >, device::recip_rd<short4, short4 >, device::recip_ru<short4, short4 >, device::recip_rn<short4, short4 >}
                },
                {
                    {device::recip_rz<short , int  >, device::recip_rd<short , int  >, device::recip_ru<short , int  >, device::recip_rn<short , int  >},
                    {device::recip_rz<short2, int2 >, device::recip_rd<short2, int2 >, device::recip_ru<short2, int2 >, device::recip_rn<short2, int2 >},
                    {device::recip_rz<short3, int3 >, device::recip_rd<short3, int3 >, device::recip_ru<short3, int3 >, device::recip_rn<short3, int3 >},
                    {device::recip_rz<short4, int4 >, device::recip_rd<short4, int4 >, device::recip_ru<short4, int4 >, device::recip_rn<short4, int4 >}
                },
                {
                    {device::recip_rz<short , float  >, device::recip_rd<short , float  >, device::recip_ru<short , float  >, device::recip_rn<short , float  >},
                    {device::recip_rz<short2, float2 >, device::recip_rd<short2, float2 >, device::recip_ru<short2, float2 >, device::recip_rn<short2, float2 >},
                    {device::recip_rz<short3, float3 >, device::recip_rd<short3, float3 >, device::recip_ru<short3, float3 >, device::recip_rn<short3, float3 >},
                    {device::recip_rz<short4, float4 >, device::recip_rd<short4, float4 >, device::recip_ru<short4, float4 >, device::recip_rn<short4, float4 >}
                },
                {
                    {device::recip_rz<short , double  >, device::recip_rd<short , double  >, device::recip_ru<short , double  >, device::recip_rn<short , double  >},
                    {device::recip_rz<short2, double2 >, device::recip_rd<short2, double2 >, device::recip_ru<short2, double2 >, device::recip_rn<short2, double2 >},
                    {device::recip_rz<short3, double3 >, device::recip_rd<short3, double3 >, device::recip_ru<short3, double3 >, device::recip_rn<short3, double3 >},
                    {device::recip_rz<short4, double4 >, device::recip_rd<short4, double4 >, device::recip_ru<short4, double4 >, device::recip_rn<short4, double4 >}
                }
            },
            {

                {
                    {device::recip_rz<int , uchar  >, device::recip_rd<int , uchar  >, device::recip_ru<int , uchar  >, device::recip_rn<int , uchar  >},
                    {device::recip_rz<int2, uchar2 >, device::recip_rd<int2, uchar2 >, device::recip_ru<int2, uchar2 >, device::recip_rn<int2, uchar2 >},
                    {device::recip_rz<int3, uchar3 >, device::recip_rd<int3, uchar3 >, device::recip_ru<int3, uchar3 >, device::recip_rn<int3, uchar3 >},
                    {device::recip_rz<int4, uchar4 >, device::recip_rd<int4, uchar4 >, device::recip_ru<int4, uchar4 >, device::recip_rn<int4, uchar4 >}
                },
                {
                    {device::recip_rz<int , schar >, device::recip_rd<int , schar >, device::recip_ru<int , schar >, device::recip_rn<int , schar >},
                    {device::recip_rz<int2, char2 >, device::recip_rd<int2, char2 >, device::recip_ru<int2, char2 >, device::recip_rn<int2, char2 >},
                    {device::recip_rz<int3, char3 >, device::recip_rd<int3, char3 >, device::recip_ru<int3, char3 >, device::recip_rn<int3, char3 >},
                    {device::recip_rz<int4, char4 >, device::recip_rd<int4, char4 >, device::recip_ru<int4, char4 >, device::recip_rn<int4, char4 >}
                },
                {
                    {device::recip_rz<int , ushort  >, device::recip_rd<int , ushort  >, device::recip_ru<int , ushort  >, device::recip_rn<int , ushort  >},
                    {device::recip_rz<int2, ushort2 >, device::recip_rd<int2, ushort2 >, device::recip_ru<int2, ushort2 >, device::recip_rn<int2, ushort2 >},
                    {device::recip_rz<int3, ushort3 >, device::recip_rd<int3, ushort3 >, device::recip_ru<int3, ushort3 >, device::recip_rn<int3, ushort3 >},
                    {device::recip_rz<int4, ushort4 >, device::recip_rd<int4, ushort4 >, device::recip_ru<int4, ushort4 >, device::recip_rn<int4, ushort4 >}
                },
                {
                    {device::recip_rz<int , short  >, device::recip_rd<int , short  >, device::recip_ru<int , short  >, device::recip_rn<int , short  >},
                    {device::recip_rz<int2, short2 >, device::recip_rd<int2, short2 >, device::recip_ru<int2, short2 >, device::recip_rn<int2, short2 >},
                    {device::recip_rz<int3, short3 >, device::recip_rd<int3, short3 >, device::recip_ru<int3, short3 >, device::recip_rn<int3, short3 >},
                    {device::recip_rz<int4, short4 >, device::recip_rd<int4, short4 >, device::recip_ru<int4, short4 >, device::recip_rn<int4, short4 >}
                },
                {
                    {device::recip_rz<int , int  >, device::recip_rd<int , int  >, device::recip_ru<int , int  >, device::recip_rn<int , int  >},
                    {device::recip_rz<int2, int2 >, device::recip_rd<int2, int2 >, device::recip_ru<int2, int2 >, device::recip_rn<int2, int2 >},
                    {device::recip_rz<int3, int3 >, device::recip_rd<int3, int3 >, device::recip_ru<int3, int3 >, device::recip_rn<int3, int3 >},
                    {device::recip_rz<int4, int4 >, device::recip_rd<int4, int4 >, device::recip_ru<int4, int4 >, device::recip_rn<int4, int4 >}
                },
                {
                    {device::recip_rz<int , float  >, device::recip_rd<int , float  >, device::recip_ru<int , float  >, device::recip_rn<int , float  >},
                    {device::recip_rz<int2, float2 >, device::recip_rd<int2, float2 >, device::recip_ru<int2, float2 >, device::recip_rn<int2, float2 >},
                    {device::recip_rz<int3, float3 >, device::recip_rd<int3, float3 >, device::recip_ru<int3, float3 >, device::recip_rn<int3, float3 >},
                    {device::recip_rz<int4, float4 >, device::recip_rd<int4, float4 >, device::recip_ru<int4, float4 >, device::recip_rn<int4, float4 >}
                },
                {
                    {device::recip_rz<int , double  >, device::recip_rd<int , double  >, device::recip_ru<int , double  >, device::recip_rn<int , double  >},
                    {device::recip_rz<int2, double2 >, device::recip_rd<int2, double2 >, device::recip_ru<int2, double2 >, device::recip_rn<int2, double2 >},
                    {device::recip_rz<int3, double3 >, device::recip_rd<int3, double3 >, device::recip_ru<int3, double3 >, device::recip_rn<int3, double3 >},
                    {device::recip_rz<int4, double4 >, device::recip_rd<int4, double4 >, device::recip_ru<int4, double4 >, device::recip_rn<int4, double4 >}
                }
            },
            {

                {
                    {device::recip_rz<float , uchar  >, device::recip_rd<float , uchar  >, device::recip_ru<float , uchar  >, device::recip_rn<float , uchar  >},
                    {device::recip_rz<float2, uchar2 >, device::recip_rd<float2, uchar2 >, device::recip_ru<float2, uchar2 >, device::recip_rn<float2, uchar2 >},
                    {device::recip_rz<float3, uchar3 >, device::recip_rd<float3, uchar3 >, device::recip_ru<float3, uchar3 >, device::recip_rn<float3, uchar3 >},
                    {device::recip_rz<float4, uchar4 >, device::recip_rd<float4, uchar4 >, device::recip_ru<float4, uchar4 >, device::recip_rn<float4, uchar4 >}
                },
                {
                    {device::recip_rz<float , schar >, device::recip_rd<float , schar >, device::recip_ru<float , schar >, device::recip_rn<float , schar >},
                    {device::recip_rz<float2, char2 >, device::recip_rd<float2, char2 >, device::recip_ru<float2, char2 >, device::recip_rn<float2, char2 >},
                    {device::recip_rz<float3, char3 >, device::recip_rd<float3, char3 >, device::recip_ru<float3, char3 >, device::recip_rn<float3, char3 >},
                    {device::recip_rz<float4, char4 >, device::recip_rd<float4, char4 >, device::recip_ru<float4, char4 >, device::recip_rn<float4, char4 >}
                },
                {
                    {device::recip_rz<float , ushort  >, device::recip_rd<float , ushort  >, device::recip_ru<float , ushort  >, device::recip_rn<float , ushort  >},
                    {device::recip_rz<float2, ushort2 >, device::recip_rd<float2, ushort2 >, device::recip_ru<float2, ushort2 >, device::recip_rn<float2, ushort2 >},
                    {device::recip_rz<float3, ushort3 >, device::recip_rd<float3, ushort3 >, device::recip_ru<float3, ushort3 >, device::recip_rn<float3, ushort3 >},
                    {device::recip_rz<float4, ushort4 >, device::recip_rd<float4, ushort4 >, device::recip_ru<float4, ushort4 >, device::recip_rn<float4, ushort4 >}
                },
                {
                    {device::recip_rz<float , short  >, device::recip_rd<float , short  >, device::recip_ru<float , short  >, device::recip_rn<float , short  >},
                    {device::recip_rz<float2, short2 >, device::recip_rd<float2, short2 >, device::recip_ru<float2, short2 >, device::recip_rn<float2, short2 >},
                    {device::recip_rz<float3, short3 >, device::recip_rd<float3, short3 >, device::recip_ru<float3, short3 >, device::recip_rn<float3, short3 >},
                    {device::recip_rz<float4, short4 >, device::recip_rd<float4, short4 >, device::recip_ru<float4, short4 >, device::recip_rn<float4, short4 >}
                },
                {
                    {device::recip_rz<float , int  >, device::recip_rd<float , int  >, device::recip_ru<float , int  >, device::recip_rn<float , int  >},
                    {device::recip_rz<float2, int2 >, device::recip_rd<float2, int2 >, device::recip_ru<float2, int2 >, device::recip_rn<float2, int2 >},
                    {device::recip_rz<float3, int3 >, device::recip_rd<float3, int3 >, device::recip_ru<float3, int3 >, device::recip_rn<float3, int3 >},
                    {device::recip_rz<float4, int4 >, device::recip_rd<float4, int4 >, device::recip_ru<float4, int4 >, device::recip_rn<float4, int4 >}
                },
                {
                    {device::recip_rz<float , float  >, device::recip_rd<float , float  >, device::recip_ru<float , float  >, device::recip_rn<float , float  >},
                    {device::recip_rz<float2, float2 >, device::recip_rd<float2, float2 >, device::recip_ru<float2, float2 >, device::recip_rn<float2, float2 >},
                    {device::recip_rz<float3, float3 >, device::recip_rd<float3, float3 >, device::recip_ru<float3, float3 >, device::recip_rn<float3, float3 >},
                    {device::recip_rz<float4, float4 >, device::recip_rd<float4, float4 >, device::recip_ru<float4, float4 >, device::recip_rn<float4, float4 >}
                },
                {
                    {device::recip_rz<float , double  >, device::recip_rd<float , double  >, device::recip_ru<float , double  >, device::recip_rn<float , double  >},
                    {device::recip_rz<float2, double2 >, device::recip_rd<float2, double2 >, device::recip_ru<float2, double2 >, device::recip_rn<float2, double2 >},
                    {device::recip_rz<float3, double3 >, device::recip_rd<float3, double3 >, device::recip_ru<float3, double3 >, device::recip_rn<float3, double3 >},
                    {device::recip_rz<float4, double4 >, device::recip_rd<float4, double4 >, device::recip_ru<float4, double4 >, device::recip_rn<float4, double4 >}
                }
            },
            {
                {
                    {device::recip_rz<double , uchar  >, device::recip_rd<double , uchar  >, device::recip_ru<double , uchar  >, device::recip_rn<double , uchar  >},
                    {device::recip_rz<double2, uchar2 >, device::recip_rd<double2, uchar2 >, device::recip_ru<double2, uchar2 >, device::recip_rn<double2, uchar2 >},
                    {device::recip_rz<double3, uchar3 >, device::recip_rd<double3, uchar3 >, device::recip_ru<double3, uchar3 >, device::recip_rn<double3, uchar3 >},
                    {device::recip_rz<double4, uchar4 >, device::recip_rd<double4, uchar4 >, device::recip_ru<double4, uchar4 >, device::recip_rn<double4, uchar4 >}
                },
                {
                    {device::recip_rz<double , schar >, device::recip_rd<double , schar >, device::recip_ru<double , schar >, device::recip_rn<double , schar >},
                    {device::recip_rz<double2, char2 >, device::recip_rd<double2, char2 >, device::recip_ru<double2, char2 >, device::recip_rn<double2, char2 >},
                    {device::recip_rz<double3, char3 >, device::recip_rd<double3, char3 >, device::recip_ru<double3, char3 >, device::recip_rn<double3, char3 >},
                    {device::recip_rz<double4, char4 >, device::recip_rd<double4, char4 >, device::recip_ru<double4, char4 >, device::recip_rn<double4, char4 >}
                },
                {
                    {device::recip_rz<double , ushort  >, device::recip_rd<double , ushort  >, device::recip_ru<double , ushort  >, device::recip_rn<double , ushort  >},
                    {device::recip_rz<double2, ushort2 >, device::recip_rd<double2, ushort2 >, device::recip_ru<double2, ushort2 >, device::recip_rn<double2, ushort2 >},
                    {device::recip_rz<double3, ushort3 >, device::recip_rd<double3, ushort3 >, device::recip_ru<double3, ushort3 >, device::recip_rn<double3, ushort3 >},
                    {device::recip_rz<double4, ushort4 >, device::recip_rd<double4, ushort4 >, device::recip_ru<double4, ushort4 >, device::recip_rn<double4, ushort4 >}
                },
                {
                    {device::recip_rz<double , short  >, device::recip_rd<double , short  >, device::recip_ru<double , short  >, device::recip_rn<double , short  >},
                    {device::recip_rz<double2, short2 >, device::recip_rd<double2, short2 >, device::recip_ru<double2, short2 >, device::recip_rn<double2, short2 >},
                    {device::recip_rz<double3, short3 >, device::recip_rd<double3, short3 >, device::recip_ru<double3, short3 >, device::recip_rn<double3, short3 >},
                    {device::recip_rz<double4, short4 >, device::recip_rd<double4, short4 >, device::recip_ru<double4, short4 >, device::recip_rn<double4, short4 >}
                },
                {
                    {device::recip_rz<double , int  >, device::recip_rd<double , int  >, device::recip_ru<double , int  >, device::recip_rn<double , int  >},
                    {device::recip_rz<double2, int2 >, device::recip_rd<double2, int2 >, device::recip_ru<double2, int2 >, device::recip_rn<double2, int2 >},
                    {device::recip_rz<double3, int3 >, device::recip_rd<double3, int3 >, device::recip_ru<double3, int3 >, device::recip_rn<double3, int3 >},
                    {device::recip_rz<double4, int4 >, device::recip_rd<double4, int4 >, device::recip_ru<double4, int4 >, device::recip_rn<double4, int4 >}
                },
                {
                    {device::recip_rz<double , float  >, device::recip_rd<double , float  >, device::recip_ru<double , float  >, device::recip_rn<double , float  >},
                    {device::recip_rz<double2, float2 >, device::recip_rd<double2, float2 >, device::recip_ru<double2, float2 >, device::recip_rn<double2, float2 >},
                    {device::recip_rz<double3, float3 >, device::recip_rd<double3, float3 >, device::recip_ru<double3, float3 >, device::recip_rn<double3, float3 >},
                    {device::recip_rz<double4, float4 >, device::recip_rd<double4, float4 >, device::recip_ru<double4, float4 >, device::recip_rn<double4, float4 >}
                },
                {
                    {device::recip_rz<double , double  >, device::recip_rd<double , double  >, device::recip_ru<double , double  >, device::recip_rn<double , double  >},
                    {device::recip_rz<double2, double2 >, device::recip_rd<double2, double2 >, device::recip_ru<double2, double2 >, device::recip_rn<double2, double2 >},
                    {device::recip_rz<double3, double3 >, device::recip_rd<double3, double3 >, device::recip_ru<double3, double3 >, device::recip_rn<double3, double3 >},
                    {device::recip_rz<double4, double4 >, device::recip_rd<double4, double4 >, device::recip_ru<double4, double4 >, device::recip_rn<double4, double4 >}
                }
            }
        };

        function_type fun = funcs[src.depth()-CV_32F][CV_MAT_DEPTH(dtype)][src.channels()-1][flag_norm];

        fun(src, dst, _stream);
    }
    else
    {
        GpuMat mask = _mask.getGpuMat();

        typedef void(*masked_function_type)(const GpuMat&, GpuMat&, const GpuMat&, Stream&);

        static const masked_function_type masked_funcs[7][7][4][4] = {
            {

                {
                    {device::masked_recip_rz<uchar  , uchar  >, device::masked_recip_rd<uchar  , uchar  >, device::masked_recip_ru<uchar  , uchar  >, device::masked_recip_rn<uchar  , uchar  >},
                    {device::masked_recip_rz<uchar2 , uchar2 >, device::masked_recip_rd<uchar2 , uchar2 >, device::masked_recip_ru<uchar2 , uchar2 >, device::masked_recip_rn<uchar2 , uchar2 >},
                    {device::masked_recip_rz<uchar3 , uchar3 >, device::masked_recip_rd<uchar3 , uchar3 >, device::masked_recip_ru<uchar3 , uchar3 >, device::masked_recip_rn<uchar3 , uchar3 >},
                    {device::masked_recip_rz<uchar4 , uchar4 >, device::masked_recip_rd<uchar4 , uchar4 >, device::masked_recip_ru<uchar4 , uchar4 >, device::masked_recip_rn<uchar4 , uchar4 >}
                },
                {
                    {device::masked_recip_rz<uchar  , schar >, device::masked_recip_rd<uchar  , schar >, device::masked_recip_ru<uchar  , schar >, device::masked_recip_rn<uchar  , schar >},
                    {device::masked_recip_rz<uchar2 , char2 >, device::masked_recip_rd<uchar2 , char2 >, device::masked_recip_ru<uchar2 , char2 >, device::masked_recip_rn<uchar2 , char2 >},
                    {device::masked_recip_rz<uchar3 , char3 >, device::masked_recip_rd<uchar3 , char3 >, device::masked_recip_ru<uchar3 , char3 >, device::masked_recip_rn<uchar3 , char3 >},
                    {device::masked_recip_rz<uchar4 , char4 >, device::masked_recip_rd<uchar4 , char4 >, device::masked_recip_ru<uchar4 , char4 >, device::masked_recip_rn<uchar4 , char4 >}
                },
                {
                    {device::masked_recip_rz<uchar  , ushort  >, device::masked_recip_rd<uchar  , ushort  >, device::masked_recip_ru<uchar  , ushort  >, device::masked_recip_rn<uchar  , ushort  >},
                    {device::masked_recip_rz<uchar2 , ushort2 >, device::masked_recip_rd<uchar2 , ushort2 >, device::masked_recip_ru<uchar2 , ushort2 >, device::masked_recip_rn<uchar2 , ushort2 >},
                    {device::masked_recip_rz<uchar3 , ushort3 >, device::masked_recip_rd<uchar3 , ushort3 >, device::masked_recip_ru<uchar3 , ushort3 >, device::masked_recip_rn<uchar3 , ushort3 >},
                    {device::masked_recip_rz<uchar4 , ushort4 >, device::masked_recip_rd<uchar4 , ushort4 >, device::masked_recip_ru<uchar4 , ushort4 >, device::masked_recip_rn<uchar4 , ushort4 >}
                },
                {
                    {device::masked_recip_rz<uchar  , short  >, device::masked_recip_rd<uchar  , short  >, device::masked_recip_ru<uchar  , short  >, device::masked_recip_rn<uchar  , short  >},
                    {device::masked_recip_rz<uchar2 , short2 >, device::masked_recip_rd<uchar2 , short2 >, device::masked_recip_ru<uchar2 , short2 >, device::masked_recip_rn<uchar2 , short2 >},
                    {device::masked_recip_rz<uchar3 , short3 >, device::masked_recip_rd<uchar3 , short3 >, device::masked_recip_ru<uchar3 , short3 >, device::masked_recip_rn<uchar3 , short3 >},
                    {device::masked_recip_rz<uchar4 , short4 >, device::masked_recip_rd<uchar4 , short4 >, device::masked_recip_ru<uchar4 , short4 >, device::masked_recip_rn<uchar4 , short4 >}
                },
                {
                    {device::masked_recip_rz<uchar  , int  >, device::masked_recip_rd<uchar  , int  >, device::masked_recip_ru<uchar ,  int  >, device::masked_recip_rn<uchar  , int  >},
                    {device::masked_recip_rz<uchar2 , int2 >, device::masked_recip_rd<uchar2 , int2 >, device::masked_recip_ru<uchar2 , int2 >, device::masked_recip_rn<uchar2 , int2 >},
                    {device::masked_recip_rz<uchar3 , int3 >, device::masked_recip_rd<uchar3 , int3 >, device::masked_recip_ru<uchar3 , int3 >, device::masked_recip_rn<uchar3 , int3 >},
                    {device::masked_recip_rz<uchar4 , int4 >, device::masked_recip_rd<uchar4 , int4 >, device::masked_recip_ru<uchar4 , int4 >, device::masked_recip_rn<uchar4 , int4 >}
                },
                {
                    {device::masked_recip_rz<uchar  , float  >, device::masked_recip_rd<uchar  , float  >, device::masked_recip_ru<uchar  , float  >, device::masked_recip_rn<uchar  , float  >},
                    {device::masked_recip_rz<uchar2 , float2 >, device::masked_recip_rd<uchar2 , float2 >, device::masked_recip_ru<uchar2 , float2 >, device::masked_recip_rn<uchar2 , float2 >},
                    {device::masked_recip_rz<uchar3 , float3 >, device::masked_recip_rd<uchar3 , float3 >, device::masked_recip_ru<uchar3 , float3 >, device::masked_recip_rn<uchar3 , float3 >},
                    {device::masked_recip_rz<uchar4 , float4 >, device::masked_recip_rd<uchar4 , float4 >, device::masked_recip_ru<uchar4 , float4 >, device::masked_recip_rn<uchar4 , float4 >}
                },
                {
                    {device::masked_recip_rz<uchar  , double  >, device::masked_recip_rd<uchar  , double  >, device::masked_recip_ru<uchar , double  >, device::masked_recip_rn< uchar  , double  >},
                    {device::masked_recip_rz<uchar2 , double2 >, device::masked_recip_rd<uchar2 , double2 >, device::masked_recip_ru<uchar2 , double2 >, device::masked_recip_rn<uchar2 , double2 >},
                    {device::masked_recip_rz<uchar3 , double3 >, device::masked_recip_rd<uchar3 , double3 >, device::masked_recip_ru<uchar3 , double3 >, device::masked_recip_rn<uchar3 , double3 >},
                    {device::masked_recip_rz<uchar4 , double4 >, device::masked_recip_rd<uchar4 , double4 >, device::masked_recip_ru<uchar4 , double4 >, device::masked_recip_rn<uchar4 , double4 >}
                }
            },
            {

                {
                    {device::masked_recip_rz<schar , uchar >, device::masked_recip_rd<schar  , uchar >, device::masked_recip_ru<schar  , uchar  >, device::masked_recip_rn<schar , uchar  >},
                    {device::masked_recip_rz<char2 , uchar2 >, device::masked_recip_rd<char2 , uchar2 >, device::masked_recip_ru<char2 , uchar2 >, device::masked_recip_rn<char2 , uchar2 >},
                    {device::masked_recip_rz<char3 , uchar3 >, device::masked_recip_rd<char3 , uchar3 >, device::masked_recip_ru<char3 , uchar3 >, device::masked_recip_rn<char3 , uchar3 >},
                    {device::masked_recip_rz<char4 , uchar4 >, device::masked_recip_rd<char4 , uchar4 >, device::masked_recip_ru<char4 , uchar4 >, device::masked_recip_rn<char4 , uchar4 >}
                },
                {
                    {device::masked_recip_rz<schar , schar >, device::masked_recip_rd<schar , schar >, device::masked_recip_ru<schar , schar >, device::masked_recip_rn<schar , schar >},
                    {device::masked_recip_rz<char2 , char2 >, device::masked_recip_rd<char2 , char2 >, device::masked_recip_ru<char2 , char2 >, device::masked_recip_rn<char2 , char2 >},
                    {device::masked_recip_rz<char3 , char3 >, device::masked_recip_rd<char3 , char3 >, device::masked_recip_ru<char3 , char3 >, device::masked_recip_rn<char3 , char3 >},
                    {device::masked_recip_rz<char4 , char4 >, device::masked_recip_rd<char4 , char4 >, device::masked_recip_ru<char4 , char4 >, device::masked_recip_rn<char4 , char4 >}
                },
                {
                    {device::masked_recip_rz<schar , ushort  >, device::masked_recip_rd<schar , ushort  >, device::masked_recip_ru<schar,  ushort  >, device::masked_recip_rn<schar , ushort  >},
                    {device::masked_recip_rz<char2 , ushort2 >, device::masked_recip_rd<char2 , ushort2 >, device::masked_recip_ru<char2 , ushort2 >, device::masked_recip_rn<char2 , ushort2 >},
                    {device::masked_recip_rz<char3 , ushort3 >, device::masked_recip_rd<char3 , ushort3 >, device::masked_recip_ru<char3 , ushort3 >, device::masked_recip_rn<char3 , ushort3 >},
                    {device::masked_recip_rz<char4 , ushort4 >, device::masked_recip_rd<char4 , ushort4 >, device::masked_recip_ru<char4 , ushort4 >, device::masked_recip_rn<char4 , ushort4 >}
                },
                {
                    {device::masked_recip_rz<schar , short  >, device::masked_recip_rd<schar , short  >, device::masked_recip_ru<schar , short  >, device::masked_recip_rn<schar , short  >},
                    {device::masked_recip_rz<char2 , short2 >, device::masked_recip_rd<char2 , short2 >, device::masked_recip_ru<char2 , short2 >, device::masked_recip_rn<char2 , short2 >},
                    {device::masked_recip_rz<char3 , short3 >, device::masked_recip_rd<char3 , short3 >, device::masked_recip_ru<char3 , short3 >, device::masked_recip_rn<char3 , short3 >},
                    {device::masked_recip_rz<char4 , short4 >, device::masked_recip_rd<char4 , short4 >, device::masked_recip_ru<char4 , short4 >, device::masked_recip_rn<char4 , short4 >}
                },
                {
                    {device::masked_recip_rz<schar , int  >, device::masked_recip_rd<schar , int  >, device::masked_recip_ru<schar , int  >, device::masked_recip_rn<schar , int  >},
                    {device::masked_recip_rz<char2 , int2 >, device::masked_recip_rd<char2 , int2 >, device::masked_recip_ru<char2 , int2 >, device::masked_recip_rn<char2 , int2 >},
                    {device::masked_recip_rz<char3 , int3 >, device::masked_recip_rd<char3 , int3 >, device::masked_recip_ru<char3 , int3 >, device::masked_recip_rn<char3 , int3 >},
                    {device::masked_recip_rz<char4 , int4 >, device::masked_recip_rd<char4 , int4 >, device::masked_recip_ru<char4 , int4 >, device::masked_recip_rn<char4 , int4 >}
                },
                {
                    {device::masked_recip_rz<schar , float  >, device::masked_recip_rd<schar , float  >, device::masked_recip_ru<schar , float  >, device::masked_recip_rn<schar , float  >},
                    {device::masked_recip_rz<char2 , float2 >, device::masked_recip_rd<char2 , float2 >, device::masked_recip_ru<char2 , float2 >, device::masked_recip_rn<char2 , float2 >},
                    {device::masked_recip_rz<char3 , float3 >, device::masked_recip_rd<char3 , float3 >, device::masked_recip_ru<char3 , float3 >, device::masked_recip_rn<char3 , float3 >},
                    {device::masked_recip_rz<char4 , float4 >, device::masked_recip_rd<char4 , float4 >, device::masked_recip_ru<char4 , float4 >, device::masked_recip_rn<char4 , float4 >}
                },
                {
                    {device::masked_recip_rz<schar , double  >, device::masked_recip_rd<schar , double  >, device::masked_recip_ru<schar , double  >, device::masked_recip_rn<schar, double   >},
                    {device::masked_recip_rz<char2 , double2 >, device::masked_recip_rd<char2 , double2 >, device::masked_recip_ru<char2 , double2 >, device::masked_recip_rn<char2 , double2 >},
                    {device::masked_recip_rz<char3 , double3 >, device::masked_recip_rd<char3 , double3 >, device::masked_recip_ru<char3 , double3 >, device::masked_recip_rn<char3 , double3 >},
                    {device::masked_recip_rz<char4 , double4 >, device::masked_recip_rd<char4 , double4 >, device::masked_recip_ru<char4 , double4 >, device::masked_recip_rn<char4 , double4 >}
                }
            },
            {

                {
                    {device::masked_recip_rz<ushort , uchar  >, device::masked_recip_rd<ushort , uchar  >, device::masked_recip_ru<ushort , uchar  >, device::masked_recip_rn<ushort , uchar  >},
                    {device::masked_recip_rz<ushort2, uchar2 >, device::masked_recip_rd<ushort3, uchar3 >, device::masked_recip_ru<ushort2, uchar2 >, device::masked_recip_rn<ushort2, uchar2 >},
                    {device::masked_recip_rz<ushort3, uchar3 >, device::masked_recip_rd<ushort3, uchar3 >, device::masked_recip_ru<ushort3, uchar3 >, device::masked_recip_rn<ushort3, uchar3 >},
                    {device::masked_recip_rz<ushort4, uchar4 >, device::masked_recip_rd<ushort4, uchar4 >, device::masked_recip_ru<ushort4, uchar4 >, device::masked_recip_rn<ushort4, uchar4 >}
                },
                {
                    {device::masked_recip_rz<ushort , schar >, device::masked_recip_rd<ushort , schar >, device::masked_recip_ru<ushort , schar >, device::masked_recip_rn<ushort , schar >},
                    {device::masked_recip_rz<ushort2, char2 >, device::masked_recip_rd<ushort2, char2 >, device::masked_recip_ru<ushort2, char2 >, device::masked_recip_rn<ushort2, char2 >},
                    {device::masked_recip_rz<ushort3, char3 >, device::masked_recip_rd<ushort3, char3 >, device::masked_recip_ru<ushort3, char3 >, device::masked_recip_rn<ushort3, char3 >},
                    {device::masked_recip_rz<ushort4, char4 >, device::masked_recip_rd<ushort4, char4 >, device::masked_recip_ru<ushort4, char4 >, device::masked_recip_rn<ushort4, char4 >}
                },
                {
                    {device::masked_recip_rz<ushort , ushort  >, device::masked_recip_rd<ushort , ushort  >, device::masked_recip_ru<ushort , ushort  >, device::masked_recip_rn<ushort , ushort  >},
                    {device::masked_recip_rz<ushort2, ushort2 >, device::masked_recip_rd<ushort2, ushort2 >, device::masked_recip_ru<ushort2, ushort2 >, device::masked_recip_rn<ushort2, ushort2 >},
                    {device::masked_recip_rz<ushort3, ushort3 >, device::masked_recip_rd<ushort3, ushort3 >, device::masked_recip_ru<ushort3, ushort3 >, device::masked_recip_rn<ushort3, ushort3 >},
                    {device::masked_recip_rz<ushort4, ushort4 >, device::masked_recip_rd<ushort4, ushort4 >, device::masked_recip_ru<ushort4, ushort4 >, device::masked_recip_rn<ushort4, ushort4 >}
                },
                {
                    {device::masked_recip_rz<ushort , short  >, device::masked_recip_rd<ushort , short  >, device::masked_recip_ru<ushort , short  >, device::masked_recip_rn<ushort , short  >},
                    {device::masked_recip_rz<ushort2, short2 >, device::masked_recip_rd<ushort2, short2 >, device::masked_recip_ru<ushort2, short2 >, device::masked_recip_rn<ushort2, short2 >},
                    {device::masked_recip_rz<ushort3, short3 >, device::masked_recip_rd<ushort3, short3 >, device::masked_recip_ru<ushort3, short3 >, device::masked_recip_rn<ushort3, short3 >},
                    {device::masked_recip_rz<ushort4, short4 >, device::masked_recip_rd<ushort4, short4 >, device::masked_recip_ru<ushort4, short4 >, device::masked_recip_rn<ushort4, short4 >}
                },
                {
                    {device::masked_recip_rz<ushort , int  >, device::masked_recip_rd<ushort , int  >, device::masked_recip_ru<ushort , int  >, device::masked_recip_rn<ushort , int  >},
                    {device::masked_recip_rz<ushort2, int2 >, device::masked_recip_rd<ushort2, int2 >, device::masked_recip_ru<ushort2, int2 >, device::masked_recip_rn<ushort2, int2 >},
                    {device::masked_recip_rz<ushort3, int3 >, device::masked_recip_rd<ushort3, int3 >, device::masked_recip_ru<ushort3, int3 >, device::masked_recip_rn<ushort3, int3 >},
                    {device::masked_recip_rz<ushort4, int4 >, device::masked_recip_rd<ushort4, int4 >, device::masked_recip_ru<ushort4, int4 >, device::masked_recip_rn<ushort4, int4 >}
                },
                {
                    {device::masked_recip_rz<ushort , float  >, device::masked_recip_rd<ushort , float  >, device::masked_recip_ru<ushort , float  >, device::masked_recip_rn<ushort , float  >},
                    {device::masked_recip_rz<ushort2, float2 >, device::masked_recip_rd<ushort2, float2 >, device::masked_recip_ru<ushort2, float2 >, device::masked_recip_rn<ushort2, float2 >},
                    {device::masked_recip_rz<ushort3, float3 >, device::masked_recip_rd<ushort3, float3 >, device::masked_recip_ru<ushort3, float3 >, device::masked_recip_rn<ushort3, float3 >},
                    {device::masked_recip_rz<ushort4, float4 >, device::masked_recip_rd<ushort4, float4 >, device::masked_recip_ru<ushort4, float4 >, device::masked_recip_rn<ushort4, float4 >}
                },
                {
                    {device::masked_recip_rz<ushort , double  >, device::masked_recip_rd<ushort , double  >, device::masked_recip_ru<ushort , double  >, device::masked_recip_rn<ushort , double  >},
                    {device::masked_recip_rz<ushort2, double2 >, device::masked_recip_rd<ushort2, double2 >, device::masked_recip_ru<ushort2, double2 >, device::masked_recip_rn<ushort2, double2 >},
                    {device::masked_recip_rz<ushort3, double3 >, device::masked_recip_rd<ushort3, double3 >, device::masked_recip_ru<ushort3, double3 >, device::masked_recip_rn<ushort3, double3 >},
                    {device::masked_recip_rz<ushort4, double4 >, device::masked_recip_rd<ushort4, double4 >, device::masked_recip_ru<ushort4, double4 >, device::masked_recip_rn<ushort4, double4 >}
                }
            },
            {

                {
                    {device::masked_recip_rz<short , uchar  >, device::masked_recip_rd<short , uchar  >, device::masked_recip_ru<short , uchar  >, device::masked_recip_rn<short , uchar  >},
                    {device::masked_recip_rz<short2, uchar2 >, device::masked_recip_rd<short2, uchar2 >, device::masked_recip_ru<short2, uchar2 >, device::masked_recip_rn<short2, uchar2 >},
                    {device::masked_recip_rz<short3, uchar3 >, device::masked_recip_rd<short3, uchar3 >, device::masked_recip_ru<short3, uchar3 >, device::masked_recip_rn<short3, uchar3 >},
                    {device::masked_recip_rz<short4, uchar4 >, device::masked_recip_rd<short4, uchar4 >, device::masked_recip_ru<short4, uchar4 >, device::masked_recip_rn<short4, uchar4 >}
                },
                {
                    {device::masked_recip_rz<short , schar >, device::masked_recip_rd<short , schar   >, device::masked_recip_ru<short , schar >, device::masked_recip_rn<short , schar >},
                    {device::masked_recip_rz<short2, char2 >, device::masked_recip_rd<short2, char2 >, device::masked_recip_ru<short2, char2 >, device::masked_recip_rn<short2, char2 >},
                    {device::masked_recip_rz<short3, char3 >, device::masked_recip_rd<short3, char3 >, device::masked_recip_ru<short3, char3 >, device::masked_recip_rn<short3, char3 >},
                    {device::masked_recip_rz<short4, char4 >, device::masked_recip_rd<short4, char4 >, device::masked_recip_ru<short4, char4 >, device::masked_recip_rn<short4, char4 >}
                },
                {
                    {device::masked_recip_rz<short , ushort  >, device::masked_recip_rd<short , ushort  >, device::masked_recip_ru<short , ushort  >, device::masked_recip_rn<short , ushort  >},
                    {device::masked_recip_rz<short2, ushort2 >, device::masked_recip_rd<short2, ushort2 >, device::masked_recip_ru<short2, ushort2 >, device::masked_recip_rn<short2, ushort2 >},
                    {device::masked_recip_rz<short3, ushort3 >, device::masked_recip_rd<short3, ushort3 >, device::masked_recip_ru<short3, ushort3 >, device::masked_recip_rn<short3, ushort3 >},
                    {device::masked_recip_rz<short4, ushort4 >, device::masked_recip_rd<short4, ushort4 >, device::masked_recip_ru<short4, ushort4 >, device::masked_recip_rn<short4, ushort4 >}
                },
                {
                    {device::masked_recip_rz<short , short  >, device::masked_recip_rd<short , short  >, device::masked_recip_ru<short , short  >, device::masked_recip_rn<short , short  >},
                    {device::masked_recip_rz<short2, short2 >, device::masked_recip_rd<short2, short2 >, device::masked_recip_ru<short2, short2 >, device::masked_recip_rn<short2, short2 >},
                    {device::masked_recip_rz<short3, short3 >, device::masked_recip_rd<short3, short3 >, device::masked_recip_ru<short3, short3 >, device::masked_recip_rn<short3, short3 >},
                    {device::masked_recip_rz<short4, short4 >, device::masked_recip_rd<short4, short4 >, device::masked_recip_ru<short4, short4 >, device::masked_recip_rn<short4, short4 >}
                },
                {
                    {device::masked_recip_rz<short , int  >, device::masked_recip_rd<short , int  >, device::masked_recip_ru<short , int  >, device::masked_recip_rn<short , int  >},
                    {device::masked_recip_rz<short2, int2 >, device::masked_recip_rd<short2, int2 >, device::masked_recip_ru<short2, int2 >, device::masked_recip_rn<short2, int2 >},
                    {device::masked_recip_rz<short3, int3 >, device::masked_recip_rd<short3, int3 >, device::masked_recip_ru<short3, int3 >, device::masked_recip_rn<short3, int3 >},
                    {device::masked_recip_rz<short4, int4 >, device::masked_recip_rd<short4, int4 >, device::masked_recip_ru<short4, int4 >, device::masked_recip_rn<short4, int4 >}
                },
                {
                    {device::masked_recip_rz<short , float  >, device::masked_recip_rd<short , float  >, device::masked_recip_ru<short , float  >, device::masked_recip_rn<short , float  >},
                    {device::masked_recip_rz<short2, float2 >, device::masked_recip_rd<short2, float2 >, device::masked_recip_ru<short2, float2 >, device::masked_recip_rn<short2, float2 >},
                    {device::masked_recip_rz<short3, float3 >, device::masked_recip_rd<short3, float3 >, device::masked_recip_ru<short3, float3 >, device::masked_recip_rn<short3, float3 >},
                    {device::masked_recip_rz<short4, float4 >, device::masked_recip_rd<short4, float4 >, device::masked_recip_ru<short4, float4 >, device::masked_recip_rn<short4, float4 >}
                },
                {
                    {device::masked_recip_rz<short , double  >, device::masked_recip_rd<short , double  >, device::masked_recip_ru<short , double  >, device::masked_recip_rn<short , double  >},
                    {device::masked_recip_rz<short2, double2 >, device::masked_recip_rd<short2, double2 >, device::masked_recip_ru<short2, double2 >, device::masked_recip_rn<short2, double2 >},
                    {device::masked_recip_rz<short3, double3 >, device::masked_recip_rd<short3, double3 >, device::masked_recip_ru<short3, double3 >, device::masked_recip_rn<short3, double3 >},
                    {device::masked_recip_rz<short4, double4 >, device::masked_recip_rd<short4, double4 >, device::masked_recip_ru<short4, double4 >, device::masked_recip_rn<short4, double4 >}
                }
            },
            {

                {
                    {device::masked_recip_rz<int , uchar  >, device::masked_recip_rd<int , uchar  >, device::masked_recip_ru<int , uchar  >, device::masked_recip_rn<int , uchar  >},
                    {device::masked_recip_rz<int2, uchar2 >, device::masked_recip_rd<int2, uchar2 >, device::masked_recip_ru<int2, uchar2 >, device::masked_recip_rn<int2, uchar2 >},
                    {device::masked_recip_rz<int3, uchar3 >, device::masked_recip_rd<int3, uchar3 >, device::masked_recip_ru<int3, uchar3 >, device::masked_recip_rn<int3, uchar3 >},
                    {device::masked_recip_rz<int4, uchar4 >, device::masked_recip_rd<int4, uchar4 >, device::masked_recip_ru<int4, uchar4 >, device::masked_recip_rn<int4, uchar4 >}
                },
                {
                    {device::masked_recip_rz<int , schar >, device::masked_recip_rd<int , schar >, device::masked_recip_ru<int , schar >, device::masked_recip_rn<int , schar >},
                    {device::masked_recip_rz<int2, char2 >, device::masked_recip_rd<int2, char2 >, device::masked_recip_ru<int2, char2 >, device::masked_recip_rn<int2, char2 >},
                    {device::masked_recip_rz<int3, char3 >, device::masked_recip_rd<int3, char3 >, device::masked_recip_ru<int3, char3 >, device::masked_recip_rn<int3, char3 >},
                    {device::masked_recip_rz<int4, char4 >, device::masked_recip_rd<int4, char4 >, device::masked_recip_ru<int4, char4 >, device::masked_recip_rn<int4, char4 >}
                },
                {
                    {device::masked_recip_rz<int , ushort  >, device::masked_recip_rd<int , ushort  >, device::masked_recip_ru<int , ushort  >, device::masked_recip_rn<int , ushort  >},
                    {device::masked_recip_rz<int2, ushort2 >, device::masked_recip_rd<int2, ushort2 >, device::masked_recip_ru<int2, ushort2 >, device::masked_recip_rn<int2, ushort2 >},
                    {device::masked_recip_rz<int3, ushort3 >, device::masked_recip_rd<int3, ushort3 >, device::masked_recip_ru<int3, ushort3 >, device::masked_recip_rn<int3, ushort3 >},
                    {device::masked_recip_rz<int4, ushort4 >, device::masked_recip_rd<int4, ushort4 >, device::masked_recip_ru<int4, ushort4 >, device::masked_recip_rn<int4, ushort4 >}
                },
                {
                    {device::masked_recip_rz<int , short  >, device::masked_recip_rd<int , short  >, device::masked_recip_ru<int , short  >, device::masked_recip_rn<int , short  >},
                    {device::masked_recip_rz<int2, short2 >, device::masked_recip_rd<int2, short2 >, device::masked_recip_ru<int2, short2 >, device::masked_recip_rn<int2, short2 >},
                    {device::masked_recip_rz<int3, short3 >, device::masked_recip_rd<int3, short3 >, device::masked_recip_ru<int3, short3 >, device::masked_recip_rn<int3, short3 >},
                    {device::masked_recip_rz<int4, short4 >, device::masked_recip_rd<int4, short4 >, device::masked_recip_ru<int4, short4 >, device::masked_recip_rn<int4, short4 >}
                },
                {
                    {device::masked_recip_rz<int , int  >, device::masked_recip_rd<int , int  >, device::masked_recip_ru<int , int  >, device::masked_recip_rn<int , int  >},
                    {device::masked_recip_rz<int2, int2 >, device::masked_recip_rd<int2, int2 >, device::masked_recip_ru<int2, int2 >, device::masked_recip_rn<int2, int2 >},
                    {device::masked_recip_rz<int3, int3 >, device::masked_recip_rd<int3, int3 >, device::masked_recip_ru<int3, int3 >, device::masked_recip_rn<int3, int3 >},
                    {device::masked_recip_rz<int4, int4 >, device::masked_recip_rd<int4, int4 >, device::masked_recip_ru<int4, int4 >, device::masked_recip_rn<int4, int4 >}
                },
                {
                    {device::masked_recip_rz<int , float  >, device::masked_recip_rd<int , float  >, device::masked_recip_ru<int , float  >, device::masked_recip_rn<int , float  >},
                    {device::masked_recip_rz<int2, float2 >, device::masked_recip_rd<int2, float2 >, device::masked_recip_ru<int2, float2 >, device::masked_recip_rn<int2, float2 >},
                    {device::masked_recip_rz<int3, float3 >, device::masked_recip_rd<int3, float3 >, device::masked_recip_ru<int3, float3 >, device::masked_recip_rn<int3, float3 >},
                    {device::masked_recip_rz<int4, float4 >, device::masked_recip_rd<int4, float4 >, device::masked_recip_ru<int4, float4 >, device::masked_recip_rn<int4, float4 >}
                },
                {
                    {device::masked_recip_rz<int , double  >, device::masked_recip_rd<int , double  >, device::masked_recip_ru<int , double  >, device::masked_recip_rn<int , double  >},
                    {device::masked_recip_rz<int2, double2 >, device::masked_recip_rd<int2, double2 >, device::masked_recip_ru<int2, double2 >, device::masked_recip_rn<int2, double2 >},
                    {device::masked_recip_rz<int3, double3 >, device::masked_recip_rd<int3, double3 >, device::masked_recip_ru<int3, double3 >, device::masked_recip_rn<int3, double3 >},
                    {device::masked_recip_rz<int4, double4 >, device::masked_recip_rd<int4, double4 >, device::masked_recip_ru<int4, double4 >, device::masked_recip_rn<int4, double4 >}
                }
            },
            {

                {
                    {device::masked_recip_rz<float , uchar  >, device::masked_recip_rd<float , uchar  >, device::masked_recip_ru<float , uchar  >, device::masked_recip_rn<float , uchar  >},
                    {device::masked_recip_rz<float2, uchar2 >, device::masked_recip_rd<float2, uchar2 >, device::masked_recip_ru<float2, uchar2 >, device::masked_recip_rn<float2, uchar2 >},
                    {device::masked_recip_rz<float3, uchar3 >, device::masked_recip_rd<float3, uchar3 >, device::masked_recip_ru<float3, uchar3 >, device::masked_recip_rn<float3, uchar3 >},
                    {device::masked_recip_rz<float4, uchar4 >, device::masked_recip_rd<float4, uchar4 >, device::masked_recip_ru<float4, uchar4 >, device::masked_recip_rn<float4, uchar4 >}
                },
                {
                    {device::masked_recip_rz<float , schar >, device::masked_recip_rd<float , schar >, device::masked_recip_ru<float , schar >, device::masked_recip_rn<float , schar >},
                    {device::masked_recip_rz<float2, char2 >, device::masked_recip_rd<float2, char2 >, device::masked_recip_ru<float2, char2 >, device::masked_recip_rn<float2, char2 >},
                    {device::masked_recip_rz<float3, char3 >, device::masked_recip_rd<float3, char3 >, device::masked_recip_ru<float3, char3 >, device::masked_recip_rn<float3, char3 >},
                    {device::masked_recip_rz<float4, char4 >, device::masked_recip_rd<float4, char4 >, device::masked_recip_ru<float4, char4 >, device::masked_recip_rn<float4, char4 >}
                },
                {
                    {device::masked_recip_rz<float , ushort  >, device::masked_recip_rd<float , ushort  >, device::masked_recip_ru<float , ushort  >, device::masked_recip_rn<float , ushort  >},
                    {device::masked_recip_rz<float2, ushort2 >, device::masked_recip_rd<float2, ushort2 >, device::masked_recip_ru<float2, ushort2 >, device::masked_recip_rn<float2, ushort2 >},
                    {device::masked_recip_rz<float3, ushort3 >, device::masked_recip_rd<float3, ushort3 >, device::masked_recip_ru<float3, ushort3 >, device::masked_recip_rn<float3, ushort3 >},
                    {device::masked_recip_rz<float4, ushort4 >, device::masked_recip_rd<float4, ushort4 >, device::masked_recip_ru<float4, ushort4 >, device::masked_recip_rn<float4, ushort4 >}
                },
                {
                    {device::masked_recip_rz<float , short  >, device::masked_recip_rd<float , short  >, device::masked_recip_ru<float , short  >, device::masked_recip_rn<float , short  >},
                    {device::masked_recip_rz<float2, short2 >, device::masked_recip_rd<float2, short2 >, device::masked_recip_ru<float2, short2 >, device::masked_recip_rn<float2, short2 >},
                    {device::masked_recip_rz<float3, short3 >, device::masked_recip_rd<float3, short3 >, device::masked_recip_ru<float3, short3 >, device::masked_recip_rn<float3, short3 >},
                    {device::masked_recip_rz<float4, short4 >, device::masked_recip_rd<float4, short4 >, device::masked_recip_ru<float4, short4 >, device::masked_recip_rn<float4, short4 >}
                },
                {
                    {device::masked_recip_rz<float , int  >, device::masked_recip_rd<float , int  >, device::masked_recip_ru<float , int  >, device::masked_recip_rn<float , int  >},
                    {device::masked_recip_rz<float2, int2 >, device::masked_recip_rd<float2, int2 >, device::masked_recip_ru<float2, int2 >, device::masked_recip_rn<float2, int2 >},
                    {device::masked_recip_rz<float3, int3 >, device::masked_recip_rd<float3, int3 >, device::masked_recip_ru<float3, int3 >, device::masked_recip_rn<float3, int3 >},
                    {device::masked_recip_rz<float4, int4 >, device::masked_recip_rd<float4, int4 >, device::masked_recip_ru<float4, int4 >, device::masked_recip_rn<float4, int4 >}
                },
                {
                    {device::masked_recip_rz<float , float  >, device::masked_recip_rd<float , float  >, device::masked_recip_ru<float , float  >, device::masked_recip_rn<float , float  >},
                    {device::masked_recip_rz<float2, float2 >, device::masked_recip_rd<float2, float2 >, device::masked_recip_ru<float2, float2 >, device::masked_recip_rn<float2, float2 >},
                    {device::masked_recip_rz<float3, float3 >, device::masked_recip_rd<float3, float3 >, device::masked_recip_ru<float3, float3 >, device::masked_recip_rn<float3, float3 >},
                    {device::masked_recip_rz<float4, float4 >, device::masked_recip_rd<float4, float4 >, device::masked_recip_ru<float4, float4 >, device::masked_recip_rn<float4, float4 >}
                },
                {
                    {device::masked_recip_rz<float , double  >, device::masked_recip_rd<float , double  >, device::masked_recip_ru<float , double  >, device::masked_recip_rn<float , double  >},
                    {device::masked_recip_rz<float2, double2 >, device::masked_recip_rd<float2, double2 >, device::masked_recip_ru<float2, double2 >, device::masked_recip_rn<float2, double2 >},
                    {device::masked_recip_rz<float3, double3 >, device::masked_recip_rd<float3, double3 >, device::masked_recip_ru<float3, double3 >, device::masked_recip_rn<float3, double3 >},
                    {device::masked_recip_rz<float4, double4 >, device::masked_recip_rd<float4, double4 >, device::masked_recip_ru<float4, double4 >, device::masked_recip_rn<float4, double4 >}
                }
            },
            {
                {
                    {device::masked_recip_rz<double , uchar  >, device::masked_recip_rd<double , uchar  >, device::masked_recip_ru<double , uchar  >, device::masked_recip_rn<double , uchar  >},
                    {device::masked_recip_rz<double2, uchar2 >, device::masked_recip_rd<double2, uchar2 >, device::masked_recip_ru<double2, uchar2 >, device::masked_recip_rn<double2, uchar2 >},
                    {device::masked_recip_rz<double3, uchar3 >, device::masked_recip_rd<double3, uchar3 >, device::masked_recip_ru<double3, uchar3 >, device::masked_recip_rn<double3, uchar3 >},
                    {device::masked_recip_rz<double4, uchar4 >, device::masked_recip_rd<double4, uchar4 >, device::masked_recip_ru<double4, uchar4 >, device::masked_recip_rn<double4, uchar4 >}
                },
                {
                    {device::masked_recip_rz<double , schar >, device::masked_recip_rd<double , schar >, device::masked_recip_ru<double , schar >, device::masked_recip_rn<double , schar >},
                    {device::masked_recip_rz<double2, char2 >, device::masked_recip_rd<double2, char2 >, device::masked_recip_ru<double2, char2 >, device::masked_recip_rn<double2, char2 >},
                    {device::masked_recip_rz<double3, char3 >, device::masked_recip_rd<double3, char3 >, device::masked_recip_ru<double3, char3 >, device::masked_recip_rn<double3, char3 >},
                    {device::masked_recip_rz<double4, char4 >, device::masked_recip_rd<double4, char4 >, device::masked_recip_ru<double4, char4 >, device::masked_recip_rn<double4, char4 >}
                },
                {
                    {device::masked_recip_rz<double , ushort  >, device::masked_recip_rd<double , ushort  >, device::masked_recip_ru<double , ushort  >, device::masked_recip_rn<double , ushort  >},
                    {device::masked_recip_rz<double2, ushort2 >, device::masked_recip_rd<double2, ushort2 >, device::masked_recip_ru<double2, ushort2 >, device::masked_recip_rn<double2, ushort2 >},
                    {device::masked_recip_rz<double3, ushort3 >, device::masked_recip_rd<double3, ushort3 >, device::masked_recip_ru<double3, ushort3 >, device::masked_recip_rn<double3, ushort3 >},
                    {device::masked_recip_rz<double4, ushort4 >, device::masked_recip_rd<double4, ushort4 >, device::masked_recip_ru<double4, ushort4 >, device::masked_recip_rn<double4, ushort4 >}
                },
                {
                    {device::masked_recip_rz<double , short  >, device::masked_recip_rd<double , short  >, device::masked_recip_ru<double , short  >, device::masked_recip_rn<double , short  >},
                    {device::masked_recip_rz<double2, short2 >, device::masked_recip_rd<double2, short2 >, device::masked_recip_ru<double2, short2 >, device::masked_recip_rn<double2, short2 >},
                    {device::masked_recip_rz<double3, short3 >, device::masked_recip_rd<double3, short3 >, device::masked_recip_ru<double3, short3 >, device::masked_recip_rn<double3, short3 >},
                    {device::masked_recip_rz<double4, short4 >, device::masked_recip_rd<double4, short4 >, device::masked_recip_ru<double4, short4 >, device::masked_recip_rn<double4, short4 >}
                },
                {
                    {device::masked_recip_rz<double , int  >, device::masked_recip_rd<double , int  >, device::masked_recip_ru<double , int  >, device::masked_recip_rn<double , int  >},
                    {device::masked_recip_rz<double2, int2 >, device::masked_recip_rd<double2, int2 >, device::masked_recip_ru<double2, int2 >, device::masked_recip_rn<double2, int2 >},
                    {device::masked_recip_rz<double3, int3 >, device::masked_recip_rd<double3, int3 >, device::masked_recip_ru<double3, int3 >, device::masked_recip_rn<double3, int3 >},
                    {device::masked_recip_rz<double4, int4 >, device::masked_recip_rd<double4, int4 >, device::masked_recip_ru<double4, int4 >, device::masked_recip_rn<double4, int4 >}
                },
                {
                    {device::masked_recip_rz<double , float  >, device::masked_recip_rd<double , float  >, device::masked_recip_ru<double , float  >, device::masked_recip_rn<double , float  >},
                    {device::masked_recip_rz<double2, float2 >, device::masked_recip_rd<double2, float2 >, device::masked_recip_ru<double2, float2 >, device::masked_recip_rn<double2, float2 >},
                    {device::masked_recip_rz<double3, float3 >, device::masked_recip_rd<double3, float3 >, device::masked_recip_ru<double3, float3 >, device::masked_recip_rn<double3, float3 >},
                    {device::masked_recip_rz<double4, float4 >, device::masked_recip_rd<double4, float4 >, device::masked_recip_ru<double4, float4 >, device::masked_recip_rn<double4, float4 >}
                },
                {
                    {device::masked_recip_rz<double , double  >, device::masked_recip_rd<double , double  >, device::masked_recip_ru<double , double  >, device::masked_recip_rn<double , double  >},
                    {device::masked_recip_rz<double2, double2 >, device::masked_recip_rd<double2, double2 >, device::masked_recip_ru<double2, double2 >, device::masked_recip_rn<double2, double2 >},
                    {device::masked_recip_rz<double3, double3 >, device::masked_recip_rd<double3, double3 >, device::masked_recip_ru<double3, double3 >, device::masked_recip_rn<double3, double3 >},
                    {device::masked_recip_rz<double4, double4 >, device::masked_recip_rd<double4, double4 >, device::masked_recip_ru<double4, double4 >, device::masked_recip_rn<double4, double4 >}
                }
            }
        };

        masked_function_type fun = masked_funcs[src.depth()][CV_MAT_DEPTH(dtype)][src.channels()-1][flag_norm];

        fun(src, dst, mask, _stream);
    }

    if(reconstruction_needed)
    {
        GpuMat tmp = dst.reshape(cn);

        dst = tmp;
    }

    dst.copyTo(_dst, _stream);
}


namespace device
{

template<class SrcType, class DstType>
void recip_sqrt(const GpuMat& _src, GpuMat& _dst, Stream& _stream);



template<class SrcType, class DstType>
void masked_recip_sqrt(const GpuMat& _src, GpuMat& _dst, const GpuMat& _mask, Stream& _stream);


} // device

void reciprocal_sqrt(InputArray _src, OutputArray _dst, int dtype, InputArray _mask, Stream& _stream)
{
    CV_Assert(_src.isGpuMat() && _dst.isGpuMat() && (_mask.empty() || (_mask.isGpuMat() && (_mask.size() == _src.size()) && (_mask.type() == CV_8UC1) ) ));

    GpuMat src = _src.getGpuMat();
    GpuMat dst;



    int stype = src.type();
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int wdepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruction_needed(false);

    if(cn>4)
    {
        reconstruction_needed = true;

        GpuMat tmp;

        if(!src.isContinuous())
        {
            src.copyTo(tmp, _stream);
            src.release();
            src = tmp;
        }

        tmp = src.reshape(1);
        src = tmp;
    }

    dst.create(src.size(), !reconstruction_needed ? wtype : wdepth);

    if(_mask.empty())
    {

        typedef void(*function_type)(const GpuMat&, GpuMat&, Stream&);

        static const function_type funcs[7][7][4] = {
            {
                { device::recip_sqrt<uchar, uchar  >, device::recip_sqrt<uchar2, uchar2  >, device::recip_sqrt<uchar3, uchar3  >, device::recip_sqrt<uchar4, uchar4  > },
                { device::recip_sqrt<uchar, schar  >, device::recip_sqrt<uchar2, char2   >, device::recip_sqrt<uchar3, char3   >, device::recip_sqrt<uchar4, char4   > },
                { device::recip_sqrt<uchar, ushort >, device::recip_sqrt<uchar2, ushort2 >, device::recip_sqrt<uchar3, ushort3 >, device::recip_sqrt<uchar4, ushort4 > },
                { device::recip_sqrt<uchar, short  >, device::recip_sqrt<uchar2, short2  >, device::recip_sqrt<uchar3, short3  >, device::recip_sqrt<uchar4, short4  > },
                { device::recip_sqrt<uchar, int    >, device::recip_sqrt<uchar2, int2    >, device::recip_sqrt<uchar3, int3    >, device::recip_sqrt<uchar4, int4    > },
                { device::recip_sqrt<uchar, float  >, device::recip_sqrt<uchar2, float2  >, device::recip_sqrt<uchar3, float3  >, device::recip_sqrt<uchar4, float4  > },
                { device::recip_sqrt<uchar, double >, device::recip_sqrt<uchar2, double2 >, device::recip_sqrt<uchar3, double3 >, device::recip_sqrt<uchar4, double4 >}
            },
            {
                { device::recip_sqrt<schar, uchar  >, device::recip_sqrt<char2, uchar2  >, device::recip_sqrt<char3, uchar3  >, device::recip_sqrt<char4, uchar4  > },
                { device::recip_sqrt<schar, schar  >, device::recip_sqrt<char2, char2   >, device::recip_sqrt<char3, char3   >, device::recip_sqrt<char4, char4   > },
                { device::recip_sqrt<schar, ushort >, device::recip_sqrt<char2, ushort2 >, device::recip_sqrt<char3, ushort3 >, device::recip_sqrt<char4, ushort4 > },
                { device::recip_sqrt<schar, short  >, device::recip_sqrt<char2, short2  >, device::recip_sqrt<char3, short3  >, device::recip_sqrt<char4, short4  > },
                { device::recip_sqrt<schar, int    >, device::recip_sqrt<char2, int2    >, device::recip_sqrt<char3, int3    >, device::recip_sqrt<char4, int4    > },
                { device::recip_sqrt<schar, float  >, device::recip_sqrt<char2, float2  >, device::recip_sqrt<char3, float3  >, device::recip_sqrt<char4, float4  > },
                { device::recip_sqrt<schar, double >, device::recip_sqrt<char2, double2 >, device::recip_sqrt<char3, double3 >, device::recip_sqrt<char4, double4 >}
            },
            {

                { device::recip_sqrt<ushort, uchar  >, device::recip_sqrt<ushort2, uchar2  >, device::recip_sqrt<ushort3, uchar3  >, device::recip_sqrt<ushort4, uchar4  > },
                { device::recip_sqrt<ushort, schar  >, device::recip_sqrt<ushort2, char2   >, device::recip_sqrt<ushort3, char3   >, device::recip_sqrt<ushort4, char4   > },
                { device::recip_sqrt<ushort, ushort >, device::recip_sqrt<ushort2, ushort2 >, device::recip_sqrt<ushort3, ushort3 >, device::recip_sqrt<ushort4, ushort4 > },
                { device::recip_sqrt<ushort, short  >, device::recip_sqrt<ushort2, short2  >, device::recip_sqrt<ushort3, short3  >, device::recip_sqrt<ushort4, short4  > },
                { device::recip_sqrt<ushort, int    >, device::recip_sqrt<ushort2, int2    >, device::recip_sqrt<ushort3, int3    >, device::recip_sqrt<ushort4, int4    > },
                { device::recip_sqrt<ushort, float  >, device::recip_sqrt<ushort2, float2  >, device::recip_sqrt<ushort3, float3  >, device::recip_sqrt<ushort4, float4  > },
                { device::recip_sqrt<ushort, double >, device::recip_sqrt<ushort2, double2 >, device::recip_sqrt<ushort3, double3 >, device::recip_sqrt<ushort4, double4 >}
            },
            {
                { device::recip_sqrt<short, uchar  >, device::recip_sqrt<short2, uchar2  >, device::recip_sqrt<short3, uchar3  >, device::recip_sqrt<short4, uchar4  > },
                { device::recip_sqrt<short, schar  >, device::recip_sqrt<short2, char2   >, device::recip_sqrt<short3, char3   >, device::recip_sqrt<short4, char4   > },
                { device::recip_sqrt<short, ushort >, device::recip_sqrt<short2, ushort2 >, device::recip_sqrt<short3, ushort3 >, device::recip_sqrt<short4, ushort4 > },
                { device::recip_sqrt<short, short  >, device::recip_sqrt<short2, short2  >, device::recip_sqrt<short3, short3  >, device::recip_sqrt<short4, short4  > },
                { device::recip_sqrt<short, int    >, device::recip_sqrt<short2, int2    >, device::recip_sqrt<short3, int3    >, device::recip_sqrt<short4, int4    > },
                { device::recip_sqrt<short, float  >, device::recip_sqrt<short2, float2  >, device::recip_sqrt<short3, float3  >, device::recip_sqrt<short4, float4  > },
                { device::recip_sqrt<short, double >, device::recip_sqrt<short2, double2 >, device::recip_sqrt<short3, double3 >, device::recip_sqrt<short4, double4 >}
            },
            {
                { device::recip_sqrt<int, uchar  >, device::recip_sqrt<int2, uchar2  >, device::recip_sqrt<int3, uchar3  >, device::recip_sqrt<int4, uchar4  > },
                { device::recip_sqrt<int, schar  >, device::recip_sqrt<int2, char2   >, device::recip_sqrt<int3, char3   >, device::recip_sqrt<int4, char4   > },
                { device::recip_sqrt<int, ushort >, device::recip_sqrt<int2, ushort2 >, device::recip_sqrt<int3, ushort3 >, device::recip_sqrt<int4, ushort4 > },
                { device::recip_sqrt<int, short  >, device::recip_sqrt<int2, short2  >, device::recip_sqrt<int3, short3  >, device::recip_sqrt<int4, short4  > },
                { device::recip_sqrt<int, int    >, device::recip_sqrt<int2, int2    >, device::recip_sqrt<int3, int3    >, device::recip_sqrt<int4, int4    > },
                { device::recip_sqrt<int, float  >, device::recip_sqrt<int2, float2  >, device::recip_sqrt<int3, float3  >, device::recip_sqrt<int4, float4  > },
                { device::recip_sqrt<int, double >, device::recip_sqrt<int2, double2 >, device::recip_sqrt<int3, double3 >, device::recip_sqrt<int4, double4 >}
            },
            {
                { device::recip_sqrt<float, uchar  >, device::recip_sqrt<float2, uchar2  >, device::recip_sqrt<float3, uchar3  >, device::recip_sqrt<float4, uchar4  > },
                { device::recip_sqrt<float, schar  >, device::recip_sqrt<float2, char2   >, device::recip_sqrt<float3, char3   >, device::recip_sqrt<float4, char4   > },
                { device::recip_sqrt<float, ushort >, device::recip_sqrt<float2, ushort2 >, device::recip_sqrt<float3, ushort3 >, device::recip_sqrt<float4, ushort4 > },
                { device::recip_sqrt<float, short  >, device::recip_sqrt<float2, short2  >, device::recip_sqrt<float3, short3  >, device::recip_sqrt<float4, short4  > },
                { device::recip_sqrt<float, int    >, device::recip_sqrt<float2, int2    >, device::recip_sqrt<float3, int3    >, device::recip_sqrt<float4, int4    > },
                { device::recip_sqrt<float, float  >, device::recip_sqrt<float2, float2  >, device::recip_sqrt<float3, float3  >, device::recip_sqrt<float4, float4  > },
                { device::recip_sqrt<float, double >, device::recip_sqrt<float2, double2 >, device::recip_sqrt<float3, double3 >, device::recip_sqrt<float4, double4 >}
            },
            {
                { device::recip_sqrt<double, uchar  >, device::recip_sqrt<double2, uchar2  >, device::recip_sqrt<double3, uchar3  >, device::recip_sqrt<double4, uchar4  > },
                { device::recip_sqrt<double, schar  >, device::recip_sqrt<double2, char2   >, device::recip_sqrt<double3, char3   >, device::recip_sqrt<double4, char4   > },
                { device::recip_sqrt<double, ushort >, device::recip_sqrt<double2, ushort2 >, device::recip_sqrt<double3, ushort3 >, device::recip_sqrt<double4, ushort4 > },
                { device::recip_sqrt<double, short  >, device::recip_sqrt<double2, short2  >, device::recip_sqrt<double3, short3  >, device::recip_sqrt<double4, short4  > },
                { device::recip_sqrt<double, int    >, device::recip_sqrt<double2, int2    >, device::recip_sqrt<double3, int3    >, device::recip_sqrt<double4, int4    > },
                { device::recip_sqrt<double, float  >, device::recip_sqrt<double2, float2  >, device::recip_sqrt<double3, float3  >, device::recip_sqrt<double4, float4  > },
                { device::recip_sqrt<double, double >, device::recip_sqrt<double2, double2 >, device::recip_sqrt<double3, double3 >, device::recip_sqrt<double4, double4 >}
            }
        };

        function_type fun = funcs[src.depth()-CV_32F][CV_MAT_DEPTH(dtype)][src.channels()-1];

        fun(src, dst, _stream);
    }
    else
    {
        GpuMat mask = _mask.getGpuMat();

        typedef void(*masked_function_type)(const GpuMat&, GpuMat&, const GpuMat&, Stream&);

        static const masked_function_type masked_funcs[7][7][4] = {
            {
                { device::masked_recip_sqrt<uchar, uchar  >, device::masked_recip_sqrt<uchar2, uchar2  >, device::masked_recip_sqrt<uchar3, uchar3  >, device::masked_recip_sqrt<uchar4, uchar4  > },
                { device::masked_recip_sqrt<uchar, schar  >, device::masked_recip_sqrt<uchar2, char2   >, device::masked_recip_sqrt<uchar3, char3   >, device::masked_recip_sqrt<uchar4, char4   > },
                { device::masked_recip_sqrt<uchar, ushort >, device::masked_recip_sqrt<uchar2, ushort2 >, device::masked_recip_sqrt<uchar3, ushort3 >, device::masked_recip_sqrt<uchar4, ushort4 > },
                { device::masked_recip_sqrt<uchar, short  >, device::masked_recip_sqrt<uchar2, short2  >, device::masked_recip_sqrt<uchar3, short3  >, device::masked_recip_sqrt<uchar4, short4  > },
                { device::masked_recip_sqrt<uchar, int    >, device::masked_recip_sqrt<uchar2, int2    >, device::masked_recip_sqrt<uchar3, int3    >, device::masked_recip_sqrt<uchar4, int4    > },
                { device::masked_recip_sqrt<uchar, float  >, device::masked_recip_sqrt<uchar2, float2  >, device::masked_recip_sqrt<uchar3, float3  >, device::masked_recip_sqrt<uchar4, float4  > },
                { device::masked_recip_sqrt<uchar, double >, device::masked_recip_sqrt<uchar2, double2 >, device::masked_recip_sqrt<uchar3, double3 >, device::masked_recip_sqrt<uchar4, double4 >}
            },
            {
                { device::masked_recip_sqrt<schar, uchar  >, device::masked_recip_sqrt<char2, uchar2  >, device::masked_recip_sqrt<char3, uchar3  >, device::masked_recip_sqrt<char4, uchar4  > },
                { device::masked_recip_sqrt<schar, schar  >, device::masked_recip_sqrt<char2, char2   >, device::masked_recip_sqrt<char3, char3   >, device::masked_recip_sqrt<char4, char4   > },
                { device::masked_recip_sqrt<schar, ushort >, device::masked_recip_sqrt<char2, ushort2 >, device::masked_recip_sqrt<char3, ushort3 >, device::masked_recip_sqrt<char4, ushort4 > },
                { device::masked_recip_sqrt<schar, short  >, device::masked_recip_sqrt<char2, short2  >, device::masked_recip_sqrt<char3, short3  >, device::masked_recip_sqrt<char4, short4  > },
                { device::masked_recip_sqrt<schar, int    >, device::masked_recip_sqrt<char2, int2    >, device::masked_recip_sqrt<char3, int3    >, device::masked_recip_sqrt<char4, int4    > },
                { device::masked_recip_sqrt<schar, float  >, device::masked_recip_sqrt<char2, float2  >, device::masked_recip_sqrt<char3, float3  >, device::masked_recip_sqrt<char4, float4  > },
                { device::masked_recip_sqrt<schar, double >, device::masked_recip_sqrt<char2, double2 >, device::masked_recip_sqrt<char3, double3 >, device::masked_recip_sqrt<char4, double4 >}
            },
            {
                { device::masked_recip_sqrt<ushort, uchar  >, device::masked_recip_sqrt<ushort2, uchar2  >, device::masked_recip_sqrt<ushort3, uchar3  >, device::masked_recip_sqrt<ushort4, uchar4  > },
                { device::masked_recip_sqrt<ushort, schar  >, device::masked_recip_sqrt<ushort2, char2   >, device::masked_recip_sqrt<ushort3, char3   >, device::masked_recip_sqrt<ushort4, char4   > },
                { device::masked_recip_sqrt<ushort, ushort >, device::masked_recip_sqrt<ushort2, ushort2 >, device::masked_recip_sqrt<ushort3, ushort3 >, device::masked_recip_sqrt<ushort4, ushort4 > },
                { device::masked_recip_sqrt<ushort, short  >, device::masked_recip_sqrt<ushort2, short2  >, device::masked_recip_sqrt<ushort3, short3  >, device::masked_recip_sqrt<ushort4, short4  > },
                { device::masked_recip_sqrt<ushort, int    >, device::masked_recip_sqrt<ushort2, int2    >, device::masked_recip_sqrt<ushort3, int3    >, device::masked_recip_sqrt<ushort4, int4    > },
                { device::masked_recip_sqrt<ushort, float  >, device::masked_recip_sqrt<ushort2, float2  >, device::masked_recip_sqrt<ushort3, float3  >, device::masked_recip_sqrt<ushort4, float4  > },
                { device::masked_recip_sqrt<ushort, double >, device::masked_recip_sqrt<ushort2, double2 >, device::masked_recip_sqrt<ushort3, double3 >, device::masked_recip_sqrt<ushort4, double4 >}
            },
            {
                { device::masked_recip_sqrt<short, uchar  >, device::masked_recip_sqrt<short2, uchar2  >, device::masked_recip_sqrt<short3, uchar3  >, device::masked_recip_sqrt<short4, uchar4  > },
                { device::masked_recip_sqrt<short, schar  >, device::masked_recip_sqrt<short2, char2   >, device::masked_recip_sqrt<short3, char3   >, device::masked_recip_sqrt<short4, char4   > },
                { device::masked_recip_sqrt<short, ushort >, device::masked_recip_sqrt<short2, ushort2 >, device::masked_recip_sqrt<short3, ushort3 >, device::masked_recip_sqrt<short4, ushort4 > },
                { device::masked_recip_sqrt<short, short  >, device::masked_recip_sqrt<short2, short2  >, device::masked_recip_sqrt<short3, short3  >, device::masked_recip_sqrt<short4, short4  > },
                { device::masked_recip_sqrt<short, int    >, device::masked_recip_sqrt<short2, int2    >, device::masked_recip_sqrt<short3, int3    >, device::masked_recip_sqrt<short4, int4    > },
                { device::masked_recip_sqrt<short, float  >, device::masked_recip_sqrt<short2, float2  >, device::masked_recip_sqrt<short3, float3  >, device::masked_recip_sqrt<short4, float4  > },
                { device::masked_recip_sqrt<short, double >, device::masked_recip_sqrt<short2, double2 >, device::masked_recip_sqrt<short3, double3 >, device::masked_recip_sqrt<short4, double4 >}
            },
            {

                { device::masked_recip_sqrt<int, uchar  >, device::masked_recip_sqrt<int2, uchar2  >, device::masked_recip_sqrt<int3, uchar3  >, device::masked_recip_sqrt<int4, uchar4  > },
                { device::masked_recip_sqrt<int, schar  >, device::masked_recip_sqrt<int2, char2   >, device::masked_recip_sqrt<int3, char3   >, device::masked_recip_sqrt<int4, char4   > },
                { device::masked_recip_sqrt<int, ushort >, device::masked_recip_sqrt<int2, ushort2 >, device::masked_recip_sqrt<int3, ushort3 >, device::masked_recip_sqrt<int4, ushort4 > },
                { device::masked_recip_sqrt<int, short  >, device::masked_recip_sqrt<int2, short2  >, device::masked_recip_sqrt<int3, short3  >, device::masked_recip_sqrt<int4, short4  > },
                { device::masked_recip_sqrt<int, int    >, device::masked_recip_sqrt<int2, int2    >, device::masked_recip_sqrt<int3, int3    >, device::masked_recip_sqrt<int4, int4    > },
                { device::masked_recip_sqrt<int, float  >, device::masked_recip_sqrt<int2, float2  >, device::masked_recip_sqrt<int3, float3  >, device::masked_recip_sqrt<int4, float4  > },
                { device::masked_recip_sqrt<int, double >, device::masked_recip_sqrt<int2, double2 >, device::masked_recip_sqrt<int3, double3 >, device::masked_recip_sqrt<int4, double4 >}
            },
            {
                { device::masked_recip_sqrt<float, uchar  >, device::masked_recip_sqrt<float2, uchar2  >, device::masked_recip_sqrt<float3, uchar3  >, device::masked_recip_sqrt<float4, uchar4  > },
                { device::masked_recip_sqrt<float, schar  >, device::masked_recip_sqrt<float2, char2   >, device::masked_recip_sqrt<float3, char3   >, device::masked_recip_sqrt<float4, char4   > },
                { device::masked_recip_sqrt<float, ushort >, device::masked_recip_sqrt<float2, ushort2 >, device::masked_recip_sqrt<float3, ushort3 >, device::masked_recip_sqrt<float4, ushort4 > },
                { device::masked_recip_sqrt<float, short  >, device::masked_recip_sqrt<float2, short2  >, device::masked_recip_sqrt<float3, short3  >, device::masked_recip_sqrt<float4, short4  > },
                { device::masked_recip_sqrt<float, int    >, device::masked_recip_sqrt<float2, int2    >, device::masked_recip_sqrt<float3, int3    >, device::masked_recip_sqrt<float4, int4    > },
                { device::masked_recip_sqrt<float, float  >, device::masked_recip_sqrt<float2, float2  >, device::masked_recip_sqrt<float3, float3  >, device::masked_recip_sqrt<float4, float4  > },
                { device::masked_recip_sqrt<float, double >, device::masked_recip_sqrt<float2, double2 >, device::masked_recip_sqrt<float3, double3 >, device::masked_recip_sqrt<float4, double4 >}
            },
            {
                { device::masked_recip_sqrt<double, uchar  >, device::masked_recip_sqrt<double2, uchar2  >, device::masked_recip_sqrt<double3, uchar3  >, device::masked_recip_sqrt<double4, uchar4  > },
                { device::masked_recip_sqrt<double, schar  >, device::masked_recip_sqrt<double2, char2   >, device::masked_recip_sqrt<double3, char3   >, device::masked_recip_sqrt<double4, char4   > },
                { device::masked_recip_sqrt<double, ushort >, device::masked_recip_sqrt<double2, ushort2 >, device::masked_recip_sqrt<double3, ushort3 >, device::masked_recip_sqrt<double4, ushort4 > },
                { device::masked_recip_sqrt<double, short  >, device::masked_recip_sqrt<double2, short2  >, device::masked_recip_sqrt<double3, short3  >, device::masked_recip_sqrt<double4, short4  > },
                { device::masked_recip_sqrt<double, int    >, device::masked_recip_sqrt<double2, int2    >, device::masked_recip_sqrt<double3, int3    >, device::masked_recip_sqrt<double4, int4    > },
                { device::masked_recip_sqrt<double, float  >, device::masked_recip_sqrt<double2, float2  >, device::masked_recip_sqrt<double3, float3  >, device::masked_recip_sqrt<double4, float4  > },
                { device::masked_recip_sqrt<double, double >, device::masked_recip_sqrt<double2, double2 >, device::masked_recip_sqrt<double3, double3 >, device::masked_recip_sqrt<double4, double4 >}
            }
        };

        masked_function_type fun = masked_funcs[src.depth()][CV_MAT_DEPTH(dtype)][src.channels()-1];

        fun(src, dst, mask, _stream);
    }

    if(reconstruction_needed)
    {
        GpuMat tmp = dst.reshape(cn);

        dst = tmp;
    }

    dst.copyTo(_dst, _stream);
}


namespace device
{

// sxsps
template<class DstType, int op>
void fmx(const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

// axapa
template<class SrcType1, class SrcType2, class SrcType3, class DstType, int op>
void fmx(const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

// axsps
template<class SrcType, class DstType, int op>
void fmx(const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

// sxaps
template<class SrcType, class DstType, int op>
void fmx(const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

// sxspa
template<class SrcType, class DstType, int op>
void fmx(const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

// sxapa
template<class SrcType1, class SrcType2, class DstType, int op>
void fmx(const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

// axspa
template<class SrcType, class DstType, int op>
void fmx(const GpuMat&, const Scalar&,  const GpuMat&, GpuMat&, const GpuMat&, Stream&);

// axaps
template<class SrcType1, class SrcType2, class DstType, int op>
void fmx(const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);


} // device


namespace
{

template <int op>
void fmx_sosos_caller(const Scalar& _src1, const Scalar& _src2, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
    typedef void (*function_type)(const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

    static const function_type functions[7] = {device::fmx<uchar, op>, device::fmx<schar, op>, device::fmx<ushort, op>, device::fmx<short, op>, device::fmx<int, op>, device::fmx<float, op>, device::fmx<double, op>};

    function_type fun = functions[dtype == -1 ? CV_64F : CV_MAT_DEPTH(dtype)];

    fun(_src1, _src2, _src3, _dst, _mask, _stream);
}


template <int op>
void fmx_aoaoa(const GpuMat& _src1, const GpuMat& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
    CV_Assert((_src1.size() == _src2.size()) && (_src1.size() == _src3.size()) && (_mask.empty() || (_mask.size() == _src1.size())) );

    typedef void (*function_type)(const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

    static const function_type functions[7][7][7][7][4] = {
        {
            {
                {
                    { device::fmx<uchar, uchar, uchar, uchar, op>, device::fmx<uchar2, uchar2, uchar2, uchar2, op>, device::fmx<uchar3, uchar3, uchar3, uchar3, op>, device::fmx<uchar4, uchar4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, uchar, uchar, schar, op>, device::fmx<uchar2, uchar2, uchar2, char2, op>, device::fmx<uchar3, uchar3, uchar3, char3, op>, device::fmx<uchar4, uchar4, uchar4, char4, op>  },
                    { device::fmx<uchar, uchar, uchar, ushort, op>, device::fmx<uchar2, uchar2, uchar2, ushort2, op>, device::fmx<uchar3, uchar3, uchar3, ushort3, op>, device::fmx<uchar4, uchar4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, uchar, uchar, short, op>, device::fmx<uchar2, uchar2, uchar2, short2, op>, device::fmx<uchar3, uchar3, uchar3, short3, op>, device::fmx<uchar4, uchar4, uchar4, short4, op>  },
                    { device::fmx<uchar, uchar, uchar, int, op>, device::fmx<uchar2, uchar2, uchar2, int2, op>, device::fmx<uchar3, uchar3, uchar3, int3, op>, device::fmx<uchar4, uchar4, uchar4, int4, op>  },
                    { device::fmx<uchar, uchar, uchar, float, op>, device::fmx<uchar2, uchar2, uchar2, float2, op>, device::fmx<uchar3, uchar3, uchar3, float3, op>, device::fmx<uchar4, uchar4, uchar4, float4, op>  },
                    { device::fmx<uchar, uchar, uchar, double, op>, device::fmx<uchar2, uchar2, uchar2, double2, op>, device::fmx<uchar3, uchar3, uchar3, double3, op>, device::fmx<uchar4, uchar4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<uchar, uchar, schar, uchar, op>, device::fmx<uchar2, uchar2, char2, uchar2, op>, device::fmx<uchar3, uchar3, char3, uchar3, op>, device::fmx<uchar4, uchar4, char4, uchar4, op>  },
                    { device::fmx<uchar, uchar, schar, schar, op>, device::fmx<uchar2, uchar2, char2, char2, op>, device::fmx<uchar3, uchar3, char3, char3, op>, device::fmx<uchar4, uchar4, char4, char4, op>  },
                    { device::fmx<uchar, uchar, schar, ushort, op>, device::fmx<uchar2, uchar2, char2, ushort2, op>, device::fmx<uchar3, uchar3, char3, ushort3, op>, device::fmx<uchar4, uchar4, char4, ushort4, op>  },
                    { device::fmx<uchar, uchar, schar, short, op>, device::fmx<uchar2, uchar2, char2, short2, op>, device::fmx<uchar3, uchar3, char3, short3, op>, device::fmx<uchar4, uchar4, char4, short4, op>  },
                    { device::fmx<uchar, uchar, schar, int, op>, device::fmx<uchar2, uchar2, char2, int2, op>, device::fmx<uchar3, uchar3, char3, int3, op>, device::fmx<uchar4, uchar4, char4, int4, op>  },
                    { device::fmx<uchar, uchar, schar, float, op>, device::fmx<uchar2, uchar2, char2, float2, op>, device::fmx<uchar3, uchar3, char3, float3, op>, device::fmx<uchar4, uchar4, char4, float4, op>  },
                    { device::fmx<uchar, uchar, schar, double, op>, device::fmx<uchar2, uchar2, char2, double2, op>, device::fmx<uchar3, uchar3, char3, double3, op>, device::fmx<uchar4, uchar4, char4, double4, op>  },
                },
                {
                    { device::fmx<uchar, uchar, ushort, uchar, op>, device::fmx<uchar2, uchar2, ushort2, uchar2, op>, device::fmx<uchar3, uchar3, ushort3, uchar3, op>, device::fmx<uchar4, uchar4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, uchar, ushort, schar, op>, device::fmx<uchar2, uchar2, ushort2, char2, op>, device::fmx<uchar3, uchar3, ushort3, char3, op>, device::fmx<uchar4, uchar4, ushort4, char4, op>  },
                    { device::fmx<uchar, uchar, ushort, ushort, op>, device::fmx<uchar2, uchar2, ushort2, ushort2, op>, device::fmx<uchar3, uchar3, ushort3, ushort3, op>, device::fmx<uchar4, uchar4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, uchar, ushort, short, op>, device::fmx<uchar2, uchar2, ushort2, short2, op>, device::fmx<uchar3, uchar3, ushort3, short3, op>, device::fmx<uchar4, uchar4, ushort4, short4, op>  },
                    { device::fmx<uchar, uchar, ushort, int, op>, device::fmx<uchar2, uchar2, ushort2, int2, op>, device::fmx<uchar3, uchar3, ushort3, int3, op>, device::fmx<uchar4, uchar4, ushort4, int4, op>  },
                    { device::fmx<uchar, uchar, ushort, float, op>, device::fmx<uchar2, uchar2, ushort2, float2, op>, device::fmx<uchar3, uchar3, ushort3, float3, op>, device::fmx<uchar4, uchar4, ushort4, float4, op>  },
                    { device::fmx<uchar, uchar, ushort, double, op>, device::fmx<uchar2, uchar2, ushort2, double2, op>, device::fmx<uchar3, uchar3, ushort3, double3, op>, device::fmx<uchar4, uchar4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<uchar, uchar, short, uchar, op>, device::fmx<uchar2, uchar2, short2, uchar2, op>, device::fmx<uchar3, uchar3, short3, uchar3, op>, device::fmx<uchar4, uchar4, short4, uchar4, op>  },
                    { device::fmx<uchar, uchar, short, schar, op>, device::fmx<uchar2, uchar2, short2, char2, op>, device::fmx<uchar3, uchar3, short3, char3, op>, device::fmx<uchar4, uchar4, short4, char4, op>  },
                    { device::fmx<uchar, uchar, short, ushort, op>, device::fmx<uchar2, uchar2, short2, ushort2, op>, device::fmx<uchar3, uchar3, short3, ushort3, op>, device::fmx<uchar4, uchar4, short4, ushort4, op>  },
                    { device::fmx<uchar, uchar, short, short, op>, device::fmx<uchar2, uchar2, short2, short2, op>, device::fmx<uchar3, uchar3, short3, short3, op>, device::fmx<uchar4, uchar4, short4, short4, op>  },
                    { device::fmx<uchar, uchar, short, int, op>, device::fmx<uchar2, uchar2, short2, int2, op>, device::fmx<uchar3, uchar3, short3, int3, op>, device::fmx<uchar4, uchar4, short4, int4, op>  },
                    { device::fmx<uchar, uchar, short, float, op>, device::fmx<uchar2, uchar2, short2, float2, op>, device::fmx<uchar3, uchar3, short3, float3, op>, device::fmx<uchar4, uchar4, short4, float4, op>  },
                    { device::fmx<uchar, uchar, short, double, op>, device::fmx<uchar2, uchar2, short2, double2, op>, device::fmx<uchar3, uchar3, short3, double3, op>, device::fmx<uchar4, uchar4, short4, double4, op>  },
                },
                {
                    { device::fmx<uchar, uchar, int, uchar, op>, device::fmx<uchar2, uchar2, int2, uchar2, op>, device::fmx<uchar3, uchar3, int3, uchar3, op>, device::fmx<uchar4, uchar4, int4, uchar4, op>  },
                    { device::fmx<uchar, uchar, int, schar, op>, device::fmx<uchar2, uchar2, int2, char2, op>, device::fmx<uchar3, uchar3, int3, char3, op>, device::fmx<uchar4, uchar4, int4, char4, op>  },
                    { device::fmx<uchar, uchar, int, ushort, op>, device::fmx<uchar2, uchar2, int2, ushort2, op>, device::fmx<uchar3, uchar3, int3, ushort3, op>, device::fmx<uchar4, uchar4, int4, ushort4, op>  },
                    { device::fmx<uchar, uchar, int, short, op>, device::fmx<uchar2, uchar2, int2, short2, op>, device::fmx<uchar3, uchar3, int3, short3, op>, device::fmx<uchar4, uchar4, int4, short4, op>  },
                    { device::fmx<uchar, uchar, int, int, op>, device::fmx<uchar2, uchar2, int2, int2, op>, device::fmx<uchar3, uchar3, int3, int3, op>, device::fmx<uchar4, uchar4, int4, int4, op>  },
                    { device::fmx<uchar, uchar, int, float, op>, device::fmx<uchar2, uchar2, int2, float2, op>, device::fmx<uchar3, uchar3, int3, float3, op>, device::fmx<uchar4, uchar4, int4, float4, op>  },
                    { device::fmx<uchar, uchar, int, double, op>, device::fmx<uchar2, uchar2, int2, double2, op>, device::fmx<uchar3, uchar3, int3, double3, op>, device::fmx<uchar4, uchar4, int4, double4, op>  },
                },
                {
                    { device::fmx<uchar, uchar, float, uchar, op>, device::fmx<uchar2, uchar2, float2, uchar2, op>, device::fmx<uchar3, uchar3, float3, uchar3, op>, device::fmx<uchar4, uchar4, float4, uchar4, op>  },
                    { device::fmx<uchar, uchar, float, schar, op>, device::fmx<uchar2, uchar2, float2, char2, op>, device::fmx<uchar3, uchar3, float3, char3, op>, device::fmx<uchar4, uchar4, float4, char4, op>  },
                    { device::fmx<uchar, uchar, float, ushort, op>, device::fmx<uchar2, uchar2, float2, ushort2, op>, device::fmx<uchar3, uchar3, float3, ushort3, op>, device::fmx<uchar4, uchar4, float4, ushort4, op>  },
                    { device::fmx<uchar, uchar, float, short, op>, device::fmx<uchar2, uchar2, float2, short2, op>, device::fmx<uchar3, uchar3, float3, short3, op>, device::fmx<uchar4, uchar4, float4, short4, op>  },
                    { device::fmx<uchar, uchar, float, int, op>, device::fmx<uchar2, uchar2, float2, int2, op>, device::fmx<uchar3, uchar3, float3, int3, op>, device::fmx<uchar4, uchar4, float4, int4, op>  },
                    { device::fmx<uchar, uchar, float, float, op>, device::fmx<uchar2, uchar2, float2, float2, op>, device::fmx<uchar3, uchar3, float3, float3, op>, device::fmx<uchar4, uchar4, float4, float4, op>  },
                    { device::fmx<uchar, uchar, float, double, op>, device::fmx<uchar2, uchar2, float2, double2, op>, device::fmx<uchar3, uchar3, float3, double3, op>, device::fmx<uchar4, uchar4, float4, double4, op>  },
                },
                {
                    { device::fmx<uchar, uchar, double, uchar, op>, device::fmx<uchar2, uchar2, double2, uchar2, op>, device::fmx<uchar3, uchar3, double3, uchar3, op>, device::fmx<uchar4, uchar4, double4, uchar4, op>  },
                    { device::fmx<uchar, uchar, double, schar, op>, device::fmx<uchar2, uchar2, double2, char2, op>, device::fmx<uchar3, uchar3, double3, char3, op>, device::fmx<uchar4, uchar4, double4, char4, op>  },
                    { device::fmx<uchar, uchar, double, ushort, op>, device::fmx<uchar2, uchar2, double2, ushort2, op>, device::fmx<uchar3, uchar3, double3, ushort3, op>, device::fmx<uchar4, uchar4, double4, ushort4, op>  },
                    { device::fmx<uchar, uchar, double, short, op>, device::fmx<uchar2, uchar2, double2, short2, op>, device::fmx<uchar3, uchar3, double3, short3, op>, device::fmx<uchar4, uchar4, double4, short4, op>  },
                    { device::fmx<uchar, uchar, double, int, op>, device::fmx<uchar2, uchar2, double2, int2, op>, device::fmx<uchar3, uchar3, double3, int3, op>, device::fmx<uchar4, uchar4, double4, int4, op>  },
                    { device::fmx<uchar, uchar, double, float, op>, device::fmx<uchar2, uchar2, double2, float2, op>, device::fmx<uchar3, uchar3, double3, float3, op>, device::fmx<uchar4, uchar4, double4, float4, op>  },
                    { device::fmx<uchar, uchar, double, double, op>, device::fmx<uchar2, uchar2, double2, double2, op>, device::fmx<uchar3, uchar3, double3, double3, op>, device::fmx<uchar4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<uchar, schar, uchar, uchar, op>, device::fmx<uchar2, char2, uchar2, uchar2, op>, device::fmx<uchar3, char3, uchar3, uchar3, op>, device::fmx<uchar4, char4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, schar, uchar, schar, op>, device::fmx<uchar2, char2, uchar2, char2, op>, device::fmx<uchar3, char3, uchar3, char3, op>, device::fmx<uchar4, char4, uchar4, char4, op>  },
                    { device::fmx<uchar, schar, uchar, ushort, op>, device::fmx<uchar2, char2, uchar2, ushort2, op>, device::fmx<uchar3, char3, uchar3, ushort3, op>, device::fmx<uchar4, char4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, schar, uchar, short, op>, device::fmx<uchar2, char2, uchar2, short2, op>, device::fmx<uchar3, char3, uchar3, short3, op>, device::fmx<uchar4, char4, uchar4, short4, op>  },
                    { device::fmx<uchar, schar, uchar, int, op>, device::fmx<uchar2, char2, uchar2, int2, op>, device::fmx<uchar3, char3, uchar3, int3, op>, device::fmx<uchar4, char4, uchar4, int4, op>  },
                    { device::fmx<uchar, schar, uchar, float, op>, device::fmx<uchar2, char2, uchar2, float2, op>, device::fmx<uchar3, char3, uchar3, float3, op>, device::fmx<uchar4, char4, uchar4, float4, op>  },
                    { device::fmx<uchar, schar, uchar, double, op>, device::fmx<uchar2, char2, uchar2, double2, op>, device::fmx<uchar3, char3, uchar3, double3, op>, device::fmx<uchar4, char4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<uchar, schar, schar, uchar, op>, device::fmx<uchar2, char2, char2, uchar2, op>, device::fmx<uchar3, char3, char3, uchar3, op>, device::fmx<uchar4, char4, char4, uchar4, op>  },
                    { device::fmx<uchar, schar, schar, schar, op>, device::fmx<uchar2, char2, char2, char2, op>, device::fmx<uchar3, char3, char3, char3, op>, device::fmx<uchar4, char4, char4, char4, op>  },
                    { device::fmx<uchar, schar, schar, ushort, op>, device::fmx<uchar2, char2, char2, ushort2, op>, device::fmx<uchar3, char3, char3, ushort3, op>, device::fmx<uchar4, char4, char4, ushort4, op>  },
                    { device::fmx<uchar, schar, schar, short, op>, device::fmx<uchar2, char2, char2, short2, op>, device::fmx<uchar3, char3, char3, short3, op>, device::fmx<uchar4, char4, char4, short4, op>  },
                    { device::fmx<uchar, schar, schar, int, op>, device::fmx<uchar2, char2, char2, int2, op>, device::fmx<uchar3, char3, char3, int3, op>, device::fmx<uchar4, char4, char4, int4, op>  },
                    { device::fmx<uchar, schar, schar, float, op>, device::fmx<uchar2, char2, char2, float2, op>, device::fmx<uchar3, char3, char3, float3, op>, device::fmx<uchar4, char4, char4, float4, op>  },
                    { device::fmx<uchar, schar, schar, double, op>, device::fmx<uchar2, char2, char2, double2, op>, device::fmx<uchar3, char3, char3, double3, op>, device::fmx<uchar4, char4, char4, double4, op>  },
                },
                {
                    { device::fmx<uchar, schar, ushort, uchar, op>, device::fmx<uchar2, char2, ushort2, uchar2, op>, device::fmx<uchar3, char3, ushort3, uchar3, op>, device::fmx<uchar4, char4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, schar, ushort, schar, op>, device::fmx<uchar2, char2, ushort2, char2, op>, device::fmx<uchar3, char3, ushort3, char3, op>, device::fmx<uchar4, char4, ushort4, char4, op>  },
                    { device::fmx<uchar, schar, ushort, ushort, op>, device::fmx<uchar2, char2, ushort2, ushort2, op>, device::fmx<uchar3, char3, ushort3, ushort3, op>, device::fmx<uchar4, char4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, schar, ushort, short, op>, device::fmx<uchar2, char2, ushort2, short2, op>, device::fmx<uchar3, char3, ushort3, short3, op>, device::fmx<uchar4, char4, ushort4, short4, op>  },
                    { device::fmx<uchar, schar, ushort, int, op>, device::fmx<uchar2, char2, ushort2, int2, op>, device::fmx<uchar3, char3, ushort3, int3, op>, device::fmx<uchar4, char4, ushort4, int4, op>  },
                    { device::fmx<uchar, schar, ushort, float, op>, device::fmx<uchar2, char2, ushort2, float2, op>, device::fmx<uchar3, char3, ushort3, float3, op>, device::fmx<uchar4, char4, ushort4, float4, op>  },
                    { device::fmx<uchar, schar, ushort, double, op>, device::fmx<uchar2, char2, ushort2, double2, op>, device::fmx<uchar3, char3, ushort3, double3, op>, device::fmx<uchar4, char4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<uchar, schar, short, uchar, op>, device::fmx<uchar2, char2, short2, uchar2, op>, device::fmx<uchar3, char3, short3, uchar3, op>, device::fmx<uchar4, char4, short4, uchar4, op>  },
                    { device::fmx<uchar, schar, short, schar, op>, device::fmx<uchar2, char2, short2, char2, op>, device::fmx<uchar3, char3, short3, char3, op>, device::fmx<uchar4, char4, short4, char4, op>  },
                    { device::fmx<uchar, schar, short, ushort, op>, device::fmx<uchar2, char2, short2, ushort2, op>, device::fmx<uchar3, char3, short3, ushort3, op>, device::fmx<uchar4, char4, short4, ushort4, op>  },
                    { device::fmx<uchar, schar, short, short, op>, device::fmx<uchar2, char2, short2, short2, op>, device::fmx<uchar3, char3, short3, short3, op>, device::fmx<uchar4, char4, short4, short4, op>  },
                    { device::fmx<uchar, schar, short, int, op>, device::fmx<uchar2, char2, short2, int2, op>, device::fmx<uchar3, char3, short3, int3, op>, device::fmx<uchar4, char4, short4, int4, op>  },
                    { device::fmx<uchar, schar, short, float, op>, device::fmx<uchar2, char2, short2, float2, op>, device::fmx<uchar3, char3, short3, float3, op>, device::fmx<uchar4, char4, short4, float4, op>  },
                    { device::fmx<uchar, schar, short, double, op>, device::fmx<uchar2, char2, short2, double2, op>, device::fmx<uchar3, char3, short3, double3, op>, device::fmx<uchar4, char4, short4, double4, op>  },
                },
                {
                    { device::fmx<uchar, schar, int, uchar, op>, device::fmx<uchar2, char2, int2, uchar2, op>, device::fmx<uchar3, char3, int3, uchar3, op>, device::fmx<uchar4, char4, int4, uchar4, op>  },
                    { device::fmx<uchar, schar, int, schar, op>, device::fmx<uchar2, char2, int2, char2, op>, device::fmx<uchar3, char3, int3, char3, op>, device::fmx<uchar4, char4, int4, char4, op>  },
                    { device::fmx<uchar, schar, int, ushort, op>, device::fmx<uchar2, char2, int2, ushort2, op>, device::fmx<uchar3, char3, int3, ushort3, op>, device::fmx<uchar4, char4, int4, ushort4, op>  },
                    { device::fmx<uchar, schar, int, short, op>, device::fmx<uchar2, char2, int2, short2, op>, device::fmx<uchar3, char3, int3, short3, op>, device::fmx<uchar4, char4, int4, short4, op>  },
                    { device::fmx<uchar, schar, int, int, op>, device::fmx<uchar2, char2, int2, int2, op>, device::fmx<uchar3, char3, int3, int3, op>, device::fmx<uchar4, char4, int4, int4, op>  },
                    { device::fmx<uchar, schar, int, float, op>, device::fmx<uchar2, char2, int2, float2, op>, device::fmx<uchar3, char3, int3, float3, op>, device::fmx<uchar4, char4, int4, float4, op>  },
                    { device::fmx<uchar, schar, int, double, op>, device::fmx<uchar2, char2, int2, double2, op>, device::fmx<uchar3, char3, int3, double3, op>, device::fmx<uchar4, char4, int4, double4, op>  },
                },
                {
                    { device::fmx<uchar, schar, float, uchar, op>, device::fmx<uchar2, char2, float2, uchar2, op>, device::fmx<uchar3, char3, float3, uchar3, op>, device::fmx<uchar4, char4, float4, uchar4, op>  },
                    { device::fmx<uchar, schar, float, schar, op>, device::fmx<uchar2, char2, float2, char2, op>, device::fmx<uchar3, char3, float3, char3, op>, device::fmx<uchar4, char4, float4, char4, op>  },
                    { device::fmx<uchar, schar, float, ushort, op>, device::fmx<uchar2, char2, float2, ushort2, op>, device::fmx<uchar3, char3, float3, ushort3, op>, device::fmx<uchar4, char4, float4, ushort4, op>  },
                    { device::fmx<uchar, schar, float, short, op>, device::fmx<uchar2, char2, float2, short2, op>, device::fmx<uchar3, char3, float3, short3, op>, device::fmx<uchar4, char4, float4, short4, op>  },
                    { device::fmx<uchar, schar, float, int, op>, device::fmx<uchar2, char2, float2, int2, op>, device::fmx<uchar3, char3, float3, int3, op>, device::fmx<uchar4, char4, float4, int4, op>  },
                    { device::fmx<uchar, schar, float, float, op>, device::fmx<uchar2, char2, float2, float2, op>, device::fmx<uchar3, char3, float3, float3, op>, device::fmx<uchar4, char4, float4, float4, op>  },
                    { device::fmx<uchar, schar, float, double, op>, device::fmx<uchar2, char2, float2, double2, op>, device::fmx<uchar3, char3, float3, double3, op>, device::fmx<uchar4, char4, float4, double4, op>  },
                },
                {
                    { device::fmx<uchar, schar, double, uchar, op>, device::fmx<uchar2, char2, double2, uchar2, op>, device::fmx<uchar3, char3, double3, uchar3, op>, device::fmx<uchar4, char4, double4, uchar4, op>  },
                    { device::fmx<uchar, schar, double, schar, op>, device::fmx<uchar2, char2, double2, char2, op>, device::fmx<uchar3, char3, double3, char3, op>, device::fmx<uchar4, char4, double4, char4, op>  },
                    { device::fmx<uchar, schar, double, ushort, op>, device::fmx<uchar2, char2, double2, ushort2, op>, device::fmx<uchar3, char3, double3, ushort3, op>, device::fmx<uchar4, char4, double4, ushort4, op>  },
                    { device::fmx<uchar, schar, double, short, op>, device::fmx<uchar2, char2, double2, short2, op>, device::fmx<uchar3, char3, double3, short3, op>, device::fmx<uchar4, char4, double4, short4, op>  },
                    { device::fmx<uchar, schar, double, int, op>, device::fmx<uchar2, char2, double2, int2, op>, device::fmx<uchar3, char3, double3, int3, op>, device::fmx<uchar4, char4, double4, int4, op>  },
                    { device::fmx<uchar, schar, double, float, op>, device::fmx<uchar2, char2, double2, float2, op>, device::fmx<uchar3, char3, double3, float3, op>, device::fmx<uchar4, char4, double4, float4, op>  },
                    { device::fmx<uchar, schar, double, double, op>, device::fmx<uchar2, char2, double2, double2, op>, device::fmx<uchar3, char3, double3, double3, op>, device::fmx<uchar4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<uchar, ushort, uchar, uchar, op>, device::fmx<uchar2, ushort2, uchar2, uchar2, op>, device::fmx<uchar3, ushort3, uchar3, uchar3, op>, device::fmx<uchar4, ushort4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, ushort, uchar, schar, op>, device::fmx<uchar2, ushort2, uchar2, char2, op>, device::fmx<uchar3, ushort3, uchar3, char3, op>, device::fmx<uchar4, ushort4, uchar4, char4, op>  },
                    { device::fmx<uchar, ushort, uchar, ushort, op>, device::fmx<uchar2, ushort2, uchar2, ushort2, op>, device::fmx<uchar3, ushort3, uchar3, ushort3, op>, device::fmx<uchar4, ushort4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, ushort, uchar, short, op>, device::fmx<uchar2, ushort2, uchar2, short2, op>, device::fmx<uchar3, ushort3, uchar3, short3, op>, device::fmx<uchar4, ushort4, uchar4, short4, op>  },
                    { device::fmx<uchar, ushort, uchar, int, op>, device::fmx<uchar2, ushort2, uchar2, int2, op>, device::fmx<uchar3, ushort3, uchar3, int3, op>, device::fmx<uchar4, ushort4, uchar4, int4, op>  },
                    { device::fmx<uchar, ushort, uchar, float, op>, device::fmx<uchar2, ushort2, uchar2, float2, op>, device::fmx<uchar3, ushort3, uchar3, float3, op>, device::fmx<uchar4, ushort4, uchar4, float4, op>  },
                    { device::fmx<uchar, ushort, uchar, double, op>, device::fmx<uchar2, ushort2, uchar2, double2, op>, device::fmx<uchar3, ushort3, uchar3, double3, op>, device::fmx<uchar4, ushort4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<uchar, ushort, schar, uchar, op>, device::fmx<uchar2, ushort2, char2, uchar2, op>, device::fmx<uchar3, ushort3, char3, uchar3, op>, device::fmx<uchar4, ushort4, char4, uchar4, op>  },
                    { device::fmx<uchar, ushort, schar, schar, op>, device::fmx<uchar2, ushort2, char2, char2, op>, device::fmx<uchar3, ushort3, char3, char3, op>, device::fmx<uchar4, ushort4, char4, char4, op>  },
                    { device::fmx<uchar, ushort, schar, ushort, op>, device::fmx<uchar2, ushort2, char2, ushort2, op>, device::fmx<uchar3, ushort3, char3, ushort3, op>, device::fmx<uchar4, ushort4, char4, ushort4, op>  },
                    { device::fmx<uchar, ushort, schar, short, op>, device::fmx<uchar2, ushort2, char2, short2, op>, device::fmx<uchar3, ushort3, char3, short3, op>, device::fmx<uchar4, ushort4, char4, short4, op>  },
                    { device::fmx<uchar, ushort, schar, int, op>, device::fmx<uchar2, ushort2, char2, int2, op>, device::fmx<uchar3, ushort3, char3, int3, op>, device::fmx<uchar4, ushort4, char4, int4, op>  },
                    { device::fmx<uchar, ushort, schar, float, op>, device::fmx<uchar2, ushort2, char2, float2, op>, device::fmx<uchar3, ushort3, char3, float3, op>, device::fmx<uchar4, ushort4, char4, float4, op>  },
                    { device::fmx<uchar, ushort, schar, double, op>, device::fmx<uchar2, ushort2, char2, double2, op>, device::fmx<uchar3, ushort3, char3, double3, op>, device::fmx<uchar4, ushort4, char4, double4, op>  },
                },
                {
                    { device::fmx<uchar, ushort, ushort, uchar, op>, device::fmx<uchar2, ushort2, ushort2, uchar2, op>, device::fmx<uchar3, ushort3, ushort3, uchar3, op>, device::fmx<uchar4, ushort4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, ushort, ushort, schar, op>, device::fmx<uchar2, ushort2, ushort2, char2, op>, device::fmx<uchar3, ushort3, ushort3, char3, op>, device::fmx<uchar4, ushort4, ushort4, char4, op>  },
                    { device::fmx<uchar, ushort, ushort, ushort, op>, device::fmx<uchar2, ushort2, ushort2, ushort2, op>, device::fmx<uchar3, ushort3, ushort3, ushort3, op>, device::fmx<uchar4, ushort4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, ushort, ushort, short, op>, device::fmx<uchar2, ushort2, ushort2, short2, op>, device::fmx<uchar3, ushort3, ushort3, short3, op>, device::fmx<uchar4, ushort4, ushort4, short4, op>  },
                    { device::fmx<uchar, ushort, ushort, int, op>, device::fmx<uchar2, ushort2, ushort2, int2, op>, device::fmx<uchar3, ushort3, ushort3, int3, op>, device::fmx<uchar4, ushort4, ushort4, int4, op>  },
                    { device::fmx<uchar, ushort, ushort, float, op>, device::fmx<uchar2, ushort2, ushort2, float2, op>, device::fmx<uchar3, ushort3, ushort3, float3, op>, device::fmx<uchar4, ushort4, ushort4, float4, op>  },
                    { device::fmx<uchar, ushort, ushort, double, op>, device::fmx<uchar2, ushort2, ushort2, double2, op>, device::fmx<uchar3, ushort3, ushort3, double3, op>, device::fmx<uchar4, ushort4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<uchar, ushort, short, uchar, op>, device::fmx<uchar2, ushort2, short2, uchar2, op>, device::fmx<uchar3, ushort3, short3, uchar3, op>, device::fmx<uchar4, ushort4, short4, uchar4, op>  },
                    { device::fmx<uchar, ushort, short, schar, op>, device::fmx<uchar2, ushort2, short2, char2, op>, device::fmx<uchar3, ushort3, short3, char3, op>, device::fmx<uchar4, ushort4, short4, char4, op>  },
                    { device::fmx<uchar, ushort, short, ushort, op>, device::fmx<uchar2, ushort2, short2, ushort2, op>, device::fmx<uchar3, ushort3, short3, ushort3, op>, device::fmx<uchar4, ushort4, short4, ushort4, op>  },
                    { device::fmx<uchar, ushort, short, short, op>, device::fmx<uchar2, ushort2, short2, short2, op>, device::fmx<uchar3, ushort3, short3, short3, op>, device::fmx<uchar4, ushort4, short4, short4, op>  },
                    { device::fmx<uchar, ushort, short, int, op>, device::fmx<uchar2, ushort2, short2, int2, op>, device::fmx<uchar3, ushort3, short3, int3, op>, device::fmx<uchar4, ushort4, short4, int4, op>  },
                    { device::fmx<uchar, ushort, short, float, op>, device::fmx<uchar2, ushort2, short2, float2, op>, device::fmx<uchar3, ushort3, short3, float3, op>, device::fmx<uchar4, ushort4, short4, float4, op>  },
                    { device::fmx<uchar, ushort, short, double, op>, device::fmx<uchar2, ushort2, short2, double2, op>, device::fmx<uchar3, ushort3, short3, double3, op>, device::fmx<uchar4, ushort4, short4, double4, op>  },
                },
                {
                    { device::fmx<uchar, ushort, int, uchar, op>, device::fmx<uchar2, ushort2, int2, uchar2, op>, device::fmx<uchar3, ushort3, int3, uchar3, op>, device::fmx<uchar4, ushort4, int4, uchar4, op>  },
                    { device::fmx<uchar, ushort, int, schar, op>, device::fmx<uchar2, ushort2, int2, char2, op>, device::fmx<uchar3, ushort3, int3, char3, op>, device::fmx<uchar4, ushort4, int4, char4, op>  },
                    { device::fmx<uchar, ushort, int, ushort, op>, device::fmx<uchar2, ushort2, int2, ushort2, op>, device::fmx<uchar3, ushort3, int3, ushort3, op>, device::fmx<uchar4, ushort4, int4, ushort4, op>  },
                    { device::fmx<uchar, ushort, int, short, op>, device::fmx<uchar2, ushort2, int2, short2, op>, device::fmx<uchar3, ushort3, int3, short3, op>, device::fmx<uchar4, ushort4, int4, short4, op>  },
                    { device::fmx<uchar, ushort, int, int, op>, device::fmx<uchar2, ushort2, int2, int2, op>, device::fmx<uchar3, ushort3, int3, int3, op>, device::fmx<uchar4, ushort4, int4, int4, op>  },
                    { device::fmx<uchar, ushort, int, float, op>, device::fmx<uchar2, ushort2, int2, float2, op>, device::fmx<uchar3, ushort3, int3, float3, op>, device::fmx<uchar4, ushort4, int4, float4, op>  },
                    { device::fmx<uchar, ushort, int, double, op>, device::fmx<uchar2, ushort2, int2, double2, op>, device::fmx<uchar3, ushort3, int3, double3, op>, device::fmx<uchar4, ushort4, int4, double4, op>  },
                },
                {
                    { device::fmx<uchar, ushort, float, uchar, op>, device::fmx<uchar2, ushort2, float2, uchar2, op>, device::fmx<uchar3, ushort3, float3, uchar3, op>, device::fmx<uchar4, ushort4, float4, uchar4, op>  },
                    { device::fmx<uchar, ushort, float, schar, op>, device::fmx<uchar2, ushort2, float2, char2, op>, device::fmx<uchar3, ushort3, float3, char3, op>, device::fmx<uchar4, ushort4, float4, char4, op>  },
                    { device::fmx<uchar, ushort, float, ushort, op>, device::fmx<uchar2, ushort2, float2, ushort2, op>, device::fmx<uchar3, ushort3, float3, ushort3, op>, device::fmx<uchar4, ushort4, float4, ushort4, op>  },
                    { device::fmx<uchar, ushort, float, short, op>, device::fmx<uchar2, ushort2, float2, short2, op>, device::fmx<uchar3, ushort3, float3, short3, op>, device::fmx<uchar4, ushort4, float4, short4, op>  },
                    { device::fmx<uchar, ushort, float, int, op>, device::fmx<uchar2, ushort2, float2, int2, op>, device::fmx<uchar3, ushort3, float3, int3, op>, device::fmx<uchar4, ushort4, float4, int4, op>  },
                    { device::fmx<uchar, ushort, float, float, op>, device::fmx<uchar2, ushort2, float2, float2, op>, device::fmx<uchar3, ushort3, float3, float3, op>, device::fmx<uchar4, ushort4, float4, float4, op>  },
                    { device::fmx<uchar, ushort, float, double, op>, device::fmx<uchar2, ushort2, float2, double2, op>, device::fmx<uchar3, ushort3, float3, double3, op>, device::fmx<uchar4, ushort4, float4, double4, op>  },
                },
                {
                    { device::fmx<uchar, ushort, double, uchar, op>, device::fmx<uchar2, ushort2, double2, uchar2, op>, device::fmx<uchar3, ushort3, double3, uchar3, op>, device::fmx<uchar4, ushort4, double4, uchar4, op>  },
                    { device::fmx<uchar, ushort, double, schar, op>, device::fmx<uchar2, ushort2, double2, char2, op>, device::fmx<uchar3, ushort3, double3, char3, op>, device::fmx<uchar4, ushort4, double4, char4, op>  },
                    { device::fmx<uchar, ushort, double, ushort, op>, device::fmx<uchar2, ushort2, double2, ushort2, op>, device::fmx<uchar3, ushort3, double3, ushort3, op>, device::fmx<uchar4, ushort4, double4, ushort4, op>  },
                    { device::fmx<uchar, ushort, double, short, op>, device::fmx<uchar2, ushort2, double2, short2, op>, device::fmx<uchar3, ushort3, double3, short3, op>, device::fmx<uchar4, ushort4, double4, short4, op>  },
                    { device::fmx<uchar, ushort, double, int, op>, device::fmx<uchar2, ushort2, double2, int2, op>, device::fmx<uchar3, ushort3, double3, int3, op>, device::fmx<uchar4, ushort4, double4, int4, op>  },
                    { device::fmx<uchar, ushort, double, float, op>, device::fmx<uchar2, ushort2, double2, float2, op>, device::fmx<uchar3, ushort3, double3, float3, op>, device::fmx<uchar4, ushort4, double4, float4, op>  },
                    { device::fmx<uchar, ushort, double, double, op>, device::fmx<uchar2, ushort2, double2, double2, op>, device::fmx<uchar3, ushort3, double3, double3, op>, device::fmx<uchar4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<uchar, short, uchar, uchar, op>, device::fmx<uchar2, short2, uchar2, uchar2, op>, device::fmx<uchar3, short3, uchar3, uchar3, op>, device::fmx<uchar4, short4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, short, uchar, schar, op>, device::fmx<uchar2, short2, uchar2, char2, op>, device::fmx<uchar3, short3, uchar3, char3, op>, device::fmx<uchar4, short4, uchar4, char4, op>  },
                    { device::fmx<uchar, short, uchar, ushort, op>, device::fmx<uchar2, short2, uchar2, ushort2, op>, device::fmx<uchar3, short3, uchar3, ushort3, op>, device::fmx<uchar4, short4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, short, uchar, short, op>, device::fmx<uchar2, short2, uchar2, short2, op>, device::fmx<uchar3, short3, uchar3, short3, op>, device::fmx<uchar4, short4, uchar4, short4, op>  },
                    { device::fmx<uchar, short, uchar, int, op>, device::fmx<uchar2, short2, uchar2, int2, op>, device::fmx<uchar3, short3, uchar3, int3, op>, device::fmx<uchar4, short4, uchar4, int4, op>  },
                    { device::fmx<uchar, short, uchar, float, op>, device::fmx<uchar2, short2, uchar2, float2, op>, device::fmx<uchar3, short3, uchar3, float3, op>, device::fmx<uchar4, short4, uchar4, float4, op>  },
                    { device::fmx<uchar, short, uchar, double, op>, device::fmx<uchar2, short2, uchar2, double2, op>, device::fmx<uchar3, short3, uchar3, double3, op>, device::fmx<uchar4, short4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<uchar, short, schar, uchar, op>, device::fmx<uchar2, short2, char2, uchar2, op>, device::fmx<uchar3, short3, char3, uchar3, op>, device::fmx<uchar4, short4, char4, uchar4, op>  },
                    { device::fmx<uchar, short, schar, schar, op>, device::fmx<uchar2, short2, char2, char2, op>, device::fmx<uchar3, short3, char3, char3, op>, device::fmx<uchar4, short4, char4, char4, op>  },
                    { device::fmx<uchar, short, schar, ushort, op>, device::fmx<uchar2, short2, char2, ushort2, op>, device::fmx<uchar3, short3, char3, ushort3, op>, device::fmx<uchar4, short4, char4, ushort4, op>  },
                    { device::fmx<uchar, short, schar, short, op>, device::fmx<uchar2, short2, char2, short2, op>, device::fmx<uchar3, short3, char3, short3, op>, device::fmx<uchar4, short4, char4, short4, op>  },
                    { device::fmx<uchar, short, schar, int, op>, device::fmx<uchar2, short2, char2, int2, op>, device::fmx<uchar3, short3, char3, int3, op>, device::fmx<uchar4, short4, char4, int4, op>  },
                    { device::fmx<uchar, short, schar, float, op>, device::fmx<uchar2, short2, char2, float2, op>, device::fmx<uchar3, short3, char3, float3, op>, device::fmx<uchar4, short4, char4, float4, op>  },
                    { device::fmx<uchar, short, schar, double, op>, device::fmx<uchar2, short2, char2, double2, op>, device::fmx<uchar3, short3, char3, double3, op>, device::fmx<uchar4, short4, char4, double4, op>  },
                },
                {
                    { device::fmx<uchar, short, ushort, uchar, op>, device::fmx<uchar2, short2, ushort2, uchar2, op>, device::fmx<uchar3, short3, ushort3, uchar3, op>, device::fmx<uchar4, short4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, short, ushort, schar, op>, device::fmx<uchar2, short2, ushort2, char2, op>, device::fmx<uchar3, short3, ushort3, char3, op>, device::fmx<uchar4, short4, ushort4, char4, op>  },
                    { device::fmx<uchar, short, ushort, ushort, op>, device::fmx<uchar2, short2, ushort2, ushort2, op>, device::fmx<uchar3, short3, ushort3, ushort3, op>, device::fmx<uchar4, short4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, short, ushort, short, op>, device::fmx<uchar2, short2, ushort2, short2, op>, device::fmx<uchar3, short3, ushort3, short3, op>, device::fmx<uchar4, short4, ushort4, short4, op>  },
                    { device::fmx<uchar, short, ushort, int, op>, device::fmx<uchar2, short2, ushort2, int2, op>, device::fmx<uchar3, short3, ushort3, int3, op>, device::fmx<uchar4, short4, ushort4, int4, op>  },
                    { device::fmx<uchar, short, ushort, float, op>, device::fmx<uchar2, short2, ushort2, float2, op>, device::fmx<uchar3, short3, ushort3, float3, op>, device::fmx<uchar4, short4, ushort4, float4, op>  },
                    { device::fmx<uchar, short, ushort, double, op>, device::fmx<uchar2, short2, ushort2, double2, op>, device::fmx<uchar3, short3, ushort3, double3, op>, device::fmx<uchar4, short4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<uchar, short, short, uchar, op>, device::fmx<uchar2, short2, short2, uchar2, op>, device::fmx<uchar3, short3, short3, uchar3, op>, device::fmx<uchar4, short4, short4, uchar4, op>  },
                    { device::fmx<uchar, short, short, schar, op>, device::fmx<uchar2, short2, short2, char2, op>, device::fmx<uchar3, short3, short3, char3, op>, device::fmx<uchar4, short4, short4, char4, op>  },
                    { device::fmx<uchar, short, short, ushort, op>, device::fmx<uchar2, short2, short2, ushort2, op>, device::fmx<uchar3, short3, short3, ushort3, op>, device::fmx<uchar4, short4, short4, ushort4, op>  },
                    { device::fmx<uchar, short, short, short, op>, device::fmx<uchar2, short2, short2, short2, op>, device::fmx<uchar3, short3, short3, short3, op>, device::fmx<uchar4, short4, short4, short4, op>  },
                    { device::fmx<uchar, short, short, int, op>, device::fmx<uchar2, short2, short2, int2, op>, device::fmx<uchar3, short3, short3, int3, op>, device::fmx<uchar4, short4, short4, int4, op>  },
                    { device::fmx<uchar, short, short, float, op>, device::fmx<uchar2, short2, short2, float2, op>, device::fmx<uchar3, short3, short3, float3, op>, device::fmx<uchar4, short4, short4, float4, op>  },
                    { device::fmx<uchar, short, short, double, op>, device::fmx<uchar2, short2, short2, double2, op>, device::fmx<uchar3, short3, short3, double3, op>, device::fmx<uchar4, short4, short4, double4, op>  },
                },
                {
                    { device::fmx<uchar, short, int, uchar, op>, device::fmx<uchar2, short2, int2, uchar2, op>, device::fmx<uchar3, short3, int3, uchar3, op>, device::fmx<uchar4, short4, int4, uchar4, op>  },
                    { device::fmx<uchar, short, int, schar, op>, device::fmx<uchar2, short2, int2, char2, op>, device::fmx<uchar3, short3, int3, char3, op>, device::fmx<uchar4, short4, int4, char4, op>  },
                    { device::fmx<uchar, short, int, ushort, op>, device::fmx<uchar2, short2, int2, ushort2, op>, device::fmx<uchar3, short3, int3, ushort3, op>, device::fmx<uchar4, short4, int4, ushort4, op>  },
                    { device::fmx<uchar, short, int, short, op>, device::fmx<uchar2, short2, int2, short2, op>, device::fmx<uchar3, short3, int3, short3, op>, device::fmx<uchar4, short4, int4, short4, op>  },
                    { device::fmx<uchar, short, int, int, op>, device::fmx<uchar2, short2, int2, int2, op>, device::fmx<uchar3, short3, int3, int3, op>, device::fmx<uchar4, short4, int4, int4, op>  },
                    { device::fmx<uchar, short, int, float, op>, device::fmx<uchar2, short2, int2, float2, op>, device::fmx<uchar3, short3, int3, float3, op>, device::fmx<uchar4, short4, int4, float4, op>  },
                    { device::fmx<uchar, short, int, double, op>, device::fmx<uchar2, short2, int2, double2, op>, device::fmx<uchar3, short3, int3, double3, op>, device::fmx<uchar4, short4, int4, double4, op>  },
                },
                {
                    { device::fmx<uchar, short, float, uchar, op>, device::fmx<uchar2, short2, float2, uchar2, op>, device::fmx<uchar3, short3, float3, uchar3, op>, device::fmx<uchar4, short4, float4, uchar4, op>  },
                    { device::fmx<uchar, short, float, schar, op>, device::fmx<uchar2, short2, float2, char2, op>, device::fmx<uchar3, short3, float3, char3, op>, device::fmx<uchar4, short4, float4, char4, op>  },
                    { device::fmx<uchar, short, float, ushort, op>, device::fmx<uchar2, short2, float2, ushort2, op>, device::fmx<uchar3, short3, float3, ushort3, op>, device::fmx<uchar4, short4, float4, ushort4, op>  },
                    { device::fmx<uchar, short, float, short, op>, device::fmx<uchar2, short2, float2, short2, op>, device::fmx<uchar3, short3, float3, short3, op>, device::fmx<uchar4, short4, float4, short4, op>  },
                    { device::fmx<uchar, short, float, int, op>, device::fmx<uchar2, short2, float2, int2, op>, device::fmx<uchar3, short3, float3, int3, op>, device::fmx<uchar4, short4, float4, int4, op>  },
                    { device::fmx<uchar, short, float, float, op>, device::fmx<uchar2, short2, float2, float2, op>, device::fmx<uchar3, short3, float3, float3, op>, device::fmx<uchar4, short4, float4, float4, op>  },
                    { device::fmx<uchar, short, float, double, op>, device::fmx<uchar2, short2, float2, double2, op>, device::fmx<uchar3, short3, float3, double3, op>, device::fmx<uchar4, short4, float4, double4, op>  },
                },
                {
                    { device::fmx<uchar, short, double, uchar, op>, device::fmx<uchar2, short2, double2, uchar2, op>, device::fmx<uchar3, short3, double3, uchar3, op>, device::fmx<uchar4, short4, double4, uchar4, op>  },
                    { device::fmx<uchar, short, double, schar, op>, device::fmx<uchar2, short2, double2, char2, op>, device::fmx<uchar3, short3, double3, char3, op>, device::fmx<uchar4, short4, double4, char4, op>  },
                    { device::fmx<uchar, short, double, ushort, op>, device::fmx<uchar2, short2, double2, ushort2, op>, device::fmx<uchar3, short3, double3, ushort3, op>, device::fmx<uchar4, short4, double4, ushort4, op>  },
                    { device::fmx<uchar, short, double, short, op>, device::fmx<uchar2, short2, double2, short2, op>, device::fmx<uchar3, short3, double3, short3, op>, device::fmx<uchar4, short4, double4, short4, op>  },
                    { device::fmx<uchar, short, double, int, op>, device::fmx<uchar2, short2, double2, int2, op>, device::fmx<uchar3, short3, double3, int3, op>, device::fmx<uchar4, short4, double4, int4, op>  },
                    { device::fmx<uchar, short, double, float, op>, device::fmx<uchar2, short2, double2, float2, op>, device::fmx<uchar3, short3, double3, float3, op>, device::fmx<uchar4, short4, double4, float4, op>  },
                    { device::fmx<uchar, short, double, double, op>, device::fmx<uchar2, short2, double2, double2, op>, device::fmx<uchar3, short3, double3, double3, op>, device::fmx<uchar4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<uchar, int, uchar, uchar, op>, device::fmx<uchar2, int2, uchar2, uchar2, op>, device::fmx<uchar3, int3, uchar3, uchar3, op>, device::fmx<uchar4, int4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, int, uchar, schar, op>, device::fmx<uchar2, int2, uchar2, char2, op>, device::fmx<uchar3, int3, uchar3, char3, op>, device::fmx<uchar4, int4, uchar4, char4, op>  },
                    { device::fmx<uchar, int, uchar, ushort, op>, device::fmx<uchar2, int2, uchar2, ushort2, op>, device::fmx<uchar3, int3, uchar3, ushort3, op>, device::fmx<uchar4, int4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, int, uchar, short, op>, device::fmx<uchar2, int2, uchar2, short2, op>, device::fmx<uchar3, int3, uchar3, short3, op>, device::fmx<uchar4, int4, uchar4, short4, op>  },
                    { device::fmx<uchar, int, uchar, int, op>, device::fmx<uchar2, int2, uchar2, int2, op>, device::fmx<uchar3, int3, uchar3, int3, op>, device::fmx<uchar4, int4, uchar4, int4, op>  },
                    { device::fmx<uchar, int, uchar, float, op>, device::fmx<uchar2, int2, uchar2, float2, op>, device::fmx<uchar3, int3, uchar3, float3, op>, device::fmx<uchar4, int4, uchar4, float4, op>  },
                    { device::fmx<uchar, int, uchar, double, op>, device::fmx<uchar2, int2, uchar2, double2, op>, device::fmx<uchar3, int3, uchar3, double3, op>, device::fmx<uchar4, int4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<uchar, int, schar, uchar, op>, device::fmx<uchar2, int2, char2, uchar2, op>, device::fmx<uchar3, int3, char3, uchar3, op>, device::fmx<uchar4, int4, char4, uchar4, op>  },
                    { device::fmx<uchar, int, schar, schar, op>, device::fmx<uchar2, int2, char2, char2, op>, device::fmx<uchar3, int3, char3, char3, op>, device::fmx<uchar4, int4, char4, char4, op>  },
                    { device::fmx<uchar, int, schar, ushort, op>, device::fmx<uchar2, int2, char2, ushort2, op>, device::fmx<uchar3, int3, char3, ushort3, op>, device::fmx<uchar4, int4, char4, ushort4, op>  },
                    { device::fmx<uchar, int, schar, short, op>, device::fmx<uchar2, int2, char2, short2, op>, device::fmx<uchar3, int3, char3, short3, op>, device::fmx<uchar4, int4, char4, short4, op>  },
                    { device::fmx<uchar, int, schar, int, op>, device::fmx<uchar2, int2, char2, int2, op>, device::fmx<uchar3, int3, char3, int3, op>, device::fmx<uchar4, int4, char4, int4, op>  },
                    { device::fmx<uchar, int, schar, float, op>, device::fmx<uchar2, int2, char2, float2, op>, device::fmx<uchar3, int3, char3, float3, op>, device::fmx<uchar4, int4, char4, float4, op>  },
                    { device::fmx<uchar, int, schar, double, op>, device::fmx<uchar2, int2, char2, double2, op>, device::fmx<uchar3, int3, char3, double3, op>, device::fmx<uchar4, int4, char4, double4, op>  },
                },
                {
                    { device::fmx<uchar, int, ushort, uchar, op>, device::fmx<uchar2, int2, ushort2, uchar2, op>, device::fmx<uchar3, int3, ushort3, uchar3, op>, device::fmx<uchar4, int4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, int, ushort, schar, op>, device::fmx<uchar2, int2, ushort2, char2, op>, device::fmx<uchar3, int3, ushort3, char3, op>, device::fmx<uchar4, int4, ushort4, char4, op>  },
                    { device::fmx<uchar, int, ushort, ushort, op>, device::fmx<uchar2, int2, ushort2, ushort2, op>, device::fmx<uchar3, int3, ushort3, ushort3, op>, device::fmx<uchar4, int4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, int, ushort, short, op>, device::fmx<uchar2, int2, ushort2, short2, op>, device::fmx<uchar3, int3, ushort3, short3, op>, device::fmx<uchar4, int4, ushort4, short4, op>  },
                    { device::fmx<uchar, int, ushort, int, op>, device::fmx<uchar2, int2, ushort2, int2, op>, device::fmx<uchar3, int3, ushort3, int3, op>, device::fmx<uchar4, int4, ushort4, int4, op>  },
                    { device::fmx<uchar, int, ushort, float, op>, device::fmx<uchar2, int2, ushort2, float2, op>, device::fmx<uchar3, int3, ushort3, float3, op>, device::fmx<uchar4, int4, ushort4, float4, op>  },
                    { device::fmx<uchar, int, ushort, double, op>, device::fmx<uchar2, int2, ushort2, double2, op>, device::fmx<uchar3, int3, ushort3, double3, op>, device::fmx<uchar4, int4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<uchar, int, short, uchar, op>, device::fmx<uchar2, int2, short2, uchar2, op>, device::fmx<uchar3, int3, short3, uchar3, op>, device::fmx<uchar4, int4, short4, uchar4, op>  },
                    { device::fmx<uchar, int, short, schar, op>, device::fmx<uchar2, int2, short2, char2, op>, device::fmx<uchar3, int3, short3, char3, op>, device::fmx<uchar4, int4, short4, char4, op>  },
                    { device::fmx<uchar, int, short, ushort, op>, device::fmx<uchar2, int2, short2, ushort2, op>, device::fmx<uchar3, int3, short3, ushort3, op>, device::fmx<uchar4, int4, short4, ushort4, op>  },
                    { device::fmx<uchar, int, short, short, op>, device::fmx<uchar2, int2, short2, short2, op>, device::fmx<uchar3, int3, short3, short3, op>, device::fmx<uchar4, int4, short4, short4, op>  },
                    { device::fmx<uchar, int, short, int, op>, device::fmx<uchar2, int2, short2, int2, op>, device::fmx<uchar3, int3, short3, int3, op>, device::fmx<uchar4, int4, short4, int4, op>  },
                    { device::fmx<uchar, int, short, float, op>, device::fmx<uchar2, int2, short2, float2, op>, device::fmx<uchar3, int3, short3, float3, op>, device::fmx<uchar4, int4, short4, float4, op>  },
                    { device::fmx<uchar, int, short, double, op>, device::fmx<uchar2, int2, short2, double2, op>, device::fmx<uchar3, int3, short3, double3, op>, device::fmx<uchar4, int4, short4, double4, op>  },
                },
                {
                    { device::fmx<uchar, int, int, uchar, op>, device::fmx<uchar2, int2, int2, uchar2, op>, device::fmx<uchar3, int3, int3, uchar3, op>, device::fmx<uchar4, int4, int4, uchar4, op>  },
                    { device::fmx<uchar, int, int, schar, op>, device::fmx<uchar2, int2, int2, char2, op>, device::fmx<uchar3, int3, int3, char3, op>, device::fmx<uchar4, int4, int4, char4, op>  },
                    { device::fmx<uchar, int, int, ushort, op>, device::fmx<uchar2, int2, int2, ushort2, op>, device::fmx<uchar3, int3, int3, ushort3, op>, device::fmx<uchar4, int4, int4, ushort4, op>  },
                    { device::fmx<uchar, int, int, short, op>, device::fmx<uchar2, int2, int2, short2, op>, device::fmx<uchar3, int3, int3, short3, op>, device::fmx<uchar4, int4, int4, short4, op>  },
                    { device::fmx<uchar, int, int, int, op>, device::fmx<uchar2, int2, int2, int2, op>, device::fmx<uchar3, int3, int3, int3, op>, device::fmx<uchar4, int4, int4, int4, op>  },
                    { device::fmx<uchar, int, int, float, op>, device::fmx<uchar2, int2, int2, float2, op>, device::fmx<uchar3, int3, int3, float3, op>, device::fmx<uchar4, int4, int4, float4, op>  },
                    { device::fmx<uchar, int, int, double, op>, device::fmx<uchar2, int2, int2, double2, op>, device::fmx<uchar3, int3, int3, double3, op>, device::fmx<uchar4, int4, int4, double4, op>  },
                },
                {
                    { device::fmx<uchar, int, float, uchar, op>, device::fmx<uchar2, int2, float2, uchar2, op>, device::fmx<uchar3, int3, float3, uchar3, op>, device::fmx<uchar4, int4, float4, uchar4, op>  },
                    { device::fmx<uchar, int, float, schar, op>, device::fmx<uchar2, int2, float2, char2, op>, device::fmx<uchar3, int3, float3, char3, op>, device::fmx<uchar4, int4, float4, char4, op>  },
                    { device::fmx<uchar, int, float, ushort, op>, device::fmx<uchar2, int2, float2, ushort2, op>, device::fmx<uchar3, int3, float3, ushort3, op>, device::fmx<uchar4, int4, float4, ushort4, op>  },
                    { device::fmx<uchar, int, float, short, op>, device::fmx<uchar2, int2, float2, short2, op>, device::fmx<uchar3, int3, float3, short3, op>, device::fmx<uchar4, int4, float4, short4, op>  },
                    { device::fmx<uchar, int, float, int, op>, device::fmx<uchar2, int2, float2, int2, op>, device::fmx<uchar3, int3, float3, int3, op>, device::fmx<uchar4, int4, float4, int4, op>  },
                    { device::fmx<uchar, int, float, float, op>, device::fmx<uchar2, int2, float2, float2, op>, device::fmx<uchar3, int3, float3, float3, op>, device::fmx<uchar4, int4, float4, float4, op>  },
                    { device::fmx<uchar, int, float, double, op>, device::fmx<uchar2, int2, float2, double2, op>, device::fmx<uchar3, int3, float3, double3, op>, device::fmx<uchar4, int4, float4, double4, op>  },
                },
                {
                    { device::fmx<uchar, int, double, uchar, op>, device::fmx<uchar2, int2, double2, uchar2, op>, device::fmx<uchar3, int3, double3, uchar3, op>, device::fmx<uchar4, int4, double4, uchar4, op>  },
                    { device::fmx<uchar, int, double, schar, op>, device::fmx<uchar2, int2, double2, char2, op>, device::fmx<uchar3, int3, double3, char3, op>, device::fmx<uchar4, int4, double4, char4, op>  },
                    { device::fmx<uchar, int, double, ushort, op>, device::fmx<uchar2, int2, double2, ushort2, op>, device::fmx<uchar3, int3, double3, ushort3, op>, device::fmx<uchar4, int4, double4, ushort4, op>  },
                    { device::fmx<uchar, int, double, short, op>, device::fmx<uchar2, int2, double2, short2, op>, device::fmx<uchar3, int3, double3, short3, op>, device::fmx<uchar4, int4, double4, short4, op>  },
                    { device::fmx<uchar, int, double, int, op>, device::fmx<uchar2, int2, double2, int2, op>, device::fmx<uchar3, int3, double3, int3, op>, device::fmx<uchar4, int4, double4, int4, op>  },
                    { device::fmx<uchar, int, double, float, op>, device::fmx<uchar2, int2, double2, float2, op>, device::fmx<uchar3, int3, double3, float3, op>, device::fmx<uchar4, int4, double4, float4, op>  },
                    { device::fmx<uchar, int, double, double, op>, device::fmx<uchar2, int2, double2, double2, op>, device::fmx<uchar3, int3, double3, double3, op>, device::fmx<uchar4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<uchar, float, uchar, uchar, op>, device::fmx<uchar2, float2, uchar2, uchar2, op>, device::fmx<uchar3, float3, uchar3, uchar3, op>, device::fmx<uchar4, float4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, float, uchar, schar, op>, device::fmx<uchar2, float2, uchar2, char2, op>, device::fmx<uchar3, float3, uchar3, char3, op>, device::fmx<uchar4, float4, uchar4, char4, op>  },
                    { device::fmx<uchar, float, uchar, ushort, op>, device::fmx<uchar2, float2, uchar2, ushort2, op>, device::fmx<uchar3, float3, uchar3, ushort3, op>, device::fmx<uchar4, float4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, float, uchar, short, op>, device::fmx<uchar2, float2, uchar2, short2, op>, device::fmx<uchar3, float3, uchar3, short3, op>, device::fmx<uchar4, float4, uchar4, short4, op>  },
                    { device::fmx<uchar, float, uchar, int, op>, device::fmx<uchar2, float2, uchar2, int2, op>, device::fmx<uchar3, float3, uchar3, int3, op>, device::fmx<uchar4, float4, uchar4, int4, op>  },
                    { device::fmx<uchar, float, uchar, float, op>, device::fmx<uchar2, float2, uchar2, float2, op>, device::fmx<uchar3, float3, uchar3, float3, op>, device::fmx<uchar4, float4, uchar4, float4, op>  },
                    { device::fmx<uchar, float, uchar, double, op>, device::fmx<uchar2, float2, uchar2, double2, op>, device::fmx<uchar3, float3, uchar3, double3, op>, device::fmx<uchar4, float4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<uchar, float, schar, uchar, op>, device::fmx<uchar2, float2, char2, uchar2, op>, device::fmx<uchar3, float3, char3, uchar3, op>, device::fmx<uchar4, float4, char4, uchar4, op>  },
                    { device::fmx<uchar, float, schar, schar, op>, device::fmx<uchar2, float2, char2, char2, op>, device::fmx<uchar3, float3, char3, char3, op>, device::fmx<uchar4, float4, char4, char4, op>  },
                    { device::fmx<uchar, float, schar, ushort, op>, device::fmx<uchar2, float2, char2, ushort2, op>, device::fmx<uchar3, float3, char3, ushort3, op>, device::fmx<uchar4, float4, char4, ushort4, op>  },
                    { device::fmx<uchar, float, schar, short, op>, device::fmx<uchar2, float2, char2, short2, op>, device::fmx<uchar3, float3, char3, short3, op>, device::fmx<uchar4, float4, char4, short4, op>  },
                    { device::fmx<uchar, float, schar, int, op>, device::fmx<uchar2, float2, char2, int2, op>, device::fmx<uchar3, float3, char3, int3, op>, device::fmx<uchar4, float4, char4, int4, op>  },
                    { device::fmx<uchar, float, schar, float, op>, device::fmx<uchar2, float2, char2, float2, op>, device::fmx<uchar3, float3, char3, float3, op>, device::fmx<uchar4, float4, char4, float4, op>  },
                    { device::fmx<uchar, float, schar, double, op>, device::fmx<uchar2, float2, char2, double2, op>, device::fmx<uchar3, float3, char3, double3, op>, device::fmx<uchar4, float4, char4, double4, op>  },
                },
                {
                    { device::fmx<uchar, float, ushort, uchar, op>, device::fmx<uchar2, float2, ushort2, uchar2, op>, device::fmx<uchar3, float3, ushort3, uchar3, op>, device::fmx<uchar4, float4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, float, ushort, schar, op>, device::fmx<uchar2, float2, ushort2, char2, op>, device::fmx<uchar3, float3, ushort3, char3, op>, device::fmx<uchar4, float4, ushort4, char4, op>  },
                    { device::fmx<uchar, float, ushort, ushort, op>, device::fmx<uchar2, float2, ushort2, ushort2, op>, device::fmx<uchar3, float3, ushort3, ushort3, op>, device::fmx<uchar4, float4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, float, ushort, short, op>, device::fmx<uchar2, float2, ushort2, short2, op>, device::fmx<uchar3, float3, ushort3, short3, op>, device::fmx<uchar4, float4, ushort4, short4, op>  },
                    { device::fmx<uchar, float, ushort, int, op>, device::fmx<uchar2, float2, ushort2, int2, op>, device::fmx<uchar3, float3, ushort3, int3, op>, device::fmx<uchar4, float4, ushort4, int4, op>  },
                    { device::fmx<uchar, float, ushort, float, op>, device::fmx<uchar2, float2, ushort2, float2, op>, device::fmx<uchar3, float3, ushort3, float3, op>, device::fmx<uchar4, float4, ushort4, float4, op>  },
                    { device::fmx<uchar, float, ushort, double, op>, device::fmx<uchar2, float2, ushort2, double2, op>, device::fmx<uchar3, float3, ushort3, double3, op>, device::fmx<uchar4, float4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<uchar, float, short, uchar, op>, device::fmx<uchar2, float2, short2, uchar2, op>, device::fmx<uchar3, float3, short3, uchar3, op>, device::fmx<uchar4, float4, short4, uchar4, op>  },
                    { device::fmx<uchar, float, short, schar, op>, device::fmx<uchar2, float2, short2, char2, op>, device::fmx<uchar3, float3, short3, char3, op>, device::fmx<uchar4, float4, short4, char4, op>  },
                    { device::fmx<uchar, float, short, ushort, op>, device::fmx<uchar2, float2, short2, ushort2, op>, device::fmx<uchar3, float3, short3, ushort3, op>, device::fmx<uchar4, float4, short4, ushort4, op>  },
                    { device::fmx<uchar, float, short, short, op>, device::fmx<uchar2, float2, short2, short2, op>, device::fmx<uchar3, float3, short3, short3, op>, device::fmx<uchar4, float4, short4, short4, op>  },
                    { device::fmx<uchar, float, short, int, op>, device::fmx<uchar2, float2, short2, int2, op>, device::fmx<uchar3, float3, short3, int3, op>, device::fmx<uchar4, float4, short4, int4, op>  },
                    { device::fmx<uchar, float, short, float, op>, device::fmx<uchar2, float2, short2, float2, op>, device::fmx<uchar3, float3, short3, float3, op>, device::fmx<uchar4, float4, short4, float4, op>  },
                    { device::fmx<uchar, float, short, double, op>, device::fmx<uchar2, float2, short2, double2, op>, device::fmx<uchar3, float3, short3, double3, op>, device::fmx<uchar4, float4, short4, double4, op>  },
                },
                {
                    { device::fmx<uchar, float, int, uchar, op>, device::fmx<uchar2, float2, int2, uchar2, op>, device::fmx<uchar3, float3, int3, uchar3, op>, device::fmx<uchar4, float4, int4, uchar4, op>  },
                    { device::fmx<uchar, float, int, schar, op>, device::fmx<uchar2, float2, int2, char2, op>, device::fmx<uchar3, float3, int3, char3, op>, device::fmx<uchar4, float4, int4, char4, op>  },
                    { device::fmx<uchar, float, int, ushort, op>, device::fmx<uchar2, float2, int2, ushort2, op>, device::fmx<uchar3, float3, int3, ushort3, op>, device::fmx<uchar4, float4, int4, ushort4, op>  },
                    { device::fmx<uchar, float, int, short, op>, device::fmx<uchar2, float2, int2, short2, op>, device::fmx<uchar3, float3, int3, short3, op>, device::fmx<uchar4, float4, int4, short4, op>  },
                    { device::fmx<uchar, float, int, int, op>, device::fmx<uchar2, float2, int2, int2, op>, device::fmx<uchar3, float3, int3, int3, op>, device::fmx<uchar4, float4, int4, int4, op>  },
                    { device::fmx<uchar, float, int, float, op>, device::fmx<uchar2, float2, int2, float2, op>, device::fmx<uchar3, float3, int3, float3, op>, device::fmx<uchar4, float4, int4, float4, op>  },
                    { device::fmx<uchar, float, int, double, op>, device::fmx<uchar2, float2, int2, double2, op>, device::fmx<uchar3, float3, int3, double3, op>, device::fmx<uchar4, float4, int4, double4, op>  },
                },
                {
                    { device::fmx<uchar, float, float, uchar, op>, device::fmx<uchar2, float2, float2, uchar2, op>, device::fmx<uchar3, float3, float3, uchar3, op>, device::fmx<uchar4, float4, float4, uchar4, op>  },
                    { device::fmx<uchar, float, float, schar, op>, device::fmx<uchar2, float2, float2, char2, op>, device::fmx<uchar3, float3, float3, char3, op>, device::fmx<uchar4, float4, float4, char4, op>  },
                    { device::fmx<uchar, float, float, ushort, op>, device::fmx<uchar2, float2, float2, ushort2, op>, device::fmx<uchar3, float3, float3, ushort3, op>, device::fmx<uchar4, float4, float4, ushort4, op>  },
                    { device::fmx<uchar, float, float, short, op>, device::fmx<uchar2, float2, float2, short2, op>, device::fmx<uchar3, float3, float3, short3, op>, device::fmx<uchar4, float4, float4, short4, op>  },
                    { device::fmx<uchar, float, float, int, op>, device::fmx<uchar2, float2, float2, int2, op>, device::fmx<uchar3, float3, float3, int3, op>, device::fmx<uchar4, float4, float4, int4, op>  },
                    { device::fmx<uchar, float, float, float, op>, device::fmx<uchar2, float2, float2, float2, op>, device::fmx<uchar3, float3, float3, float3, op>, device::fmx<uchar4, float4, float4, float4, op>  },
                    { device::fmx<uchar, float, float, double, op>, device::fmx<uchar2, float2, float2, double2, op>, device::fmx<uchar3, float3, float3, double3, op>, device::fmx<uchar4, float4, float4, double4, op>  },
                },
                {
                    { device::fmx<uchar, float, double, uchar, op>, device::fmx<uchar2, float2, double2, uchar2, op>, device::fmx<uchar3, float3, double3, uchar3, op>, device::fmx<uchar4, float4, double4, uchar4, op>  },
                    { device::fmx<uchar, float, double, schar, op>, device::fmx<uchar2, float2, double2, char2, op>, device::fmx<uchar3, float3, double3, char3, op>, device::fmx<uchar4, float4, double4, char4, op>  },
                    { device::fmx<uchar, float, double, ushort, op>, device::fmx<uchar2, float2, double2, ushort2, op>, device::fmx<uchar3, float3, double3, ushort3, op>, device::fmx<uchar4, float4, double4, ushort4, op>  },
                    { device::fmx<uchar, float, double, short, op>, device::fmx<uchar2, float2, double2, short2, op>, device::fmx<uchar3, float3, double3, short3, op>, device::fmx<uchar4, float4, double4, short4, op>  },
                    { device::fmx<uchar, float, double, int, op>, device::fmx<uchar2, float2, double2, int2, op>, device::fmx<uchar3, float3, double3, int3, op>, device::fmx<uchar4, float4, double4, int4, op>  },
                    { device::fmx<uchar, float, double, float, op>, device::fmx<uchar2, float2, double2, float2, op>, device::fmx<uchar3, float3, double3, float3, op>, device::fmx<uchar4, float4, double4, float4, op>  },
                    { device::fmx<uchar, float, double, double, op>, device::fmx<uchar2, float2, double2, double2, op>, device::fmx<uchar3, float3, double3, double3, op>, device::fmx<uchar4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<uchar, double, uchar, uchar, op>, device::fmx<uchar2, double2, uchar2, uchar2, op>, device::fmx<uchar3, double3, uchar3, uchar3, op>, device::fmx<uchar4, double4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, double, uchar, schar, op>, device::fmx<uchar2, double2, uchar2, char2, op>, device::fmx<uchar3, double3, uchar3, char3, op>, device::fmx<uchar4, double4, uchar4, char4, op>  },
                    { device::fmx<uchar, double, uchar, ushort, op>, device::fmx<uchar2, double2, uchar2, ushort2, op>, device::fmx<uchar3, double3, uchar3, ushort3, op>, device::fmx<uchar4, double4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, double, uchar, short, op>, device::fmx<uchar2, double2, uchar2, short2, op>, device::fmx<uchar3, double3, uchar3, short3, op>, device::fmx<uchar4, double4, uchar4, short4, op>  },
                    { device::fmx<uchar, double, uchar, int, op>, device::fmx<uchar2, double2, uchar2, int2, op>, device::fmx<uchar3, double3, uchar3, int3, op>, device::fmx<uchar4, double4, uchar4, int4, op>  },
                    { device::fmx<uchar, double, uchar, float, op>, device::fmx<uchar2, double2, uchar2, float2, op>, device::fmx<uchar3, double3, uchar3, float3, op>, device::fmx<uchar4, double4, uchar4, float4, op>  },
                    { device::fmx<uchar, double, uchar, double, op>, device::fmx<uchar2, double2, uchar2, double2, op>, device::fmx<uchar3, double3, uchar3, double3, op>, device::fmx<uchar4, double4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<uchar, double, schar, uchar, op>, device::fmx<uchar2, double2, char2, uchar2, op>, device::fmx<uchar3, double3, char3, uchar3, op>, device::fmx<uchar4, double4, char4, uchar4, op>  },
                    { device::fmx<uchar, double, schar, schar, op>, device::fmx<uchar2, double2, char2, char2, op>, device::fmx<uchar3, double3, char3, char3, op>, device::fmx<uchar4, double4, char4, char4, op>  },
                    { device::fmx<uchar, double, schar, ushort, op>, device::fmx<uchar2, double2, char2, ushort2, op>, device::fmx<uchar3, double3, char3, ushort3, op>, device::fmx<uchar4, double4, char4, ushort4, op>  },
                    { device::fmx<uchar, double, schar, short, op>, device::fmx<uchar2, double2, char2, short2, op>, device::fmx<uchar3, double3, char3, short3, op>, device::fmx<uchar4, double4, char4, short4, op>  },
                    { device::fmx<uchar, double, schar, int, op>, device::fmx<uchar2, double2, char2, int2, op>, device::fmx<uchar3, double3, char3, int3, op>, device::fmx<uchar4, double4, char4, int4, op>  },
                    { device::fmx<uchar, double, schar, float, op>, device::fmx<uchar2, double2, char2, float2, op>, device::fmx<uchar3, double3, char3, float3, op>, device::fmx<uchar4, double4, char4, float4, op>  },
                    { device::fmx<uchar, double, schar, double, op>, device::fmx<uchar2, double2, char2, double2, op>, device::fmx<uchar3, double3, char3, double3, op>, device::fmx<uchar4, double4, char4, double4, op>  },
                },
                {
                    { device::fmx<uchar, double, ushort, uchar, op>, device::fmx<uchar2, double2, ushort2, uchar2, op>, device::fmx<uchar3, double3, ushort3, uchar3, op>, device::fmx<uchar4, double4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, double, ushort, schar, op>, device::fmx<uchar2, double2, ushort2, char2, op>, device::fmx<uchar3, double3, ushort3, char3, op>, device::fmx<uchar4, double4, ushort4, char4, op>  },
                    { device::fmx<uchar, double, ushort, ushort, op>, device::fmx<uchar2, double2, ushort2, ushort2, op>, device::fmx<uchar3, double3, ushort3, ushort3, op>, device::fmx<uchar4, double4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, double, ushort, short, op>, device::fmx<uchar2, double2, ushort2, short2, op>, device::fmx<uchar3, double3, ushort3, short3, op>, device::fmx<uchar4, double4, ushort4, short4, op>  },
                    { device::fmx<uchar, double, ushort, int, op>, device::fmx<uchar2, double2, ushort2, int2, op>, device::fmx<uchar3, double3, ushort3, int3, op>, device::fmx<uchar4, double4, ushort4, int4, op>  },
                    { device::fmx<uchar, double, ushort, float, op>, device::fmx<uchar2, double2, ushort2, float2, op>, device::fmx<uchar3, double3, ushort3, float3, op>, device::fmx<uchar4, double4, ushort4, float4, op>  },
                    { device::fmx<uchar, double, ushort, double, op>, device::fmx<uchar2, double2, ushort2, double2, op>, device::fmx<uchar3, double3, ushort3, double3, op>, device::fmx<uchar4, double4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<uchar, double, short, uchar, op>, device::fmx<uchar2, double2, short2, uchar2, op>, device::fmx<uchar3, double3, short3, uchar3, op>, device::fmx<uchar4, double4, short4, uchar4, op>  },
                    { device::fmx<uchar, double, short, schar, op>, device::fmx<uchar2, double2, short2, char2, op>, device::fmx<uchar3, double3, short3, char3, op>, device::fmx<uchar4, double4, short4, char4, op>  },
                    { device::fmx<uchar, double, short, ushort, op>, device::fmx<uchar2, double2, short2, ushort2, op>, device::fmx<uchar3, double3, short3, ushort3, op>, device::fmx<uchar4, double4, short4, ushort4, op>  },
                    { device::fmx<uchar, double, short, short, op>, device::fmx<uchar2, double2, short2, short2, op>, device::fmx<uchar3, double3, short3, short3, op>, device::fmx<uchar4, double4, short4, short4, op>  },
                    { device::fmx<uchar, double, short, int, op>, device::fmx<uchar2, double2, short2, int2, op>, device::fmx<uchar3, double3, short3, int3, op>, device::fmx<uchar4, double4, short4, int4, op>  },
                    { device::fmx<uchar, double, short, float, op>, device::fmx<uchar2, double2, short2, float2, op>, device::fmx<uchar3, double3, short3, float3, op>, device::fmx<uchar4, double4, short4, float4, op>  },
                    { device::fmx<uchar, double, short, double, op>, device::fmx<uchar2, double2, short2, double2, op>, device::fmx<uchar3, double3, short3, double3, op>, device::fmx<uchar4, double4, short4, double4, op>  },
                },
                {
                    { device::fmx<uchar, double, int, uchar, op>, device::fmx<uchar2, double2, int2, uchar2, op>, device::fmx<uchar3, double3, int3, uchar3, op>, device::fmx<uchar4, double4, int4, uchar4, op>  },
                    { device::fmx<uchar, double, int, schar, op>, device::fmx<uchar2, double2, int2, char2, op>, device::fmx<uchar3, double3, int3, char3, op>, device::fmx<uchar4, double4, int4, char4, op>  },
                    { device::fmx<uchar, double, int, ushort, op>, device::fmx<uchar2, double2, int2, ushort2, op>, device::fmx<uchar3, double3, int3, ushort3, op>, device::fmx<uchar4, double4, int4, ushort4, op>  },
                    { device::fmx<uchar, double, int, short, op>, device::fmx<uchar2, double2, int2, short2, op>, device::fmx<uchar3, double3, int3, short3, op>, device::fmx<uchar4, double4, int4, short4, op>  },
                    { device::fmx<uchar, double, int, int, op>, device::fmx<uchar2, double2, int2, int2, op>, device::fmx<uchar3, double3, int3, int3, op>, device::fmx<uchar4, double4, int4, int4, op>  },
                    { device::fmx<uchar, double, int, float, op>, device::fmx<uchar2, double2, int2, float2, op>, device::fmx<uchar3, double3, int3, float3, op>, device::fmx<uchar4, double4, int4, float4, op>  },
                    { device::fmx<uchar, double, int, double, op>, device::fmx<uchar2, double2, int2, double2, op>, device::fmx<uchar3, double3, int3, double3, op>, device::fmx<uchar4, double4, int4, double4, op>  },
                },
                {
                    { device::fmx<uchar, double, float, uchar, op>, device::fmx<uchar2, double2, float2, uchar2, op>, device::fmx<uchar3, double3, float3, uchar3, op>, device::fmx<uchar4, double4, float4, uchar4, op>  },
                    { device::fmx<uchar, double, float, schar, op>, device::fmx<uchar2, double2, float2, char2, op>, device::fmx<uchar3, double3, float3, char3, op>, device::fmx<uchar4, double4, float4, char4, op>  },
                    { device::fmx<uchar, double, float, ushort, op>, device::fmx<uchar2, double2, float2, ushort2, op>, device::fmx<uchar3, double3, float3, ushort3, op>, device::fmx<uchar4, double4, float4, ushort4, op>  },
                    { device::fmx<uchar, double, float, short, op>, device::fmx<uchar2, double2, float2, short2, op>, device::fmx<uchar3, double3, float3, short3, op>, device::fmx<uchar4, double4, float4, short4, op>  },
                    { device::fmx<uchar, double, float, int, op>, device::fmx<uchar2, double2, float2, int2, op>, device::fmx<uchar3, double3, float3, int3, op>, device::fmx<uchar4, double4, float4, int4, op>  },
                    { device::fmx<uchar, double, float, float, op>, device::fmx<uchar2, double2, float2, float2, op>, device::fmx<uchar3, double3, float3, float3, op>, device::fmx<uchar4, double4, float4, float4, op>  },
                    { device::fmx<uchar, double, float, double, op>, device::fmx<uchar2, double2, float2, double2, op>, device::fmx<uchar3, double3, float3, double3, op>, device::fmx<uchar4, double4, float4, double4, op>  },
                },
                {
                    { device::fmx<uchar, double, double, uchar, op>, device::fmx<uchar2, double2, double2, uchar2, op>, device::fmx<uchar3, double3, double3, uchar3, op>, device::fmx<uchar4, double4, double4, uchar4, op>  },
                    { device::fmx<uchar, double, double, schar, op>, device::fmx<uchar2, double2, double2, char2, op>, device::fmx<uchar3, double3, double3, char3, op>, device::fmx<uchar4, double4, double4, char4, op>  },
                    { device::fmx<uchar, double, double, ushort, op>, device::fmx<uchar2, double2, double2, ushort2, op>, device::fmx<uchar3, double3, double3, ushort3, op>, device::fmx<uchar4, double4, double4, ushort4, op>  },
                    { device::fmx<uchar, double, double, short, op>, device::fmx<uchar2, double2, double2, short2, op>, device::fmx<uchar3, double3, double3, short3, op>, device::fmx<uchar4, double4, double4, short4, op>  },
                    { device::fmx<uchar, double, double, int, op>, device::fmx<uchar2, double2, double2, int2, op>, device::fmx<uchar3, double3, double3, int3, op>, device::fmx<uchar4, double4, double4, int4, op>  },
                    { device::fmx<uchar, double, double, float, op>, device::fmx<uchar2, double2, double2, float2, op>, device::fmx<uchar3, double3, double3, float3, op>, device::fmx<uchar4, double4, double4, float4, op>  },
                    { device::fmx<uchar, double, double, double, op>, device::fmx<uchar2, double2, double2, double2, op>, device::fmx<uchar3, double3, double3, double3, op>, device::fmx<uchar4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { device::fmx<schar, uchar, uchar, uchar, op>, device::fmx<char2, uchar2, uchar2, uchar2, op>, device::fmx<char3, uchar3, uchar3, uchar3, op>, device::fmx<char4, uchar4, uchar4, uchar4, op>  },
                    { device::fmx<schar, uchar, uchar, schar, op>, device::fmx<char2, uchar2, uchar2, char2, op>, device::fmx<char3, uchar3, uchar3, char3, op>, device::fmx<char4, uchar4, uchar4, char4, op>  },
                    { device::fmx<schar, uchar, uchar, ushort, op>, device::fmx<char2, uchar2, uchar2, ushort2, op>, device::fmx<char3, uchar3, uchar3, ushort3, op>, device::fmx<char4, uchar4, uchar4, ushort4, op>  },
                    { device::fmx<schar, uchar, uchar, short, op>, device::fmx<char2, uchar2, uchar2, short2, op>, device::fmx<char3, uchar3, uchar3, short3, op>, device::fmx<char4, uchar4, uchar4, short4, op>  },
                    { device::fmx<schar, uchar, uchar, int, op>, device::fmx<char2, uchar2, uchar2, int2, op>, device::fmx<char3, uchar3, uchar3, int3, op>, device::fmx<char4, uchar4, uchar4, int4, op>  },
                    { device::fmx<schar, uchar, uchar, float, op>, device::fmx<char2, uchar2, uchar2, float2, op>, device::fmx<char3, uchar3, uchar3, float3, op>, device::fmx<char4, uchar4, uchar4, float4, op>  },
                    { device::fmx<schar, uchar, uchar, double, op>, device::fmx<char2, uchar2, uchar2, double2, op>, device::fmx<char3, uchar3, uchar3, double3, op>, device::fmx<char4, uchar4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<schar, uchar, schar, uchar, op>, device::fmx<char2, uchar2, char2, uchar2, op>, device::fmx<char3, uchar3, char3, uchar3, op>, device::fmx<char4, uchar4, char4, uchar4, op>  },
                    { device::fmx<schar, uchar, schar, schar, op>, device::fmx<char2, uchar2, char2, char2, op>, device::fmx<char3, uchar3, char3, char3, op>, device::fmx<char4, uchar4, char4, char4, op>  },
                    { device::fmx<schar, uchar, schar, ushort, op>, device::fmx<char2, uchar2, char2, ushort2, op>, device::fmx<char3, uchar3, char3, ushort3, op>, device::fmx<char4, uchar4, char4, ushort4, op>  },
                    { device::fmx<schar, uchar, schar, short, op>, device::fmx<char2, uchar2, char2, short2, op>, device::fmx<char3, uchar3, char3, short3, op>, device::fmx<char4, uchar4, char4, short4, op>  },
                    { device::fmx<schar, uchar, schar, int, op>, device::fmx<char2, uchar2, char2, int2, op>, device::fmx<char3, uchar3, char3, int3, op>, device::fmx<char4, uchar4, char4, int4, op>  },
                    { device::fmx<schar, uchar, schar, float, op>, device::fmx<char2, uchar2, char2, float2, op>, device::fmx<char3, uchar3, char3, float3, op>, device::fmx<char4, uchar4, char4, float4, op>  },
                    { device::fmx<schar, uchar, schar, double, op>, device::fmx<char2, uchar2, char2, double2, op>, device::fmx<char3, uchar3, char3, double3, op>, device::fmx<char4, uchar4, char4, double4, op>  },
                },
                {
                    { device::fmx<schar, uchar, ushort, uchar, op>, device::fmx<char2, uchar2, ushort2, uchar2, op>, device::fmx<char3, uchar3, ushort3, uchar3, op>, device::fmx<char4, uchar4, ushort4, uchar4, op>  },
                    { device::fmx<schar, uchar, ushort, schar, op>, device::fmx<char2, uchar2, ushort2, char2, op>, device::fmx<char3, uchar3, ushort3, char3, op>, device::fmx<char4, uchar4, ushort4, char4, op>  },
                    { device::fmx<schar, uchar, ushort, ushort, op>, device::fmx<char2, uchar2, ushort2, ushort2, op>, device::fmx<char3, uchar3, ushort3, ushort3, op>, device::fmx<char4, uchar4, ushort4, ushort4, op>  },
                    { device::fmx<schar, uchar, ushort, short, op>, device::fmx<char2, uchar2, ushort2, short2, op>, device::fmx<char3, uchar3, ushort3, short3, op>, device::fmx<char4, uchar4, ushort4, short4, op>  },
                    { device::fmx<schar, uchar, ushort, int, op>, device::fmx<char2, uchar2, ushort2, int2, op>, device::fmx<char3, uchar3, ushort3, int3, op>, device::fmx<char4, uchar4, ushort4, int4, op>  },
                    { device::fmx<schar, uchar, ushort, float, op>, device::fmx<char2, uchar2, ushort2, float2, op>, device::fmx<char3, uchar3, ushort3, float3, op>, device::fmx<char4, uchar4, ushort4, float4, op>  },
                    { device::fmx<schar, uchar, ushort, double, op>, device::fmx<char2, uchar2, ushort2, double2, op>, device::fmx<char3, uchar3, ushort3, double3, op>, device::fmx<char4, uchar4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<schar, uchar, short, uchar, op>, device::fmx<char2, uchar2, short2, uchar2, op>, device::fmx<char3, uchar3, short3, uchar3, op>, device::fmx<char4, uchar4, short4, uchar4, op>  },
                    { device::fmx<schar, uchar, short, schar, op>, device::fmx<char2, uchar2, short2, char2, op>, device::fmx<char3, uchar3, short3, char3, op>, device::fmx<char4, uchar4, short4, char4, op>  },
                    { device::fmx<schar, uchar, short, ushort, op>, device::fmx<char2, uchar2, short2, ushort2, op>, device::fmx<char3, uchar3, short3, ushort3, op>, device::fmx<char4, uchar4, short4, ushort4, op>  },
                    { device::fmx<schar, uchar, short, short, op>, device::fmx<char2, uchar2, short2, short2, op>, device::fmx<char3, uchar3, short3, short3, op>, device::fmx<char4, uchar4, short4, short4, op>  },
                    { device::fmx<schar, uchar, short, int, op>, device::fmx<char2, uchar2, short2, int2, op>, device::fmx<char3, uchar3, short3, int3, op>, device::fmx<char4, uchar4, short4, int4, op>  },
                    { device::fmx<schar, uchar, short, float, op>, device::fmx<char2, uchar2, short2, float2, op>, device::fmx<char3, uchar3, short3, float3, op>, device::fmx<char4, uchar4, short4, float4, op>  },
                    { device::fmx<schar, uchar, short, double, op>, device::fmx<char2, uchar2, short2, double2, op>, device::fmx<char3, uchar3, short3, double3, op>, device::fmx<char4, uchar4, short4, double4, op>  },
                },
                {
                    { device::fmx<schar, uchar, int, uchar, op>, device::fmx<char2, uchar2, int2, uchar2, op>, device::fmx<char3, uchar3, int3, uchar3, op>, device::fmx<char4, uchar4, int4, uchar4, op>  },
                    { device::fmx<schar, uchar, int, schar, op>, device::fmx<char2, uchar2, int2, char2, op>, device::fmx<char3, uchar3, int3, char3, op>, device::fmx<char4, uchar4, int4, char4, op>  },
                    { device::fmx<schar, uchar, int, ushort, op>, device::fmx<char2, uchar2, int2, ushort2, op>, device::fmx<char3, uchar3, int3, ushort3, op>, device::fmx<char4, uchar4, int4, ushort4, op>  },
                    { device::fmx<schar, uchar, int, short, op>, device::fmx<char2, uchar2, int2, short2, op>, device::fmx<char3, uchar3, int3, short3, op>, device::fmx<char4, uchar4, int4, short4, op>  },
                    { device::fmx<schar, uchar, int, int, op>, device::fmx<char2, uchar2, int2, int2, op>, device::fmx<char3, uchar3, int3, int3, op>, device::fmx<char4, uchar4, int4, int4, op>  },
                    { device::fmx<schar, uchar, int, float, op>, device::fmx<char2, uchar2, int2, float2, op>, device::fmx<char3, uchar3, int3, float3, op>, device::fmx<char4, uchar4, int4, float4, op>  },
                    { device::fmx<schar, uchar, int, double, op>, device::fmx<char2, uchar2, int2, double2, op>, device::fmx<char3, uchar3, int3, double3, op>, device::fmx<char4, uchar4, int4, double4, op>  },
                },
                {
                    { device::fmx<schar, uchar, float, uchar, op>, device::fmx<char2, uchar2, float2, uchar2, op>, device::fmx<char3, uchar3, float3, uchar3, op>, device::fmx<char4, uchar4, float4, uchar4, op>  },
                    { device::fmx<schar, uchar, float, schar, op>, device::fmx<char2, uchar2, float2, char2, op>, device::fmx<char3, uchar3, float3, char3, op>, device::fmx<char4, uchar4, float4, char4, op>  },
                    { device::fmx<schar, uchar, float, ushort, op>, device::fmx<char2, uchar2, float2, ushort2, op>, device::fmx<char3, uchar3, float3, ushort3, op>, device::fmx<char4, uchar4, float4, ushort4, op>  },
                    { device::fmx<schar, uchar, float, short, op>, device::fmx<char2, uchar2, float2, short2, op>, device::fmx<char3, uchar3, float3, short3, op>, device::fmx<char4, uchar4, float4, short4, op>  },
                    { device::fmx<schar, uchar, float, int, op>, device::fmx<char2, uchar2, float2, int2, op>, device::fmx<char3, uchar3, float3, int3, op>, device::fmx<char4, uchar4, float4, int4, op>  },
                    { device::fmx<schar, uchar, float, float, op>, device::fmx<char2, uchar2, float2, float2, op>, device::fmx<char3, uchar3, float3, float3, op>, device::fmx<char4, uchar4, float4, float4, op>  },
                    { device::fmx<schar, uchar, float, double, op>, device::fmx<char2, uchar2, float2, double2, op>, device::fmx<char3, uchar3, float3, double3, op>, device::fmx<char4, uchar4, float4, double4, op>  },
                },
                {
                    { device::fmx<schar, uchar, double, uchar, op>, device::fmx<char2, uchar2, double2, uchar2, op>, device::fmx<char3, uchar3, double3, uchar3, op>, device::fmx<char4, uchar4, double4, uchar4, op>  },
                    { device::fmx<schar, uchar, double, schar, op>, device::fmx<char2, uchar2, double2, char2, op>, device::fmx<char3, uchar3, double3, char3, op>, device::fmx<char4, uchar4, double4, char4, op>  },
                    { device::fmx<schar, uchar, double, ushort, op>, device::fmx<char2, uchar2, double2, ushort2, op>, device::fmx<char3, uchar3, double3, ushort3, op>, device::fmx<char4, uchar4, double4, ushort4, op>  },
                    { device::fmx<schar, uchar, double, short, op>, device::fmx<char2, uchar2, double2, short2, op>, device::fmx<char3, uchar3, double3, short3, op>, device::fmx<char4, uchar4, double4, short4, op>  },
                    { device::fmx<schar, uchar, double, int, op>, device::fmx<char2, uchar2, double2, int2, op>, device::fmx<char3, uchar3, double3, int3, op>, device::fmx<char4, uchar4, double4, int4, op>  },
                    { device::fmx<schar, uchar, double, float, op>, device::fmx<char2, uchar2, double2, float2, op>, device::fmx<char3, uchar3, double3, float3, op>, device::fmx<char4, uchar4, double4, float4, op>  },
                    { device::fmx<schar, uchar, double, double, op>, device::fmx<char2, uchar2, double2, double2, op>, device::fmx<char3, uchar3, double3, double3, op>, device::fmx<char4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<schar, schar, uchar, uchar, op>, device::fmx<char2, char2, uchar2, uchar2, op>, device::fmx<char3, char3, uchar3, uchar3, op>, device::fmx<char4, char4, uchar4, uchar4, op>  },
                    { device::fmx<schar, schar, uchar, schar, op>, device::fmx<char2, char2, uchar2, char2, op>, device::fmx<char3, char3, uchar3, char3, op>, device::fmx<char4, char4, uchar4, char4, op>  },
                    { device::fmx<schar, schar, uchar, ushort, op>, device::fmx<char2, char2, uchar2, ushort2, op>, device::fmx<char3, char3, uchar3, ushort3, op>, device::fmx<char4, char4, uchar4, ushort4, op>  },
                    { device::fmx<schar, schar, uchar, short, op>, device::fmx<char2, char2, uchar2, short2, op>, device::fmx<char3, char3, uchar3, short3, op>, device::fmx<char4, char4, uchar4, short4, op>  },
                    { device::fmx<schar, schar, uchar, int, op>, device::fmx<char2, char2, uchar2, int2, op>, device::fmx<char3, char3, uchar3, int3, op>, device::fmx<char4, char4, uchar4, int4, op>  },
                    { device::fmx<schar, schar, uchar, float, op>, device::fmx<char2, char2, uchar2, float2, op>, device::fmx<char3, char3, uchar3, float3, op>, device::fmx<char4, char4, uchar4, float4, op>  },
                    { device::fmx<schar, schar, uchar, double, op>, device::fmx<char2, char2, uchar2, double2, op>, device::fmx<char3, char3, uchar3, double3, op>, device::fmx<char4, char4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<schar, schar, schar, uchar, op>, device::fmx<char2, char2, char2, uchar2, op>, device::fmx<char3, char3, char3, uchar3, op>, device::fmx<char4, char4, char4, uchar4, op>  },
                    { device::fmx<schar, schar, schar, schar, op>, device::fmx<char2, char2, char2, char2, op>, device::fmx<char3, char3, char3, char3, op>, device::fmx<char4, char4, char4, char4, op>  },
                    { device::fmx<schar, schar, schar, ushort, op>, device::fmx<char2, char2, char2, ushort2, op>, device::fmx<char3, char3, char3, ushort3, op>, device::fmx<char4, char4, char4, ushort4, op>  },
                    { device::fmx<schar, schar, schar, short, op>, device::fmx<char2, char2, char2, short2, op>, device::fmx<char3, char3, char3, short3, op>, device::fmx<char4, char4, char4, short4, op>  },
                    { device::fmx<schar, schar, schar, int, op>, device::fmx<char2, char2, char2, int2, op>, device::fmx<char3, char3, char3, int3, op>, device::fmx<char4, char4, char4, int4, op>  },
                    { device::fmx<schar, schar, schar, float, op>, device::fmx<char2, char2, char2, float2, op>, device::fmx<char3, char3, char3, float3, op>, device::fmx<char4, char4, char4, float4, op>  },
                    { device::fmx<schar, schar, schar, double, op>, device::fmx<char2, char2, char2, double2, op>, device::fmx<char3, char3, char3, double3, op>, device::fmx<char4, char4, char4, double4, op>  },
                },
                {
                    { device::fmx<schar, schar, ushort, uchar, op>, device::fmx<char2, char2, ushort2, uchar2, op>, device::fmx<char3, char3, ushort3, uchar3, op>, device::fmx<char4, char4, ushort4, uchar4, op>  },
                    { device::fmx<schar, schar, ushort, schar, op>, device::fmx<char2, char2, ushort2, char2, op>, device::fmx<char3, char3, ushort3, char3, op>, device::fmx<char4, char4, ushort4, char4, op>  },
                    { device::fmx<schar, schar, ushort, ushort, op>, device::fmx<char2, char2, ushort2, ushort2, op>, device::fmx<char3, char3, ushort3, ushort3, op>, device::fmx<char4, char4, ushort4, ushort4, op>  },
                    { device::fmx<schar, schar, ushort, short, op>, device::fmx<char2, char2, ushort2, short2, op>, device::fmx<char3, char3, ushort3, short3, op>, device::fmx<char4, char4, ushort4, short4, op>  },
                    { device::fmx<schar, schar, ushort, int, op>, device::fmx<char2, char2, ushort2, int2, op>, device::fmx<char3, char3, ushort3, int3, op>, device::fmx<char4, char4, ushort4, int4, op>  },
                    { device::fmx<schar, schar, ushort, float, op>, device::fmx<char2, char2, ushort2, float2, op>, device::fmx<char3, char3, ushort3, float3, op>, device::fmx<char4, char4, ushort4, float4, op>  },
                    { device::fmx<schar, schar, ushort, double, op>, device::fmx<char2, char2, ushort2, double2, op>, device::fmx<char3, char3, ushort3, double3, op>, device::fmx<char4, char4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<schar, schar, short, uchar, op>, device::fmx<char2, char2, short2, uchar2, op>, device::fmx<char3, char3, short3, uchar3, op>, device::fmx<char4, char4, short4, uchar4, op>  },
                    { device::fmx<schar, schar, short, schar, op>, device::fmx<char2, char2, short2, char2, op>, device::fmx<char3, char3, short3, char3, op>, device::fmx<char4, char4, short4, char4, op>  },
                    { device::fmx<schar, schar, short, ushort, op>, device::fmx<char2, char2, short2, ushort2, op>, device::fmx<char3, char3, short3, ushort3, op>, device::fmx<char4, char4, short4, ushort4, op>  },
                    { device::fmx<schar, schar, short, short, op>, device::fmx<char2, char2, short2, short2, op>, device::fmx<char3, char3, short3, short3, op>, device::fmx<char4, char4, short4, short4, op>  },
                    { device::fmx<schar, schar, short, int, op>, device::fmx<char2, char2, short2, int2, op>, device::fmx<char3, char3, short3, int3, op>, device::fmx<char4, char4, short4, int4, op>  },
                    { device::fmx<schar, schar, short, float, op>, device::fmx<char2, char2, short2, float2, op>, device::fmx<char3, char3, short3, float3, op>, device::fmx<char4, char4, short4, float4, op>  },
                    { device::fmx<schar, schar, short, double, op>, device::fmx<char2, char2, short2, double2, op>, device::fmx<char3, char3, short3, double3, op>, device::fmx<char4, char4, short4, double4, op>  },
                },
                {
                    { device::fmx<schar, schar, int, uchar, op>, device::fmx<char2, char2, int2, uchar2, op>, device::fmx<char3, char3, int3, uchar3, op>, device::fmx<char4, char4, int4, uchar4, op>  },
                    { device::fmx<schar, schar, int, schar, op>, device::fmx<char2, char2, int2, char2, op>, device::fmx<char3, char3, int3, char3, op>, device::fmx<char4, char4, int4, char4, op>  },
                    { device::fmx<schar, schar, int, ushort, op>, device::fmx<char2, char2, int2, ushort2, op>, device::fmx<char3, char3, int3, ushort3, op>, device::fmx<char4, char4, int4, ushort4, op>  },
                    { device::fmx<schar, schar, int, short, op>, device::fmx<char2, char2, int2, short2, op>, device::fmx<char3, char3, int3, short3, op>, device::fmx<char4, char4, int4, short4, op>  },
                    { device::fmx<schar, schar, int, int, op>, device::fmx<char2, char2, int2, int2, op>, device::fmx<char3, char3, int3, int3, op>, device::fmx<char4, char4, int4, int4, op>  },
                    { device::fmx<schar, schar, int, float, op>, device::fmx<char2, char2, int2, float2, op>, device::fmx<char3, char3, int3, float3, op>, device::fmx<char4, char4, int4, float4, op>  },
                    { device::fmx<schar, schar, int, double, op>, device::fmx<char2, char2, int2, double2, op>, device::fmx<char3, char3, int3, double3, op>, device::fmx<char4, char4, int4, double4, op>  },
                },
                {
                    { device::fmx<schar, schar, float, uchar, op>, device::fmx<char2, char2, float2, uchar2, op>, device::fmx<char3, char3, float3, uchar3, op>, device::fmx<char4, char4, float4, uchar4, op>  },
                    { device::fmx<schar, schar, float, schar, op>, device::fmx<char2, char2, float2, char2, op>, device::fmx<char3, char3, float3, char3, op>, device::fmx<char4, char4, float4, char4, op>  },
                    { device::fmx<schar, schar, float, ushort, op>, device::fmx<char2, char2, float2, ushort2, op>, device::fmx<char3, char3, float3, ushort3, op>, device::fmx<char4, char4, float4, ushort4, op>  },
                    { device::fmx<schar, schar, float, short, op>, device::fmx<char2, char2, float2, short2, op>, device::fmx<char3, char3, float3, short3, op>, device::fmx<char4, char4, float4, short4, op>  },
                    { device::fmx<schar, schar, float, int, op>, device::fmx<char2, char2, float2, int2, op>, device::fmx<char3, char3, float3, int3, op>, device::fmx<char4, char4, float4, int4, op>  },
                    { device::fmx<schar, schar, float, float, op>, device::fmx<char2, char2, float2, float2, op>, device::fmx<char3, char3, float3, float3, op>, device::fmx<char4, char4, float4, float4, op>  },
                    { device::fmx<schar, schar, float, double, op>, device::fmx<char2, char2, float2, double2, op>, device::fmx<char3, char3, float3, double3, op>, device::fmx<char4, char4, float4, double4, op>  },
                },
                {
                    { device::fmx<schar, schar, double, uchar, op>, device::fmx<char2, char2, double2, uchar2, op>, device::fmx<char3, char3, double3, uchar3, op>, device::fmx<char4, char4, double4, uchar4, op>  },
                    { device::fmx<schar, schar, double, schar, op>, device::fmx<char2, char2, double2, char2, op>, device::fmx<char3, char3, double3, char3, op>, device::fmx<char4, char4, double4, char4, op>  },
                    { device::fmx<schar, schar, double, ushort, op>, device::fmx<char2, char2, double2, ushort2, op>, device::fmx<char3, char3, double3, ushort3, op>, device::fmx<char4, char4, double4, ushort4, op>  },
                    { device::fmx<schar, schar, double, short, op>, device::fmx<char2, char2, double2, short2, op>, device::fmx<char3, char3, double3, short3, op>, device::fmx<char4, char4, double4, short4, op>  },
                    { device::fmx<schar, schar, double, int, op>, device::fmx<char2, char2, double2, int2, op>, device::fmx<char3, char3, double3, int3, op>, device::fmx<char4, char4, double4, int4, op>  },
                    { device::fmx<schar, schar, double, float, op>, device::fmx<char2, char2, double2, float2, op>, device::fmx<char3, char3, double3, float3, op>, device::fmx<char4, char4, double4, float4, op>  },
                    { device::fmx<schar, schar, double, double, op>, device::fmx<char2, char2, double2, double2, op>, device::fmx<char3, char3, double3, double3, op>, device::fmx<char4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<schar, ushort, uchar, uchar, op>, device::fmx<char2, ushort2, uchar2, uchar2, op>, device::fmx<char3, ushort3, uchar3, uchar3, op>, device::fmx<char4, ushort4, uchar4, uchar4, op>  },
                    { device::fmx<schar, ushort, uchar, schar, op>, device::fmx<char2, ushort2, uchar2, char2, op>, device::fmx<char3, ushort3, uchar3, char3, op>, device::fmx<char4, ushort4, uchar4, char4, op>  },
                    { device::fmx<schar, ushort, uchar, ushort, op>, device::fmx<char2, ushort2, uchar2, ushort2, op>, device::fmx<char3, ushort3, uchar3, ushort3, op>, device::fmx<char4, ushort4, uchar4, ushort4, op>  },
                    { device::fmx<schar, ushort, uchar, short, op>, device::fmx<char2, ushort2, uchar2, short2, op>, device::fmx<char3, ushort3, uchar3, short3, op>, device::fmx<char4, ushort4, uchar4, short4, op>  },
                    { device::fmx<schar, ushort, uchar, int, op>, device::fmx<char2, ushort2, uchar2, int2, op>, device::fmx<char3, ushort3, uchar3, int3, op>, device::fmx<char4, ushort4, uchar4, int4, op>  },
                    { device::fmx<schar, ushort, uchar, float, op>, device::fmx<char2, ushort2, uchar2, float2, op>, device::fmx<char3, ushort3, uchar3, float3, op>, device::fmx<char4, ushort4, uchar4, float4, op>  },
                    { device::fmx<schar, ushort, uchar, double, op>, device::fmx<char2, ushort2, uchar2, double2, op>, device::fmx<char3, ushort3, uchar3, double3, op>, device::fmx<char4, ushort4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<schar, ushort, schar, uchar, op>, device::fmx<char2, ushort2, char2, uchar2, op>, device::fmx<char3, ushort3, char3, uchar3, op>, device::fmx<char4, ushort4, char4, uchar4, op>  },
                    { device::fmx<schar, ushort, schar, schar, op>, device::fmx<char2, ushort2, char2, char2, op>, device::fmx<char3, ushort3, char3, char3, op>, device::fmx<char4, ushort4, char4, char4, op>  },
                    { device::fmx<schar, ushort, schar, ushort, op>, device::fmx<char2, ushort2, char2, ushort2, op>, device::fmx<char3, ushort3, char3, ushort3, op>, device::fmx<char4, ushort4, char4, ushort4, op>  },
                    { device::fmx<schar, ushort, schar, short, op>, device::fmx<char2, ushort2, char2, short2, op>, device::fmx<char3, ushort3, char3, short3, op>, device::fmx<char4, ushort4, char4, short4, op>  },
                    { device::fmx<schar, ushort, schar, int, op>, device::fmx<char2, ushort2, char2, int2, op>, device::fmx<char3, ushort3, char3, int3, op>, device::fmx<char4, ushort4, char4, int4, op>  },
                    { device::fmx<schar, ushort, schar, float, op>, device::fmx<char2, ushort2, char2, float2, op>, device::fmx<char3, ushort3, char3, float3, op>, device::fmx<char4, ushort4, char4, float4, op>  },
                    { device::fmx<schar, ushort, schar, double, op>, device::fmx<char2, ushort2, char2, double2, op>, device::fmx<char3, ushort3, char3, double3, op>, device::fmx<char4, ushort4, char4, double4, op>  },
                },
                {
                    { device::fmx<schar, ushort, ushort, uchar, op>, device::fmx<char2, ushort2, ushort2, uchar2, op>, device::fmx<char3, ushort3, ushort3, uchar3, op>, device::fmx<char4, ushort4, ushort4, uchar4, op>  },
                    { device::fmx<schar, ushort, ushort, schar, op>, device::fmx<char2, ushort2, ushort2, char2, op>, device::fmx<char3, ushort3, ushort3, char3, op>, device::fmx<char4, ushort4, ushort4, char4, op>  },
                    { device::fmx<schar, ushort, ushort, ushort, op>, device::fmx<char2, ushort2, ushort2, ushort2, op>, device::fmx<char3, ushort3, ushort3, ushort3, op>, device::fmx<char4, ushort4, ushort4, ushort4, op>  },
                    { device::fmx<schar, ushort, ushort, short, op>, device::fmx<char2, ushort2, ushort2, short2, op>, device::fmx<char3, ushort3, ushort3, short3, op>, device::fmx<char4, ushort4, ushort4, short4, op>  },
                    { device::fmx<schar, ushort, ushort, int, op>, device::fmx<char2, ushort2, ushort2, int2, op>, device::fmx<char3, ushort3, ushort3, int3, op>, device::fmx<char4, ushort4, ushort4, int4, op>  },
                    { device::fmx<schar, ushort, ushort, float, op>, device::fmx<char2, ushort2, ushort2, float2, op>, device::fmx<char3, ushort3, ushort3, float3, op>, device::fmx<char4, ushort4, ushort4, float4, op>  },
                    { device::fmx<schar, ushort, ushort, double, op>, device::fmx<char2, ushort2, ushort2, double2, op>, device::fmx<char3, ushort3, ushort3, double3, op>, device::fmx<char4, ushort4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<schar, ushort, short, uchar, op>, device::fmx<char2, ushort2, short2, uchar2, op>, device::fmx<char3, ushort3, short3, uchar3, op>, device::fmx<char4, ushort4, short4, uchar4, op>  },
                    { device::fmx<schar, ushort, short, schar, op>, device::fmx<char2, ushort2, short2, char2, op>, device::fmx<char3, ushort3, short3, char3, op>, device::fmx<char4, ushort4, short4, char4, op>  },
                    { device::fmx<schar, ushort, short, ushort, op>, device::fmx<char2, ushort2, short2, ushort2, op>, device::fmx<char3, ushort3, short3, ushort3, op>, device::fmx<char4, ushort4, short4, ushort4, op>  },
                    { device::fmx<schar, ushort, short, short, op>, device::fmx<char2, ushort2, short2, short2, op>, device::fmx<char3, ushort3, short3, short3, op>, device::fmx<char4, ushort4, short4, short4, op>  },
                    { device::fmx<schar, ushort, short, int, op>, device::fmx<char2, ushort2, short2, int2, op>, device::fmx<char3, ushort3, short3, int3, op>, device::fmx<char4, ushort4, short4, int4, op>  },
                    { device::fmx<schar, ushort, short, float, op>, device::fmx<char2, ushort2, short2, float2, op>, device::fmx<char3, ushort3, short3, float3, op>, device::fmx<char4, ushort4, short4, float4, op>  },
                    { device::fmx<schar, ushort, short, double, op>, device::fmx<char2, ushort2, short2, double2, op>, device::fmx<char3, ushort3, short3, double3, op>, device::fmx<char4, ushort4, short4, double4, op>  },
                },
                {
                    { device::fmx<schar, ushort, int, uchar, op>, device::fmx<char2, ushort2, int2, uchar2, op>, device::fmx<char3, ushort3, int3, uchar3, op>, device::fmx<char4, ushort4, int4, uchar4, op>  },
                    { device::fmx<schar, ushort, int, schar, op>, device::fmx<char2, ushort2, int2, char2, op>, device::fmx<char3, ushort3, int3, char3, op>, device::fmx<char4, ushort4, int4, char4, op>  },
                    { device::fmx<schar, ushort, int, ushort, op>, device::fmx<char2, ushort2, int2, ushort2, op>, device::fmx<char3, ushort3, int3, ushort3, op>, device::fmx<char4, ushort4, int4, ushort4, op>  },
                    { device::fmx<schar, ushort, int, short, op>, device::fmx<char2, ushort2, int2, short2, op>, device::fmx<char3, ushort3, int3, short3, op>, device::fmx<char4, ushort4, int4, short4, op>  },
                    { device::fmx<schar, ushort, int, int, op>, device::fmx<char2, ushort2, int2, int2, op>, device::fmx<char3, ushort3, int3, int3, op>, device::fmx<char4, ushort4, int4, int4, op>  },
                    { device::fmx<schar, ushort, int, float, op>, device::fmx<char2, ushort2, int2, float2, op>, device::fmx<char3, ushort3, int3, float3, op>, device::fmx<char4, ushort4, int4, float4, op>  },
                    { device::fmx<schar, ushort, int, double, op>, device::fmx<char2, ushort2, int2, double2, op>, device::fmx<char3, ushort3, int3, double3, op>, device::fmx<char4, ushort4, int4, double4, op>  },
                },
                {
                    { device::fmx<schar, ushort, float, uchar, op>, device::fmx<char2, ushort2, float2, uchar2, op>, device::fmx<char3, ushort3, float3, uchar3, op>, device::fmx<char4, ushort4, float4, uchar4, op>  },
                    { device::fmx<schar, ushort, float, schar, op>, device::fmx<char2, ushort2, float2, char2, op>, device::fmx<char3, ushort3, float3, char3, op>, device::fmx<char4, ushort4, float4, char4, op>  },
                    { device::fmx<schar, ushort, float, ushort, op>, device::fmx<char2, ushort2, float2, ushort2, op>, device::fmx<char3, ushort3, float3, ushort3, op>, device::fmx<char4, ushort4, float4, ushort4, op>  },
                    { device::fmx<schar, ushort, float, short, op>, device::fmx<char2, ushort2, float2, short2, op>, device::fmx<char3, ushort3, float3, short3, op>, device::fmx<char4, ushort4, float4, short4, op>  },
                    { device::fmx<schar, ushort, float, int, op>, device::fmx<char2, ushort2, float2, int2, op>, device::fmx<char3, ushort3, float3, int3, op>, device::fmx<char4, ushort4, float4, int4, op>  },
                    { device::fmx<schar, ushort, float, float, op>, device::fmx<char2, ushort2, float2, float2, op>, device::fmx<char3, ushort3, float3, float3, op>, device::fmx<char4, ushort4, float4, float4, op>  },
                    { device::fmx<schar, ushort, float, double, op>, device::fmx<char2, ushort2, float2, double2, op>, device::fmx<char3, ushort3, float3, double3, op>, device::fmx<char4, ushort4, float4, double4, op>  },
                },
                {
                    { device::fmx<schar, ushort, double, uchar, op>, device::fmx<char2, ushort2, double2, uchar2, op>, device::fmx<char3, ushort3, double3, uchar3, op>, device::fmx<char4, ushort4, double4, uchar4, op>  },
                    { device::fmx<schar, ushort, double, schar, op>, device::fmx<char2, ushort2, double2, char2, op>, device::fmx<char3, ushort3, double3, char3, op>, device::fmx<char4, ushort4, double4, char4, op>  },
                    { device::fmx<schar, ushort, double, ushort, op>, device::fmx<char2, ushort2, double2, ushort2, op>, device::fmx<char3, ushort3, double3, ushort3, op>, device::fmx<char4, ushort4, double4, ushort4, op>  },
                    { device::fmx<schar, ushort, double, short, op>, device::fmx<char2, ushort2, double2, short2, op>, device::fmx<char3, ushort3, double3, short3, op>, device::fmx<char4, ushort4, double4, short4, op>  },
                    { device::fmx<schar, ushort, double, int, op>, device::fmx<char2, ushort2, double2, int2, op>, device::fmx<char3, ushort3, double3, int3, op>, device::fmx<char4, ushort4, double4, int4, op>  },
                    { device::fmx<schar, ushort, double, float, op>, device::fmx<char2, ushort2, double2, float2, op>, device::fmx<char3, ushort3, double3, float3, op>, device::fmx<char4, ushort4, double4, float4, op>  },
                    { device::fmx<schar, ushort, double, double, op>, device::fmx<char2, ushort2, double2, double2, op>, device::fmx<char3, ushort3, double3, double3, op>, device::fmx<char4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<schar, short, uchar, uchar, op>, device::fmx<char2, short2, uchar2, uchar2, op>, device::fmx<char3, short3, uchar3, uchar3, op>, device::fmx<char4, short4, uchar4, uchar4, op>  },
                    { device::fmx<schar, short, uchar, schar, op>, device::fmx<char2, short2, uchar2, char2, op>, device::fmx<char3, short3, uchar3, char3, op>, device::fmx<char4, short4, uchar4, char4, op>  },
                    { device::fmx<schar, short, uchar, ushort, op>, device::fmx<char2, short2, uchar2, ushort2, op>, device::fmx<char3, short3, uchar3, ushort3, op>, device::fmx<char4, short4, uchar4, ushort4, op>  },
                    { device::fmx<schar, short, uchar, short, op>, device::fmx<char2, short2, uchar2, short2, op>, device::fmx<char3, short3, uchar3, short3, op>, device::fmx<char4, short4, uchar4, short4, op>  },
                    { device::fmx<schar, short, uchar, int, op>, device::fmx<char2, short2, uchar2, int2, op>, device::fmx<char3, short3, uchar3, int3, op>, device::fmx<char4, short4, uchar4, int4, op>  },
                    { device::fmx<schar, short, uchar, float, op>, device::fmx<char2, short2, uchar2, float2, op>, device::fmx<char3, short3, uchar3, float3, op>, device::fmx<char4, short4, uchar4, float4, op>  },
                    { device::fmx<schar, short, uchar, double, op>, device::fmx<char2, short2, uchar2, double2, op>, device::fmx<char3, short3, uchar3, double3, op>, device::fmx<char4, short4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<schar, short, schar, uchar, op>, device::fmx<char2, short2, char2, uchar2, op>, device::fmx<char3, short3, char3, uchar3, op>, device::fmx<char4, short4, char4, uchar4, op>  },
                    { device::fmx<schar, short, schar, schar, op>, device::fmx<char2, short2, char2, char2, op>, device::fmx<char3, short3, char3, char3, op>, device::fmx<char4, short4, char4, char4, op>  },
                    { device::fmx<schar, short, schar, ushort, op>, device::fmx<char2, short2, char2, ushort2, op>, device::fmx<char3, short3, char3, ushort3, op>, device::fmx<char4, short4, char4, ushort4, op>  },
                    { device::fmx<schar, short, schar, short, op>, device::fmx<char2, short2, char2, short2, op>, device::fmx<char3, short3, char3, short3, op>, device::fmx<char4, short4, char4, short4, op>  },
                    { device::fmx<schar, short, schar, int, op>, device::fmx<char2, short2, char2, int2, op>, device::fmx<char3, short3, char3, int3, op>, device::fmx<char4, short4, char4, int4, op>  },
                    { device::fmx<schar, short, schar, float, op>, device::fmx<char2, short2, char2, float2, op>, device::fmx<char3, short3, char3, float3, op>, device::fmx<char4, short4, char4, float4, op>  },
                    { device::fmx<schar, short, schar, double, op>, device::fmx<char2, short2, char2, double2, op>, device::fmx<char3, short3, char3, double3, op>, device::fmx<char4, short4, char4, double4, op>  },
                },
                {
                    { device::fmx<schar, short, ushort, uchar, op>, device::fmx<char2, short2, ushort2, uchar2, op>, device::fmx<char3, short3, ushort3, uchar3, op>, device::fmx<char4, short4, ushort4, uchar4, op>  },
                    { device::fmx<schar, short, ushort, schar, op>, device::fmx<char2, short2, ushort2, char2, op>, device::fmx<char3, short3, ushort3, char3, op>, device::fmx<char4, short4, ushort4, char4, op>  },
                    { device::fmx<schar, short, ushort, ushort, op>, device::fmx<char2, short2, ushort2, ushort2, op>, device::fmx<char3, short3, ushort3, ushort3, op>, device::fmx<char4, short4, ushort4, ushort4, op>  },
                    { device::fmx<schar, short, ushort, short, op>, device::fmx<char2, short2, ushort2, short2, op>, device::fmx<char3, short3, ushort3, short3, op>, device::fmx<char4, short4, ushort4, short4, op>  },
                    { device::fmx<schar, short, ushort, int, op>, device::fmx<char2, short2, ushort2, int2, op>, device::fmx<char3, short3, ushort3, int3, op>, device::fmx<char4, short4, ushort4, int4, op>  },
                    { device::fmx<schar, short, ushort, float, op>, device::fmx<char2, short2, ushort2, float2, op>, device::fmx<char3, short3, ushort3, float3, op>, device::fmx<char4, short4, ushort4, float4, op>  },
                    { device::fmx<schar, short, ushort, double, op>, device::fmx<char2, short2, ushort2, double2, op>, device::fmx<char3, short3, ushort3, double3, op>, device::fmx<char4, short4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<schar, short, short, uchar, op>, device::fmx<char2, short2, short2, uchar2, op>, device::fmx<char3, short3, short3, uchar3, op>, device::fmx<char4, short4, short4, uchar4, op>  },
                    { device::fmx<schar, short, short, schar, op>, device::fmx<char2, short2, short2, char2, op>, device::fmx<char3, short3, short3, char3, op>, device::fmx<char4, short4, short4, char4, op>  },
                    { device::fmx<schar, short, short, ushort, op>, device::fmx<char2, short2, short2, ushort2, op>, device::fmx<char3, short3, short3, ushort3, op>, device::fmx<char4, short4, short4, ushort4, op>  },
                    { device::fmx<schar, short, short, short, op>, device::fmx<char2, short2, short2, short2, op>, device::fmx<char3, short3, short3, short3, op>, device::fmx<char4, short4, short4, short4, op>  },
                    { device::fmx<schar, short, short, int, op>, device::fmx<char2, short2, short2, int2, op>, device::fmx<char3, short3, short3, int3, op>, device::fmx<char4, short4, short4, int4, op>  },
                    { device::fmx<schar, short, short, float, op>, device::fmx<char2, short2, short2, float2, op>, device::fmx<char3, short3, short3, float3, op>, device::fmx<char4, short4, short4, float4, op>  },
                    { device::fmx<schar, short, short, double, op>, device::fmx<char2, short2, short2, double2, op>, device::fmx<char3, short3, short3, double3, op>, device::fmx<char4, short4, short4, double4, op>  },
                },
                {
                    { device::fmx<schar, short, int, uchar, op>, device::fmx<char2, short2, int2, uchar2, op>, device::fmx<char3, short3, int3, uchar3, op>, device::fmx<char4, short4, int4, uchar4, op>  },
                    { device::fmx<schar, short, int, schar, op>, device::fmx<char2, short2, int2, char2, op>, device::fmx<char3, short3, int3, char3, op>, device::fmx<char4, short4, int4, char4, op>  },
                    { device::fmx<schar, short, int, ushort, op>, device::fmx<char2, short2, int2, ushort2, op>, device::fmx<char3, short3, int3, ushort3, op>, device::fmx<char4, short4, int4, ushort4, op>  },
                    { device::fmx<schar, short, int, short, op>, device::fmx<char2, short2, int2, short2, op>, device::fmx<char3, short3, int3, short3, op>, device::fmx<char4, short4, int4, short4, op>  },
                    { device::fmx<schar, short, int, int, op>, device::fmx<char2, short2, int2, int2, op>, device::fmx<char3, short3, int3, int3, op>, device::fmx<char4, short4, int4, int4, op>  },
                    { device::fmx<schar, short, int, float, op>, device::fmx<char2, short2, int2, float2, op>, device::fmx<char3, short3, int3, float3, op>, device::fmx<char4, short4, int4, float4, op>  },
                    { device::fmx<schar, short, int, double, op>, device::fmx<char2, short2, int2, double2, op>, device::fmx<char3, short3, int3, double3, op>, device::fmx<char4, short4, int4, double4, op>  },
                },
                {
                    { device::fmx<schar, short, float, uchar, op>, device::fmx<char2, short2, float2, uchar2, op>, device::fmx<char3, short3, float3, uchar3, op>, device::fmx<char4, short4, float4, uchar4, op>  },
                    { device::fmx<schar, short, float, schar, op>, device::fmx<char2, short2, float2, char2, op>, device::fmx<char3, short3, float3, char3, op>, device::fmx<char4, short4, float4, char4, op>  },
                    { device::fmx<schar, short, float, ushort, op>, device::fmx<char2, short2, float2, ushort2, op>, device::fmx<char3, short3, float3, ushort3, op>, device::fmx<char4, short4, float4, ushort4, op>  },
                    { device::fmx<schar, short, float, short, op>, device::fmx<char2, short2, float2, short2, op>, device::fmx<char3, short3, float3, short3, op>, device::fmx<char4, short4, float4, short4, op>  },
                    { device::fmx<schar, short, float, int, op>, device::fmx<char2, short2, float2, int2, op>, device::fmx<char3, short3, float3, int3, op>, device::fmx<char4, short4, float4, int4, op>  },
                    { device::fmx<schar, short, float, float, op>, device::fmx<char2, short2, float2, float2, op>, device::fmx<char3, short3, float3, float3, op>, device::fmx<char4, short4, float4, float4, op>  },
                    { device::fmx<schar, short, float, double, op>, device::fmx<char2, short2, float2, double2, op>, device::fmx<char3, short3, float3, double3, op>, device::fmx<char4, short4, float4, double4, op>  },
                },
                {
                    { device::fmx<schar, short, double, uchar, op>, device::fmx<char2, short2, double2, uchar2, op>, device::fmx<char3, short3, double3, uchar3, op>, device::fmx<char4, short4, double4, uchar4, op>  },
                    { device::fmx<schar, short, double, schar, op>, device::fmx<char2, short2, double2, char2, op>, device::fmx<char3, short3, double3, char3, op>, device::fmx<char4, short4, double4, char4, op>  },
                    { device::fmx<schar, short, double, ushort, op>, device::fmx<char2, short2, double2, ushort2, op>, device::fmx<char3, short3, double3, ushort3, op>, device::fmx<char4, short4, double4, ushort4, op>  },
                    { device::fmx<schar, short, double, short, op>, device::fmx<char2, short2, double2, short2, op>, device::fmx<char3, short3, double3, short3, op>, device::fmx<char4, short4, double4, short4, op>  },
                    { device::fmx<schar, short, double, int, op>, device::fmx<char2, short2, double2, int2, op>, device::fmx<char3, short3, double3, int3, op>, device::fmx<char4, short4, double4, int4, op>  },
                    { device::fmx<schar, short, double, float, op>, device::fmx<char2, short2, double2, float2, op>, device::fmx<char3, short3, double3, float3, op>, device::fmx<char4, short4, double4, float4, op>  },
                    { device::fmx<schar, short, double, double, op>, device::fmx<char2, short2, double2, double2, op>, device::fmx<char3, short3, double3, double3, op>, device::fmx<char4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<schar, int, uchar, uchar, op>, device::fmx<char2, int2, uchar2, uchar2, op>, device::fmx<char3, int3, uchar3, uchar3, op>, device::fmx<char4, int4, uchar4, uchar4, op>  },
                    { device::fmx<schar, int, uchar, schar, op>, device::fmx<char2, int2, uchar2, char2, op>, device::fmx<char3, int3, uchar3, char3, op>, device::fmx<char4, int4, uchar4, char4, op>  },
                    { device::fmx<schar, int, uchar, ushort, op>, device::fmx<char2, int2, uchar2, ushort2, op>, device::fmx<char3, int3, uchar3, ushort3, op>, device::fmx<char4, int4, uchar4, ushort4, op>  },
                    { device::fmx<schar, int, uchar, short, op>, device::fmx<char2, int2, uchar2, short2, op>, device::fmx<char3, int3, uchar3, short3, op>, device::fmx<char4, int4, uchar4, short4, op>  },
                    { device::fmx<schar, int, uchar, int, op>, device::fmx<char2, int2, uchar2, int2, op>, device::fmx<char3, int3, uchar3, int3, op>, device::fmx<char4, int4, uchar4, int4, op>  },
                    { device::fmx<schar, int, uchar, float, op>, device::fmx<char2, int2, uchar2, float2, op>, device::fmx<char3, int3, uchar3, float3, op>, device::fmx<char4, int4, uchar4, float4, op>  },
                    { device::fmx<schar, int, uchar, double, op>, device::fmx<char2, int2, uchar2, double2, op>, device::fmx<char3, int3, uchar3, double3, op>, device::fmx<char4, int4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<schar, int, schar, uchar, op>, device::fmx<char2, int2, char2, uchar2, op>, device::fmx<char3, int3, char3, uchar3, op>, device::fmx<char4, int4, char4, uchar4, op>  },
                    { device::fmx<schar, int, schar, schar, op>, device::fmx<char2, int2, char2, char2, op>, device::fmx<char3, int3, char3, char3, op>, device::fmx<char4, int4, char4, char4, op>  },
                    { device::fmx<schar, int, schar, ushort, op>, device::fmx<char2, int2, char2, ushort2, op>, device::fmx<char3, int3, char3, ushort3, op>, device::fmx<char4, int4, char4, ushort4, op>  },
                    { device::fmx<schar, int, schar, short, op>, device::fmx<char2, int2, char2, short2, op>, device::fmx<char3, int3, char3, short3, op>, device::fmx<char4, int4, char4, short4, op>  },
                    { device::fmx<schar, int, schar, int, op>, device::fmx<char2, int2, char2, int2, op>, device::fmx<char3, int3, char3, int3, op>, device::fmx<char4, int4, char4, int4, op>  },
                    { device::fmx<schar, int, schar, float, op>, device::fmx<char2, int2, char2, float2, op>, device::fmx<char3, int3, char3, float3, op>, device::fmx<char4, int4, char4, float4, op>  },
                    { device::fmx<schar, int, schar, double, op>, device::fmx<char2, int2, char2, double2, op>, device::fmx<char3, int3, char3, double3, op>, device::fmx<char4, int4, char4, double4, op>  },
                },
                {
                    { device::fmx<schar, int, ushort, uchar, op>, device::fmx<char2, int2, ushort2, uchar2, op>, device::fmx<char3, int3, ushort3, uchar3, op>, device::fmx<char4, int4, ushort4, uchar4, op>  },
                    { device::fmx<schar, int, ushort, schar, op>, device::fmx<char2, int2, ushort2, char2, op>, device::fmx<char3, int3, ushort3, char3, op>, device::fmx<char4, int4, ushort4, char4, op>  },
                    { device::fmx<schar, int, ushort, ushort, op>, device::fmx<char2, int2, ushort2, ushort2, op>, device::fmx<char3, int3, ushort3, ushort3, op>, device::fmx<char4, int4, ushort4, ushort4, op>  },
                    { device::fmx<schar, int, ushort, short, op>, device::fmx<char2, int2, ushort2, short2, op>, device::fmx<char3, int3, ushort3, short3, op>, device::fmx<char4, int4, ushort4, short4, op>  },
                    { device::fmx<schar, int, ushort, int, op>, device::fmx<char2, int2, ushort2, int2, op>, device::fmx<char3, int3, ushort3, int3, op>, device::fmx<char4, int4, ushort4, int4, op>  },
                    { device::fmx<schar, int, ushort, float, op>, device::fmx<char2, int2, ushort2, float2, op>, device::fmx<char3, int3, ushort3, float3, op>, device::fmx<char4, int4, ushort4, float4, op>  },
                    { device::fmx<schar, int, ushort, double, op>, device::fmx<char2, int2, ushort2, double2, op>, device::fmx<char3, int3, ushort3, double3, op>, device::fmx<char4, int4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<schar, int, short, uchar, op>, device::fmx<char2, int2, short2, uchar2, op>, device::fmx<char3, int3, short3, uchar3, op>, device::fmx<char4, int4, short4, uchar4, op>  },
                    { device::fmx<schar, int, short, schar, op>, device::fmx<char2, int2, short2, char2, op>, device::fmx<char3, int3, short3, char3, op>, device::fmx<char4, int4, short4, char4, op>  },
                    { device::fmx<schar, int, short, ushort, op>, device::fmx<char2, int2, short2, ushort2, op>, device::fmx<char3, int3, short3, ushort3, op>, device::fmx<char4, int4, short4, ushort4, op>  },
                    { device::fmx<schar, int, short, short, op>, device::fmx<char2, int2, short2, short2, op>, device::fmx<char3, int3, short3, short3, op>, device::fmx<char4, int4, short4, short4, op>  },
                    { device::fmx<schar, int, short, int, op>, device::fmx<char2, int2, short2, int2, op>, device::fmx<char3, int3, short3, int3, op>, device::fmx<char4, int4, short4, int4, op>  },
                    { device::fmx<schar, int, short, float, op>, device::fmx<char2, int2, short2, float2, op>, device::fmx<char3, int3, short3, float3, op>, device::fmx<char4, int4, short4, float4, op>  },
                    { device::fmx<schar, int, short, double, op>, device::fmx<char2, int2, short2, double2, op>, device::fmx<char3, int3, short3, double3, op>, device::fmx<char4, int4, short4, double4, op>  },
                },
                {
                    { device::fmx<schar, int, int, uchar, op>, device::fmx<char2, int2, int2, uchar2, op>, device::fmx<char3, int3, int3, uchar3, op>, device::fmx<char4, int4, int4, uchar4, op>  },
                    { device::fmx<schar, int, int, schar, op>, device::fmx<char2, int2, int2, char2, op>, device::fmx<char3, int3, int3, char3, op>, device::fmx<char4, int4, int4, char4, op>  },
                    { device::fmx<schar, int, int, ushort, op>, device::fmx<char2, int2, int2, ushort2, op>, device::fmx<char3, int3, int3, ushort3, op>, device::fmx<char4, int4, int4, ushort4, op>  },
                    { device::fmx<schar, int, int, short, op>, device::fmx<char2, int2, int2, short2, op>, device::fmx<char3, int3, int3, short3, op>, device::fmx<char4, int4, int4, short4, op>  },
                    { device::fmx<schar, int, int, int, op>, device::fmx<char2, int2, int2, int2, op>, device::fmx<char3, int3, int3, int3, op>, device::fmx<char4, int4, int4, int4, op>  },
                    { device::fmx<schar, int, int, float, op>, device::fmx<char2, int2, int2, float2, op>, device::fmx<char3, int3, int3, float3, op>, device::fmx<char4, int4, int4, float4, op>  },
                    { device::fmx<schar, int, int, double, op>, device::fmx<char2, int2, int2, double2, op>, device::fmx<char3, int3, int3, double3, op>, device::fmx<char4, int4, int4, double4, op>  },
                },
                {
                    { device::fmx<schar, int, float, uchar, op>, device::fmx<char2, int2, float2, uchar2, op>, device::fmx<char3, int3, float3, uchar3, op>, device::fmx<char4, int4, float4, uchar4, op>  },
                    { device::fmx<schar, int, float, schar, op>, device::fmx<char2, int2, float2, char2, op>, device::fmx<char3, int3, float3, char3, op>, device::fmx<char4, int4, float4, char4, op>  },
                    { device::fmx<schar, int, float, ushort, op>, device::fmx<char2, int2, float2, ushort2, op>, device::fmx<char3, int3, float3, ushort3, op>, device::fmx<char4, int4, float4, ushort4, op>  },
                    { device::fmx<schar, int, float, short, op>, device::fmx<char2, int2, float2, short2, op>, device::fmx<char3, int3, float3, short3, op>, device::fmx<char4, int4, float4, short4, op>  },
                    { device::fmx<schar, int, float, int, op>, device::fmx<char2, int2, float2, int2, op>, device::fmx<char3, int3, float3, int3, op>, device::fmx<char4, int4, float4, int4, op>  },
                    { device::fmx<schar, int, float, float, op>, device::fmx<char2, int2, float2, float2, op>, device::fmx<char3, int3, float3, float3, op>, device::fmx<char4, int4, float4, float4, op>  },
                    { device::fmx<schar, int, float, double, op>, device::fmx<char2, int2, float2, double2, op>, device::fmx<char3, int3, float3, double3, op>, device::fmx<char4, int4, float4, double4, op>  },
                },
                {
                    { device::fmx<schar, int, double, uchar, op>, device::fmx<char2, int2, double2, uchar2, op>, device::fmx<char3, int3, double3, uchar3, op>, device::fmx<char4, int4, double4, uchar4, op>  },
                    { device::fmx<schar, int, double, schar, op>, device::fmx<char2, int2, double2, char2, op>, device::fmx<char3, int3, double3, char3, op>, device::fmx<char4, int4, double4, char4, op>  },
                    { device::fmx<schar, int, double, ushort, op>, device::fmx<char2, int2, double2, ushort2, op>, device::fmx<char3, int3, double3, ushort3, op>, device::fmx<char4, int4, double4, ushort4, op>  },
                    { device::fmx<schar, int, double, short, op>, device::fmx<char2, int2, double2, short2, op>, device::fmx<char3, int3, double3, short3, op>, device::fmx<char4, int4, double4, short4, op>  },
                    { device::fmx<schar, int, double, int, op>, device::fmx<char2, int2, double2, int2, op>, device::fmx<char3, int3, double3, int3, op>, device::fmx<char4, int4, double4, int4, op>  },
                    { device::fmx<schar, int, double, float, op>, device::fmx<char2, int2, double2, float2, op>, device::fmx<char3, int3, double3, float3, op>, device::fmx<char4, int4, double4, float4, op>  },
                    { device::fmx<schar, int, double, double, op>, device::fmx<char2, int2, double2, double2, op>, device::fmx<char3, int3, double3, double3, op>, device::fmx<char4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<schar, float, uchar, uchar, op>, device::fmx<char2, float2, uchar2, uchar2, op>, device::fmx<char3, float3, uchar3, uchar3, op>, device::fmx<char4, float4, uchar4, uchar4, op>  },
                    { device::fmx<schar, float, uchar, schar, op>, device::fmx<char2, float2, uchar2, char2, op>, device::fmx<char3, float3, uchar3, char3, op>, device::fmx<char4, float4, uchar4, char4, op>  },
                    { device::fmx<schar, float, uchar, ushort, op>, device::fmx<char2, float2, uchar2, ushort2, op>, device::fmx<char3, float3, uchar3, ushort3, op>, device::fmx<char4, float4, uchar4, ushort4, op>  },
                    { device::fmx<schar, float, uchar, short, op>, device::fmx<char2, float2, uchar2, short2, op>, device::fmx<char3, float3, uchar3, short3, op>, device::fmx<char4, float4, uchar4, short4, op>  },
                    { device::fmx<schar, float, uchar, int, op>, device::fmx<char2, float2, uchar2, int2, op>, device::fmx<char3, float3, uchar3, int3, op>, device::fmx<char4, float4, uchar4, int4, op>  },
                    { device::fmx<schar, float, uchar, float, op>, device::fmx<char2, float2, uchar2, float2, op>, device::fmx<char3, float3, uchar3, float3, op>, device::fmx<char4, float4, uchar4, float4, op>  },
                    { device::fmx<schar, float, uchar, double, op>, device::fmx<char2, float2, uchar2, double2, op>, device::fmx<char3, float3, uchar3, double3, op>, device::fmx<char4, float4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<schar, float, schar, uchar, op>, device::fmx<char2, float2, char2, uchar2, op>, device::fmx<char3, float3, char3, uchar3, op>, device::fmx<char4, float4, char4, uchar4, op>  },
                    { device::fmx<schar, float, schar, schar, op>, device::fmx<char2, float2, char2, char2, op>, device::fmx<char3, float3, char3, char3, op>, device::fmx<char4, float4, char4, char4, op>  },
                    { device::fmx<schar, float, schar, ushort, op>, device::fmx<char2, float2, char2, ushort2, op>, device::fmx<char3, float3, char3, ushort3, op>, device::fmx<char4, float4, char4, ushort4, op>  },
                    { device::fmx<schar, float, schar, short, op>, device::fmx<char2, float2, char2, short2, op>, device::fmx<char3, float3, char3, short3, op>, device::fmx<char4, float4, char4, short4, op>  },
                    { device::fmx<schar, float, schar, int, op>, device::fmx<char2, float2, char2, int2, op>, device::fmx<char3, float3, char3, int3, op>, device::fmx<char4, float4, char4, int4, op>  },
                    { device::fmx<schar, float, schar, float, op>, device::fmx<char2, float2, char2, float2, op>, device::fmx<char3, float3, char3, float3, op>, device::fmx<char4, float4, char4, float4, op>  },
                    { device::fmx<schar, float, schar, double, op>, device::fmx<char2, float2, char2, double2, op>, device::fmx<char3, float3, char3, double3, op>, device::fmx<char4, float4, char4, double4, op>  },
                },
                {
                    { device::fmx<schar, float, ushort, uchar, op>, device::fmx<char2, float2, ushort2, uchar2, op>, device::fmx<char3, float3, ushort3, uchar3, op>, device::fmx<char4, float4, ushort4, uchar4, op>  },
                    { device::fmx<schar, float, ushort, schar, op>, device::fmx<char2, float2, ushort2, char2, op>, device::fmx<char3, float3, ushort3, char3, op>, device::fmx<char4, float4, ushort4, char4, op>  },
                    { device::fmx<schar, float, ushort, ushort, op>, device::fmx<char2, float2, ushort2, ushort2, op>, device::fmx<char3, float3, ushort3, ushort3, op>, device::fmx<char4, float4, ushort4, ushort4, op>  },
                    { device::fmx<schar, float, ushort, short, op>, device::fmx<char2, float2, ushort2, short2, op>, device::fmx<char3, float3, ushort3, short3, op>, device::fmx<char4, float4, ushort4, short4, op>  },
                    { device::fmx<schar, float, ushort, int, op>, device::fmx<char2, float2, ushort2, int2, op>, device::fmx<char3, float3, ushort3, int3, op>, device::fmx<char4, float4, ushort4, int4, op>  },
                    { device::fmx<schar, float, ushort, float, op>, device::fmx<char2, float2, ushort2, float2, op>, device::fmx<char3, float3, ushort3, float3, op>, device::fmx<char4, float4, ushort4, float4, op>  },
                    { device::fmx<schar, float, ushort, double, op>, device::fmx<char2, float2, ushort2, double2, op>, device::fmx<char3, float3, ushort3, double3, op>, device::fmx<char4, float4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<schar, float, short, uchar, op>, device::fmx<char2, float2, short2, uchar2, op>, device::fmx<char3, float3, short3, uchar3, op>, device::fmx<char4, float4, short4, uchar4, op>  },
                    { device::fmx<schar, float, short, schar, op>, device::fmx<char2, float2, short2, char2, op>, device::fmx<char3, float3, short3, char3, op>, device::fmx<char4, float4, short4, char4, op>  },
                    { device::fmx<schar, float, short, ushort, op>, device::fmx<char2, float2, short2, ushort2, op>, device::fmx<char3, float3, short3, ushort3, op>, device::fmx<char4, float4, short4, ushort4, op>  },
                    { device::fmx<schar, float, short, short, op>, device::fmx<char2, float2, short2, short2, op>, device::fmx<char3, float3, short3, short3, op>, device::fmx<char4, float4, short4, short4, op>  },
                    { device::fmx<schar, float, short, int, op>, device::fmx<char2, float2, short2, int2, op>, device::fmx<char3, float3, short3, int3, op>, device::fmx<char4, float4, short4, int4, op>  },
                    { device::fmx<schar, float, short, float, op>, device::fmx<char2, float2, short2, float2, op>, device::fmx<char3, float3, short3, float3, op>, device::fmx<char4, float4, short4, float4, op>  },
                    { device::fmx<schar, float, short, double, op>, device::fmx<char2, float2, short2, double2, op>, device::fmx<char3, float3, short3, double3, op>, device::fmx<char4, float4, short4, double4, op>  },
                },
                {
                    { device::fmx<schar, float, int, uchar, op>, device::fmx<char2, float2, int2, uchar2, op>, device::fmx<char3, float3, int3, uchar3, op>, device::fmx<char4, float4, int4, uchar4, op>  },
                    { device::fmx<schar, float, int, schar, op>, device::fmx<char2, float2, int2, char2, op>, device::fmx<char3, float3, int3, char3, op>, device::fmx<char4, float4, int4, char4, op>  },
                    { device::fmx<schar, float, int, ushort, op>, device::fmx<char2, float2, int2, ushort2, op>, device::fmx<char3, float3, int3, ushort3, op>, device::fmx<char4, float4, int4, ushort4, op>  },
                    { device::fmx<schar, float, int, short, op>, device::fmx<char2, float2, int2, short2, op>, device::fmx<char3, float3, int3, short3, op>, device::fmx<char4, float4, int4, short4, op>  },
                    { device::fmx<schar, float, int, int, op>, device::fmx<char2, float2, int2, int2, op>, device::fmx<char3, float3, int3, int3, op>, device::fmx<char4, float4, int4, int4, op>  },
                    { device::fmx<schar, float, int, float, op>, device::fmx<char2, float2, int2, float2, op>, device::fmx<char3, float3, int3, float3, op>, device::fmx<char4, float4, int4, float4, op>  },
                    { device::fmx<schar, float, int, double, op>, device::fmx<char2, float2, int2, double2, op>, device::fmx<char3, float3, int3, double3, op>, device::fmx<char4, float4, int4, double4, op>  },
                },
                {
                    { device::fmx<schar, float, float, uchar, op>, device::fmx<char2, float2, float2, uchar2, op>, device::fmx<char3, float3, float3, uchar3, op>, device::fmx<char4, float4, float4, uchar4, op>  },
                    { device::fmx<schar, float, float, schar, op>, device::fmx<char2, float2, float2, char2, op>, device::fmx<char3, float3, float3, char3, op>, device::fmx<char4, float4, float4, char4, op>  },
                    { device::fmx<schar, float, float, ushort, op>, device::fmx<char2, float2, float2, ushort2, op>, device::fmx<char3, float3, float3, ushort3, op>, device::fmx<char4, float4, float4, ushort4, op>  },
                    { device::fmx<schar, float, float, short, op>, device::fmx<char2, float2, float2, short2, op>, device::fmx<char3, float3, float3, short3, op>, device::fmx<char4, float4, float4, short4, op>  },
                    { device::fmx<schar, float, float, int, op>, device::fmx<char2, float2, float2, int2, op>, device::fmx<char3, float3, float3, int3, op>, device::fmx<char4, float4, float4, int4, op>  },
                    { device::fmx<schar, float, float, float, op>, device::fmx<char2, float2, float2, float2, op>, device::fmx<char3, float3, float3, float3, op>, device::fmx<char4, float4, float4, float4, op>  },
                    { device::fmx<schar, float, float, double, op>, device::fmx<char2, float2, float2, double2, op>, device::fmx<char3, float3, float3, double3, op>, device::fmx<char4, float4, float4, double4, op>  },
                },
                {
                    { device::fmx<schar, float, double, uchar, op>, device::fmx<char2, float2, double2, uchar2, op>, device::fmx<char3, float3, double3, uchar3, op>, device::fmx<char4, float4, double4, uchar4, op>  },
                    { device::fmx<schar, float, double, schar, op>, device::fmx<char2, float2, double2, char2, op>, device::fmx<char3, float3, double3, char3, op>, device::fmx<char4, float4, double4, char4, op>  },
                    { device::fmx<schar, float, double, ushort, op>, device::fmx<char2, float2, double2, ushort2, op>, device::fmx<char3, float3, double3, ushort3, op>, device::fmx<char4, float4, double4, ushort4, op>  },
                    { device::fmx<schar, float, double, short, op>, device::fmx<char2, float2, double2, short2, op>, device::fmx<char3, float3, double3, short3, op>, device::fmx<char4, float4, double4, short4, op>  },
                    { device::fmx<schar, float, double, int, op>, device::fmx<char2, float2, double2, int2, op>, device::fmx<char3, float3, double3, int3, op>, device::fmx<char4, float4, double4, int4, op>  },
                    { device::fmx<schar, float, double, float, op>, device::fmx<char2, float2, double2, float2, op>, device::fmx<char3, float3, double3, float3, op>, device::fmx<char4, float4, double4, float4, op>  },
                    { device::fmx<schar, float, double, double, op>, device::fmx<char2, float2, double2, double2, op>, device::fmx<char3, float3, double3, double3, op>, device::fmx<char4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<schar, double, uchar, uchar, op>, device::fmx<char2, double2, uchar2, uchar2, op>, device::fmx<char3, double3, uchar3, uchar3, op>, device::fmx<char4, double4, uchar4, uchar4, op>  },
                    { device::fmx<schar, double, uchar, schar, op>, device::fmx<char2, double2, uchar2, char2, op>, device::fmx<char3, double3, uchar3, char3, op>, device::fmx<char4, double4, uchar4, char4, op>  },
                    { device::fmx<schar, double, uchar, ushort, op>, device::fmx<char2, double2, uchar2, ushort2, op>, device::fmx<char3, double3, uchar3, ushort3, op>, device::fmx<char4, double4, uchar4, ushort4, op>  },
                    { device::fmx<schar, double, uchar, short, op>, device::fmx<char2, double2, uchar2, short2, op>, device::fmx<char3, double3, uchar3, short3, op>, device::fmx<char4, double4, uchar4, short4, op>  },
                    { device::fmx<schar, double, uchar, int, op>, device::fmx<char2, double2, uchar2, int2, op>, device::fmx<char3, double3, uchar3, int3, op>, device::fmx<char4, double4, uchar4, int4, op>  },
                    { device::fmx<schar, double, uchar, float, op>, device::fmx<char2, double2, uchar2, float2, op>, device::fmx<char3, double3, uchar3, float3, op>, device::fmx<char4, double4, uchar4, float4, op>  },
                    { device::fmx<schar, double, uchar, double, op>, device::fmx<char2, double2, uchar2, double2, op>, device::fmx<char3, double3, uchar3, double3, op>, device::fmx<char4, double4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<schar, double, schar, uchar, op>, device::fmx<char2, double2, char2, uchar2, op>, device::fmx<char3, double3, char3, uchar3, op>, device::fmx<char4, double4, char4, uchar4, op>  },
                    { device::fmx<schar, double, schar, schar, op>, device::fmx<char2, double2, char2, char2, op>, device::fmx<char3, double3, char3, char3, op>, device::fmx<char4, double4, char4, char4, op>  },
                    { device::fmx<schar, double, schar, ushort, op>, device::fmx<char2, double2, char2, ushort2, op>, device::fmx<char3, double3, char3, ushort3, op>, device::fmx<char4, double4, char4, ushort4, op>  },
                    { device::fmx<schar, double, schar, short, op>, device::fmx<char2, double2, char2, short2, op>, device::fmx<char3, double3, char3, short3, op>, device::fmx<char4, double4, char4, short4, op>  },
                    { device::fmx<schar, double, schar, int, op>, device::fmx<char2, double2, char2, int2, op>, device::fmx<char3, double3, char3, int3, op>, device::fmx<char4, double4, char4, int4, op>  },
                    { device::fmx<schar, double, schar, float, op>, device::fmx<char2, double2, char2, float2, op>, device::fmx<char3, double3, char3, float3, op>, device::fmx<char4, double4, char4, float4, op>  },
                    { device::fmx<schar, double, schar, double, op>, device::fmx<char2, double2, char2, double2, op>, device::fmx<char3, double3, char3, double3, op>, device::fmx<char4, double4, char4, double4, op>  },
                },
                {
                    { device::fmx<schar, double, ushort, uchar, op>, device::fmx<char2, double2, ushort2, uchar2, op>, device::fmx<char3, double3, ushort3, uchar3, op>, device::fmx<char4, double4, ushort4, uchar4, op>  },
                    { device::fmx<schar, double, ushort, schar, op>, device::fmx<char2, double2, ushort2, char2, op>, device::fmx<char3, double3, ushort3, char3, op>, device::fmx<char4, double4, ushort4, char4, op>  },
                    { device::fmx<schar, double, ushort, ushort, op>, device::fmx<char2, double2, ushort2, ushort2, op>, device::fmx<char3, double3, ushort3, ushort3, op>, device::fmx<char4, double4, ushort4, ushort4, op>  },
                    { device::fmx<schar, double, ushort, short, op>, device::fmx<char2, double2, ushort2, short2, op>, device::fmx<char3, double3, ushort3, short3, op>, device::fmx<char4, double4, ushort4, short4, op>  },
                    { device::fmx<schar, double, ushort, int, op>, device::fmx<char2, double2, ushort2, int2, op>, device::fmx<char3, double3, ushort3, int3, op>, device::fmx<char4, double4, ushort4, int4, op>  },
                    { device::fmx<schar, double, ushort, float, op>, device::fmx<char2, double2, ushort2, float2, op>, device::fmx<char3, double3, ushort3, float3, op>, device::fmx<char4, double4, ushort4, float4, op>  },
                    { device::fmx<schar, double, ushort, double, op>, device::fmx<char2, double2, ushort2, double2, op>, device::fmx<char3, double3, ushort3, double3, op>, device::fmx<char4, double4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<schar, double, short, uchar, op>, device::fmx<char2, double2, short2, uchar2, op>, device::fmx<char3, double3, short3, uchar3, op>, device::fmx<char4, double4, short4, uchar4, op>  },
                    { device::fmx<schar, double, short, schar, op>, device::fmx<char2, double2, short2, char2, op>, device::fmx<char3, double3, short3, char3, op>, device::fmx<char4, double4, short4, char4, op>  },
                    { device::fmx<schar, double, short, ushort, op>, device::fmx<char2, double2, short2, ushort2, op>, device::fmx<char3, double3, short3, ushort3, op>, device::fmx<char4, double4, short4, ushort4, op>  },
                    { device::fmx<schar, double, short, short, op>, device::fmx<char2, double2, short2, short2, op>, device::fmx<char3, double3, short3, short3, op>, device::fmx<char4, double4, short4, short4, op>  },
                    { device::fmx<schar, double, short, int, op>, device::fmx<char2, double2, short2, int2, op>, device::fmx<char3, double3, short3, int3, op>, device::fmx<char4, double4, short4, int4, op>  },
                    { device::fmx<schar, double, short, float, op>, device::fmx<char2, double2, short2, float2, op>, device::fmx<char3, double3, short3, float3, op>, device::fmx<char4, double4, short4, float4, op>  },
                    { device::fmx<schar, double, short, double, op>, device::fmx<char2, double2, short2, double2, op>, device::fmx<char3, double3, short3, double3, op>, device::fmx<char4, double4, short4, double4, op>  },
                },
                {
                    { device::fmx<schar, double, int, uchar, op>, device::fmx<char2, double2, int2, uchar2, op>, device::fmx<char3, double3, int3, uchar3, op>, device::fmx<char4, double4, int4, uchar4, op>  },
                    { device::fmx<schar, double, int, schar, op>, device::fmx<char2, double2, int2, char2, op>, device::fmx<char3, double3, int3, char3, op>, device::fmx<char4, double4, int4, char4, op>  },
                    { device::fmx<schar, double, int, ushort, op>, device::fmx<char2, double2, int2, ushort2, op>, device::fmx<char3, double3, int3, ushort3, op>, device::fmx<char4, double4, int4, ushort4, op>  },
                    { device::fmx<schar, double, int, short, op>, device::fmx<char2, double2, int2, short2, op>, device::fmx<char3, double3, int3, short3, op>, device::fmx<char4, double4, int4, short4, op>  },
                    { device::fmx<schar, double, int, int, op>, device::fmx<char2, double2, int2, int2, op>, device::fmx<char3, double3, int3, int3, op>, device::fmx<char4, double4, int4, int4, op>  },
                    { device::fmx<schar, double, int, float, op>, device::fmx<char2, double2, int2, float2, op>, device::fmx<char3, double3, int3, float3, op>, device::fmx<char4, double4, int4, float4, op>  },
                    { device::fmx<schar, double, int, double, op>, device::fmx<char2, double2, int2, double2, op>, device::fmx<char3, double3, int3, double3, op>, device::fmx<char4, double4, int4, double4, op>  },
                },
                {
                    { device::fmx<schar, double, float, uchar, op>, device::fmx<char2, double2, float2, uchar2, op>, device::fmx<char3, double3, float3, uchar3, op>, device::fmx<char4, double4, float4, uchar4, op>  },
                    { device::fmx<schar, double, float, schar, op>, device::fmx<char2, double2, float2, char2, op>, device::fmx<char3, double3, float3, char3, op>, device::fmx<char4, double4, float4, char4, op>  },
                    { device::fmx<schar, double, float, ushort, op>, device::fmx<char2, double2, float2, ushort2, op>, device::fmx<char3, double3, float3, ushort3, op>, device::fmx<char4, double4, float4, ushort4, op>  },
                    { device::fmx<schar, double, float, short, op>, device::fmx<char2, double2, float2, short2, op>, device::fmx<char3, double3, float3, short3, op>, device::fmx<char4, double4, float4, short4, op>  },
                    { device::fmx<schar, double, float, int, op>, device::fmx<char2, double2, float2, int2, op>, device::fmx<char3, double3, float3, int3, op>, device::fmx<char4, double4, float4, int4, op>  },
                    { device::fmx<schar, double, float, float, op>, device::fmx<char2, double2, float2, float2, op>, device::fmx<char3, double3, float3, float3, op>, device::fmx<char4, double4, float4, float4, op>  },
                    { device::fmx<schar, double, float, double, op>, device::fmx<char2, double2, float2, double2, op>, device::fmx<char3, double3, float3, double3, op>, device::fmx<char4, double4, float4, double4, op>  },
                },
                {
                    { device::fmx<schar, double, double, uchar, op>, device::fmx<char2, double2, double2, uchar2, op>, device::fmx<char3, double3, double3, uchar3, op>, device::fmx<char4, double4, double4, uchar4, op>  },
                    { device::fmx<schar, double, double, schar, op>, device::fmx<char2, double2, double2, char2, op>, device::fmx<char3, double3, double3, char3, op>, device::fmx<char4, double4, double4, char4, op>  },
                    { device::fmx<schar, double, double, ushort, op>, device::fmx<char2, double2, double2, ushort2, op>, device::fmx<char3, double3, double3, ushort3, op>, device::fmx<char4, double4, double4, ushort4, op>  },
                    { device::fmx<schar, double, double, short, op>, device::fmx<char2, double2, double2, short2, op>, device::fmx<char3, double3, double3, short3, op>, device::fmx<char4, double4, double4, short4, op>  },
                    { device::fmx<schar, double, double, int, op>, device::fmx<char2, double2, double2, int2, op>, device::fmx<char3, double3, double3, int3, op>, device::fmx<char4, double4, double4, int4, op>  },
                    { device::fmx<schar, double, double, float, op>, device::fmx<char2, double2, double2, float2, op>, device::fmx<char3, double3, double3, float3, op>, device::fmx<char4, double4, double4, float4, op>  },
                    { device::fmx<schar, double, double, double, op>, device::fmx<char2, double2, double2, double2, op>, device::fmx<char3, double3, double3, double3, op>, device::fmx<char4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { device::fmx<ushort, uchar, uchar, uchar, op>, device::fmx<ushort2, uchar2, uchar2, uchar2, op>, device::fmx<ushort3, uchar3, uchar3, uchar3, op>, device::fmx<ushort4, uchar4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, uchar, uchar, schar, op>, device::fmx<ushort2, uchar2, uchar2, char2, op>, device::fmx<ushort3, uchar3, uchar3, char3, op>, device::fmx<ushort4, uchar4, uchar4, char4, op>  },
                    { device::fmx<ushort, uchar, uchar, ushort, op>, device::fmx<ushort2, uchar2, uchar2, ushort2, op>, device::fmx<ushort3, uchar3, uchar3, ushort3, op>, device::fmx<ushort4, uchar4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, uchar, uchar, short, op>, device::fmx<ushort2, uchar2, uchar2, short2, op>, device::fmx<ushort3, uchar3, uchar3, short3, op>, device::fmx<ushort4, uchar4, uchar4, short4, op>  },
                    { device::fmx<ushort, uchar, uchar, int, op>, device::fmx<ushort2, uchar2, uchar2, int2, op>, device::fmx<ushort3, uchar3, uchar3, int3, op>, device::fmx<ushort4, uchar4, uchar4, int4, op>  },
                    { device::fmx<ushort, uchar, uchar, float, op>, device::fmx<ushort2, uchar2, uchar2, float2, op>, device::fmx<ushort3, uchar3, uchar3, float3, op>, device::fmx<ushort4, uchar4, uchar4, float4, op>  },
                    { device::fmx<ushort, uchar, uchar, double, op>, device::fmx<ushort2, uchar2, uchar2, double2, op>, device::fmx<ushort3, uchar3, uchar3, double3, op>, device::fmx<ushort4, uchar4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<ushort, uchar, schar, uchar, op>, device::fmx<ushort2, uchar2, char2, uchar2, op>, device::fmx<ushort3, uchar3, char3, uchar3, op>, device::fmx<ushort4, uchar4, char4, uchar4, op>  },
                    { device::fmx<ushort, uchar, schar, schar, op>, device::fmx<ushort2, uchar2, char2, char2, op>, device::fmx<ushort3, uchar3, char3, char3, op>, device::fmx<ushort4, uchar4, char4, char4, op>  },
                    { device::fmx<ushort, uchar, schar, ushort, op>, device::fmx<ushort2, uchar2, char2, ushort2, op>, device::fmx<ushort3, uchar3, char3, ushort3, op>, device::fmx<ushort4, uchar4, char4, ushort4, op>  },
                    { device::fmx<ushort, uchar, schar, short, op>, device::fmx<ushort2, uchar2, char2, short2, op>, device::fmx<ushort3, uchar3, char3, short3, op>, device::fmx<ushort4, uchar4, char4, short4, op>  },
                    { device::fmx<ushort, uchar, schar, int, op>, device::fmx<ushort2, uchar2, char2, int2, op>, device::fmx<ushort3, uchar3, char3, int3, op>, device::fmx<ushort4, uchar4, char4, int4, op>  },
                    { device::fmx<ushort, uchar, schar, float, op>, device::fmx<ushort2, uchar2, char2, float2, op>, device::fmx<ushort3, uchar3, char3, float3, op>, device::fmx<ushort4, uchar4, char4, float4, op>  },
                    { device::fmx<ushort, uchar, schar, double, op>, device::fmx<ushort2, uchar2, char2, double2, op>, device::fmx<ushort3, uchar3, char3, double3, op>, device::fmx<ushort4, uchar4, char4, double4, op>  },
                },
                {
                    { device::fmx<ushort, uchar, ushort, uchar, op>, device::fmx<ushort2, uchar2, ushort2, uchar2, op>, device::fmx<ushort3, uchar3, ushort3, uchar3, op>, device::fmx<ushort4, uchar4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, uchar, ushort, schar, op>, device::fmx<ushort2, uchar2, ushort2, char2, op>, device::fmx<ushort3, uchar3, ushort3, char3, op>, device::fmx<ushort4, uchar4, ushort4, char4, op>  },
                    { device::fmx<ushort, uchar, ushort, ushort, op>, device::fmx<ushort2, uchar2, ushort2, ushort2, op>, device::fmx<ushort3, uchar3, ushort3, ushort3, op>, device::fmx<ushort4, uchar4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, uchar, ushort, short, op>, device::fmx<ushort2, uchar2, ushort2, short2, op>, device::fmx<ushort3, uchar3, ushort3, short3, op>, device::fmx<ushort4, uchar4, ushort4, short4, op>  },
                    { device::fmx<ushort, uchar, ushort, int, op>, device::fmx<ushort2, uchar2, ushort2, int2, op>, device::fmx<ushort3, uchar3, ushort3, int3, op>, device::fmx<ushort4, uchar4, ushort4, int4, op>  },
                    { device::fmx<ushort, uchar, ushort, float, op>, device::fmx<ushort2, uchar2, ushort2, float2, op>, device::fmx<ushort3, uchar3, ushort3, float3, op>, device::fmx<ushort4, uchar4, ushort4, float4, op>  },
                    { device::fmx<ushort, uchar, ushort, double, op>, device::fmx<ushort2, uchar2, ushort2, double2, op>, device::fmx<ushort3, uchar3, ushort3, double3, op>, device::fmx<ushort4, uchar4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<ushort, uchar, short, uchar, op>, device::fmx<ushort2, uchar2, short2, uchar2, op>, device::fmx<ushort3, uchar3, short3, uchar3, op>, device::fmx<ushort4, uchar4, short4, uchar4, op>  },
                    { device::fmx<ushort, uchar, short, schar, op>, device::fmx<ushort2, uchar2, short2, char2, op>, device::fmx<ushort3, uchar3, short3, char3, op>, device::fmx<ushort4, uchar4, short4, char4, op>  },
                    { device::fmx<ushort, uchar, short, ushort, op>, device::fmx<ushort2, uchar2, short2, ushort2, op>, device::fmx<ushort3, uchar3, short3, ushort3, op>, device::fmx<ushort4, uchar4, short4, ushort4, op>  },
                    { device::fmx<ushort, uchar, short, short, op>, device::fmx<ushort2, uchar2, short2, short2, op>, device::fmx<ushort3, uchar3, short3, short3, op>, device::fmx<ushort4, uchar4, short4, short4, op>  },
                    { device::fmx<ushort, uchar, short, int, op>, device::fmx<ushort2, uchar2, short2, int2, op>, device::fmx<ushort3, uchar3, short3, int3, op>, device::fmx<ushort4, uchar4, short4, int4, op>  },
                    { device::fmx<ushort, uchar, short, float, op>, device::fmx<ushort2, uchar2, short2, float2, op>, device::fmx<ushort3, uchar3, short3, float3, op>, device::fmx<ushort4, uchar4, short4, float4, op>  },
                    { device::fmx<ushort, uchar, short, double, op>, device::fmx<ushort2, uchar2, short2, double2, op>, device::fmx<ushort3, uchar3, short3, double3, op>, device::fmx<ushort4, uchar4, short4, double4, op>  },
                },
                {
                    { device::fmx<ushort, uchar, int, uchar, op>, device::fmx<ushort2, uchar2, int2, uchar2, op>, device::fmx<ushort3, uchar3, int3, uchar3, op>, device::fmx<ushort4, uchar4, int4, uchar4, op>  },
                    { device::fmx<ushort, uchar, int, schar, op>, device::fmx<ushort2, uchar2, int2, char2, op>, device::fmx<ushort3, uchar3, int3, char3, op>, device::fmx<ushort4, uchar4, int4, char4, op>  },
                    { device::fmx<ushort, uchar, int, ushort, op>, device::fmx<ushort2, uchar2, int2, ushort2, op>, device::fmx<ushort3, uchar3, int3, ushort3, op>, device::fmx<ushort4, uchar4, int4, ushort4, op>  },
                    { device::fmx<ushort, uchar, int, short, op>, device::fmx<ushort2, uchar2, int2, short2, op>, device::fmx<ushort3, uchar3, int3, short3, op>, device::fmx<ushort4, uchar4, int4, short4, op>  },
                    { device::fmx<ushort, uchar, int, int, op>, device::fmx<ushort2, uchar2, int2, int2, op>, device::fmx<ushort3, uchar3, int3, int3, op>, device::fmx<ushort4, uchar4, int4, int4, op>  },
                    { device::fmx<ushort, uchar, int, float, op>, device::fmx<ushort2, uchar2, int2, float2, op>, device::fmx<ushort3, uchar3, int3, float3, op>, device::fmx<ushort4, uchar4, int4, float4, op>  },
                    { device::fmx<ushort, uchar, int, double, op>, device::fmx<ushort2, uchar2, int2, double2, op>, device::fmx<ushort3, uchar3, int3, double3, op>, device::fmx<ushort4, uchar4, int4, double4, op>  },
                },
                {
                    { device::fmx<ushort, uchar, float, uchar, op>, device::fmx<ushort2, uchar2, float2, uchar2, op>, device::fmx<ushort3, uchar3, float3, uchar3, op>, device::fmx<ushort4, uchar4, float4, uchar4, op>  },
                    { device::fmx<ushort, uchar, float, schar, op>, device::fmx<ushort2, uchar2, float2, char2, op>, device::fmx<ushort3, uchar3, float3, char3, op>, device::fmx<ushort4, uchar4, float4, char4, op>  },
                    { device::fmx<ushort, uchar, float, ushort, op>, device::fmx<ushort2, uchar2, float2, ushort2, op>, device::fmx<ushort3, uchar3, float3, ushort3, op>, device::fmx<ushort4, uchar4, float4, ushort4, op>  },
                    { device::fmx<ushort, uchar, float, short, op>, device::fmx<ushort2, uchar2, float2, short2, op>, device::fmx<ushort3, uchar3, float3, short3, op>, device::fmx<ushort4, uchar4, float4, short4, op>  },
                    { device::fmx<ushort, uchar, float, int, op>, device::fmx<ushort2, uchar2, float2, int2, op>, device::fmx<ushort3, uchar3, float3, int3, op>, device::fmx<ushort4, uchar4, float4, int4, op>  },
                    { device::fmx<ushort, uchar, float, float, op>, device::fmx<ushort2, uchar2, float2, float2, op>, device::fmx<ushort3, uchar3, float3, float3, op>, device::fmx<ushort4, uchar4, float4, float4, op>  },
                    { device::fmx<ushort, uchar, float, double, op>, device::fmx<ushort2, uchar2, float2, double2, op>, device::fmx<ushort3, uchar3, float3, double3, op>, device::fmx<ushort4, uchar4, float4, double4, op>  },
                },
                {
                    { device::fmx<ushort, uchar, double, uchar, op>, device::fmx<ushort2, uchar2, double2, uchar2, op>, device::fmx<ushort3, uchar3, double3, uchar3, op>, device::fmx<ushort4, uchar4, double4, uchar4, op>  },
                    { device::fmx<ushort, uchar, double, schar, op>, device::fmx<ushort2, uchar2, double2, char2, op>, device::fmx<ushort3, uchar3, double3, char3, op>, device::fmx<ushort4, uchar4, double4, char4, op>  },
                    { device::fmx<ushort, uchar, double, ushort, op>, device::fmx<ushort2, uchar2, double2, ushort2, op>, device::fmx<ushort3, uchar3, double3, ushort3, op>, device::fmx<ushort4, uchar4, double4, ushort4, op>  },
                    { device::fmx<ushort, uchar, double, short, op>, device::fmx<ushort2, uchar2, double2, short2, op>, device::fmx<ushort3, uchar3, double3, short3, op>, device::fmx<ushort4, uchar4, double4, short4, op>  },
                    { device::fmx<ushort, uchar, double, int, op>, device::fmx<ushort2, uchar2, double2, int2, op>, device::fmx<ushort3, uchar3, double3, int3, op>, device::fmx<ushort4, uchar4, double4, int4, op>  },
                    { device::fmx<ushort, uchar, double, float, op>, device::fmx<ushort2, uchar2, double2, float2, op>, device::fmx<ushort3, uchar3, double3, float3, op>, device::fmx<ushort4, uchar4, double4, float4, op>  },
                    { device::fmx<ushort, uchar, double, double, op>, device::fmx<ushort2, uchar2, double2, double2, op>, device::fmx<ushort3, uchar3, double3, double3, op>, device::fmx<ushort4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<ushort, schar, uchar, uchar, op>, device::fmx<ushort2, char2, uchar2, uchar2, op>, device::fmx<ushort3, char3, uchar3, uchar3, op>, device::fmx<ushort4, char4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, schar, uchar, schar, op>, device::fmx<ushort2, char2, uchar2, char2, op>, device::fmx<ushort3, char3, uchar3, char3, op>, device::fmx<ushort4, char4, uchar4, char4, op>  },
                    { device::fmx<ushort, schar, uchar, ushort, op>, device::fmx<ushort2, char2, uchar2, ushort2, op>, device::fmx<ushort3, char3, uchar3, ushort3, op>, device::fmx<ushort4, char4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, schar, uchar, short, op>, device::fmx<ushort2, char2, uchar2, short2, op>, device::fmx<ushort3, char3, uchar3, short3, op>, device::fmx<ushort4, char4, uchar4, short4, op>  },
                    { device::fmx<ushort, schar, uchar, int, op>, device::fmx<ushort2, char2, uchar2, int2, op>, device::fmx<ushort3, char3, uchar3, int3, op>, device::fmx<ushort4, char4, uchar4, int4, op>  },
                    { device::fmx<ushort, schar, uchar, float, op>, device::fmx<ushort2, char2, uchar2, float2, op>, device::fmx<ushort3, char3, uchar3, float3, op>, device::fmx<ushort4, char4, uchar4, float4, op>  },
                    { device::fmx<ushort, schar, uchar, double, op>, device::fmx<ushort2, char2, uchar2, double2, op>, device::fmx<ushort3, char3, uchar3, double3, op>, device::fmx<ushort4, char4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<ushort, schar, schar, uchar, op>, device::fmx<ushort2, char2, char2, uchar2, op>, device::fmx<ushort3, char3, char3, uchar3, op>, device::fmx<ushort4, char4, char4, uchar4, op>  },
                    { device::fmx<ushort, schar, schar, schar, op>, device::fmx<ushort2, char2, char2, char2, op>, device::fmx<ushort3, char3, char3, char3, op>, device::fmx<ushort4, char4, char4, char4, op>  },
                    { device::fmx<ushort, schar, schar, ushort, op>, device::fmx<ushort2, char2, char2, ushort2, op>, device::fmx<ushort3, char3, char3, ushort3, op>, device::fmx<ushort4, char4, char4, ushort4, op>  },
                    { device::fmx<ushort, schar, schar, short, op>, device::fmx<ushort2, char2, char2, short2, op>, device::fmx<ushort3, char3, char3, short3, op>, device::fmx<ushort4, char4, char4, short4, op>  },
                    { device::fmx<ushort, schar, schar, int, op>, device::fmx<ushort2, char2, char2, int2, op>, device::fmx<ushort3, char3, char3, int3, op>, device::fmx<ushort4, char4, char4, int4, op>  },
                    { device::fmx<ushort, schar, schar, float, op>, device::fmx<ushort2, char2, char2, float2, op>, device::fmx<ushort3, char3, char3, float3, op>, device::fmx<ushort4, char4, char4, float4, op>  },
                    { device::fmx<ushort, schar, schar, double, op>, device::fmx<ushort2, char2, char2, double2, op>, device::fmx<ushort3, char3, char3, double3, op>, device::fmx<ushort4, char4, char4, double4, op>  },
                },
                {
                    { device::fmx<ushort, schar, ushort, uchar, op>, device::fmx<ushort2, char2, ushort2, uchar2, op>, device::fmx<ushort3, char3, ushort3, uchar3, op>, device::fmx<ushort4, char4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, schar, ushort, schar, op>, device::fmx<ushort2, char2, ushort2, char2, op>, device::fmx<ushort3, char3, ushort3, char3, op>, device::fmx<ushort4, char4, ushort4, char4, op>  },
                    { device::fmx<ushort, schar, ushort, ushort, op>, device::fmx<ushort2, char2, ushort2, ushort2, op>, device::fmx<ushort3, char3, ushort3, ushort3, op>, device::fmx<ushort4, char4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, schar, ushort, short, op>, device::fmx<ushort2, char2, ushort2, short2, op>, device::fmx<ushort3, char3, ushort3, short3, op>, device::fmx<ushort4, char4, ushort4, short4, op>  },
                    { device::fmx<ushort, schar, ushort, int, op>, device::fmx<ushort2, char2, ushort2, int2, op>, device::fmx<ushort3, char3, ushort3, int3, op>, device::fmx<ushort4, char4, ushort4, int4, op>  },
                    { device::fmx<ushort, schar, ushort, float, op>, device::fmx<ushort2, char2, ushort2, float2, op>, device::fmx<ushort3, char3, ushort3, float3, op>, device::fmx<ushort4, char4, ushort4, float4, op>  },
                    { device::fmx<ushort, schar, ushort, double, op>, device::fmx<ushort2, char2, ushort2, double2, op>, device::fmx<ushort3, char3, ushort3, double3, op>, device::fmx<ushort4, char4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<ushort, schar, short, uchar, op>, device::fmx<ushort2, char2, short2, uchar2, op>, device::fmx<ushort3, char3, short3, uchar3, op>, device::fmx<ushort4, char4, short4, uchar4, op>  },
                    { device::fmx<ushort, schar, short, schar, op>, device::fmx<ushort2, char2, short2, char2, op>, device::fmx<ushort3, char3, short3, char3, op>, device::fmx<ushort4, char4, short4, char4, op>  },
                    { device::fmx<ushort, schar, short, ushort, op>, device::fmx<ushort2, char2, short2, ushort2, op>, device::fmx<ushort3, char3, short3, ushort3, op>, device::fmx<ushort4, char4, short4, ushort4, op>  },
                    { device::fmx<ushort, schar, short, short, op>, device::fmx<ushort2, char2, short2, short2, op>, device::fmx<ushort3, char3, short3, short3, op>, device::fmx<ushort4, char4, short4, short4, op>  },
                    { device::fmx<ushort, schar, short, int, op>, device::fmx<ushort2, char2, short2, int2, op>, device::fmx<ushort3, char3, short3, int3, op>, device::fmx<ushort4, char4, short4, int4, op>  },
                    { device::fmx<ushort, schar, short, float, op>, device::fmx<ushort2, char2, short2, float2, op>, device::fmx<ushort3, char3, short3, float3, op>, device::fmx<ushort4, char4, short4, float4, op>  },
                    { device::fmx<ushort, schar, short, double, op>, device::fmx<ushort2, char2, short2, double2, op>, device::fmx<ushort3, char3, short3, double3, op>, device::fmx<ushort4, char4, short4, double4, op>  },
                },
                {
                    { device::fmx<ushort, schar, int, uchar, op>, device::fmx<ushort2, char2, int2, uchar2, op>, device::fmx<ushort3, char3, int3, uchar3, op>, device::fmx<ushort4, char4, int4, uchar4, op>  },
                    { device::fmx<ushort, schar, int, schar, op>, device::fmx<ushort2, char2, int2, char2, op>, device::fmx<ushort3, char3, int3, char3, op>, device::fmx<ushort4, char4, int4, char4, op>  },
                    { device::fmx<ushort, schar, int, ushort, op>, device::fmx<ushort2, char2, int2, ushort2, op>, device::fmx<ushort3, char3, int3, ushort3, op>, device::fmx<ushort4, char4, int4, ushort4, op>  },
                    { device::fmx<ushort, schar, int, short, op>, device::fmx<ushort2, char2, int2, short2, op>, device::fmx<ushort3, char3, int3, short3, op>, device::fmx<ushort4, char4, int4, short4, op>  },
                    { device::fmx<ushort, schar, int, int, op>, device::fmx<ushort2, char2, int2, int2, op>, device::fmx<ushort3, char3, int3, int3, op>, device::fmx<ushort4, char4, int4, int4, op>  },
                    { device::fmx<ushort, schar, int, float, op>, device::fmx<ushort2, char2, int2, float2, op>, device::fmx<ushort3, char3, int3, float3, op>, device::fmx<ushort4, char4, int4, float4, op>  },
                    { device::fmx<ushort, schar, int, double, op>, device::fmx<ushort2, char2, int2, double2, op>, device::fmx<ushort3, char3, int3, double3, op>, device::fmx<ushort4, char4, int4, double4, op>  },
                },
                {
                    { device::fmx<ushort, schar, float, uchar, op>, device::fmx<ushort2, char2, float2, uchar2, op>, device::fmx<ushort3, char3, float3, uchar3, op>, device::fmx<ushort4, char4, float4, uchar4, op>  },
                    { device::fmx<ushort, schar, float, schar, op>, device::fmx<ushort2, char2, float2, char2, op>, device::fmx<ushort3, char3, float3, char3, op>, device::fmx<ushort4, char4, float4, char4, op>  },
                    { device::fmx<ushort, schar, float, ushort, op>, device::fmx<ushort2, char2, float2, ushort2, op>, device::fmx<ushort3, char3, float3, ushort3, op>, device::fmx<ushort4, char4, float4, ushort4, op>  },
                    { device::fmx<ushort, schar, float, short, op>, device::fmx<ushort2, char2, float2, short2, op>, device::fmx<ushort3, char3, float3, short3, op>, device::fmx<ushort4, char4, float4, short4, op>  },
                    { device::fmx<ushort, schar, float, int, op>, device::fmx<ushort2, char2, float2, int2, op>, device::fmx<ushort3, char3, float3, int3, op>, device::fmx<ushort4, char4, float4, int4, op>  },
                    { device::fmx<ushort, schar, float, float, op>, device::fmx<ushort2, char2, float2, float2, op>, device::fmx<ushort3, char3, float3, float3, op>, device::fmx<ushort4, char4, float4, float4, op>  },
                    { device::fmx<ushort, schar, float, double, op>, device::fmx<ushort2, char2, float2, double2, op>, device::fmx<ushort3, char3, float3, double3, op>, device::fmx<ushort4, char4, float4, double4, op>  },
                },
                {
                    { device::fmx<ushort, schar, double, uchar, op>, device::fmx<ushort2, char2, double2, uchar2, op>, device::fmx<ushort3, char3, double3, uchar3, op>, device::fmx<ushort4, char4, double4, uchar4, op>  },
                    { device::fmx<ushort, schar, double, schar, op>, device::fmx<ushort2, char2, double2, char2, op>, device::fmx<ushort3, char3, double3, char3, op>, device::fmx<ushort4, char4, double4, char4, op>  },
                    { device::fmx<ushort, schar, double, ushort, op>, device::fmx<ushort2, char2, double2, ushort2, op>, device::fmx<ushort3, char3, double3, ushort3, op>, device::fmx<ushort4, char4, double4, ushort4, op>  },
                    { device::fmx<ushort, schar, double, short, op>, device::fmx<ushort2, char2, double2, short2, op>, device::fmx<ushort3, char3, double3, short3, op>, device::fmx<ushort4, char4, double4, short4, op>  },
                    { device::fmx<ushort, schar, double, int, op>, device::fmx<ushort2, char2, double2, int2, op>, device::fmx<ushort3, char3, double3, int3, op>, device::fmx<ushort4, char4, double4, int4, op>  },
                    { device::fmx<ushort, schar, double, float, op>, device::fmx<ushort2, char2, double2, float2, op>, device::fmx<ushort3, char3, double3, float3, op>, device::fmx<ushort4, char4, double4, float4, op>  },
                    { device::fmx<ushort, schar, double, double, op>, device::fmx<ushort2, char2, double2, double2, op>, device::fmx<ushort3, char3, double3, double3, op>, device::fmx<ushort4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<ushort, ushort, uchar, uchar, op>, device::fmx<ushort2, ushort2, uchar2, uchar2, op>, device::fmx<ushort3, ushort3, uchar3, uchar3, op>, device::fmx<ushort4, ushort4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, ushort, uchar, schar, op>, device::fmx<ushort2, ushort2, uchar2, char2, op>, device::fmx<ushort3, ushort3, uchar3, char3, op>, device::fmx<ushort4, ushort4, uchar4, char4, op>  },
                    { device::fmx<ushort, ushort, uchar, ushort, op>, device::fmx<ushort2, ushort2, uchar2, ushort2, op>, device::fmx<ushort3, ushort3, uchar3, ushort3, op>, device::fmx<ushort4, ushort4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, ushort, uchar, short, op>, device::fmx<ushort2, ushort2, uchar2, short2, op>, device::fmx<ushort3, ushort3, uchar3, short3, op>, device::fmx<ushort4, ushort4, uchar4, short4, op>  },
                    { device::fmx<ushort, ushort, uchar, int, op>, device::fmx<ushort2, ushort2, uchar2, int2, op>, device::fmx<ushort3, ushort3, uchar3, int3, op>, device::fmx<ushort4, ushort4, uchar4, int4, op>  },
                    { device::fmx<ushort, ushort, uchar, float, op>, device::fmx<ushort2, ushort2, uchar2, float2, op>, device::fmx<ushort3, ushort3, uchar3, float3, op>, device::fmx<ushort4, ushort4, uchar4, float4, op>  },
                    { device::fmx<ushort, ushort, uchar, double, op>, device::fmx<ushort2, ushort2, uchar2, double2, op>, device::fmx<ushort3, ushort3, uchar3, double3, op>, device::fmx<ushort4, ushort4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<ushort, ushort, schar, uchar, op>, device::fmx<ushort2, ushort2, char2, uchar2, op>, device::fmx<ushort3, ushort3, char3, uchar3, op>, device::fmx<ushort4, ushort4, char4, uchar4, op>  },
                    { device::fmx<ushort, ushort, schar, schar, op>, device::fmx<ushort2, ushort2, char2, char2, op>, device::fmx<ushort3, ushort3, char3, char3, op>, device::fmx<ushort4, ushort4, char4, char4, op>  },
                    { device::fmx<ushort, ushort, schar, ushort, op>, device::fmx<ushort2, ushort2, char2, ushort2, op>, device::fmx<ushort3, ushort3, char3, ushort3, op>, device::fmx<ushort4, ushort4, char4, ushort4, op>  },
                    { device::fmx<ushort, ushort, schar, short, op>, device::fmx<ushort2, ushort2, char2, short2, op>, device::fmx<ushort3, ushort3, char3, short3, op>, device::fmx<ushort4, ushort4, char4, short4, op>  },
                    { device::fmx<ushort, ushort, schar, int, op>, device::fmx<ushort2, ushort2, char2, int2, op>, device::fmx<ushort3, ushort3, char3, int3, op>, device::fmx<ushort4, ushort4, char4, int4, op>  },
                    { device::fmx<ushort, ushort, schar, float, op>, device::fmx<ushort2, ushort2, char2, float2, op>, device::fmx<ushort3, ushort3, char3, float3, op>, device::fmx<ushort4, ushort4, char4, float4, op>  },
                    { device::fmx<ushort, ushort, schar, double, op>, device::fmx<ushort2, ushort2, char2, double2, op>, device::fmx<ushort3, ushort3, char3, double3, op>, device::fmx<ushort4, ushort4, char4, double4, op>  },
                },
                {
                    { device::fmx<ushort, ushort, ushort, uchar, op>, device::fmx<ushort2, ushort2, ushort2, uchar2, op>, device::fmx<ushort3, ushort3, ushort3, uchar3, op>, device::fmx<ushort4, ushort4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, ushort, ushort, schar, op>, device::fmx<ushort2, ushort2, ushort2, char2, op>, device::fmx<ushort3, ushort3, ushort3, char3, op>, device::fmx<ushort4, ushort4, ushort4, char4, op>  },
                    { device::fmx<ushort, ushort, ushort, ushort, op>, device::fmx<ushort2, ushort2, ushort2, ushort2, op>, device::fmx<ushort3, ushort3, ushort3, ushort3, op>, device::fmx<ushort4, ushort4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, ushort, ushort, short, op>, device::fmx<ushort2, ushort2, ushort2, short2, op>, device::fmx<ushort3, ushort3, ushort3, short3, op>, device::fmx<ushort4, ushort4, ushort4, short4, op>  },
                    { device::fmx<ushort, ushort, ushort, int, op>, device::fmx<ushort2, ushort2, ushort2, int2, op>, device::fmx<ushort3, ushort3, ushort3, int3, op>, device::fmx<ushort4, ushort4, ushort4, int4, op>  },
                    { device::fmx<ushort, ushort, ushort, float, op>, device::fmx<ushort2, ushort2, ushort2, float2, op>, device::fmx<ushort3, ushort3, ushort3, float3, op>, device::fmx<ushort4, ushort4, ushort4, float4, op>  },
                    { device::fmx<ushort, ushort, ushort, double, op>, device::fmx<ushort2, ushort2, ushort2, double2, op>, device::fmx<ushort3, ushort3, ushort3, double3, op>, device::fmx<ushort4, ushort4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<ushort, ushort, short, uchar, op>, device::fmx<ushort2, ushort2, short2, uchar2, op>, device::fmx<ushort3, ushort3, short3, uchar3, op>, device::fmx<ushort4, ushort4, short4, uchar4, op>  },
                    { device::fmx<ushort, ushort, short, schar, op>, device::fmx<ushort2, ushort2, short2, char2, op>, device::fmx<ushort3, ushort3, short3, char3, op>, device::fmx<ushort4, ushort4, short4, char4, op>  },
                    { device::fmx<ushort, ushort, short, ushort, op>, device::fmx<ushort2, ushort2, short2, ushort2, op>, device::fmx<ushort3, ushort3, short3, ushort3, op>, device::fmx<ushort4, ushort4, short4, ushort4, op>  },
                    { device::fmx<ushort, ushort, short, short, op>, device::fmx<ushort2, ushort2, short2, short2, op>, device::fmx<ushort3, ushort3, short3, short3, op>, device::fmx<ushort4, ushort4, short4, short4, op>  },
                    { device::fmx<ushort, ushort, short, int, op>, device::fmx<ushort2, ushort2, short2, int2, op>, device::fmx<ushort3, ushort3, short3, int3, op>, device::fmx<ushort4, ushort4, short4, int4, op>  },
                    { device::fmx<ushort, ushort, short, float, op>, device::fmx<ushort2, ushort2, short2, float2, op>, device::fmx<ushort3, ushort3, short3, float3, op>, device::fmx<ushort4, ushort4, short4, float4, op>  },
                    { device::fmx<ushort, ushort, short, double, op>, device::fmx<ushort2, ushort2, short2, double2, op>, device::fmx<ushort3, ushort3, short3, double3, op>, device::fmx<ushort4, ushort4, short4, double4, op>  },
                },
                {
                    { device::fmx<ushort, ushort, int, uchar, op>, device::fmx<ushort2, ushort2, int2, uchar2, op>, device::fmx<ushort3, ushort3, int3, uchar3, op>, device::fmx<ushort4, ushort4, int4, uchar4, op>  },
                    { device::fmx<ushort, ushort, int, schar, op>, device::fmx<ushort2, ushort2, int2, char2, op>, device::fmx<ushort3, ushort3, int3, char3, op>, device::fmx<ushort4, ushort4, int4, char4, op>  },
                    { device::fmx<ushort, ushort, int, ushort, op>, device::fmx<ushort2, ushort2, int2, ushort2, op>, device::fmx<ushort3, ushort3, int3, ushort3, op>, device::fmx<ushort4, ushort4, int4, ushort4, op>  },
                    { device::fmx<ushort, ushort, int, short, op>, device::fmx<ushort2, ushort2, int2, short2, op>, device::fmx<ushort3, ushort3, int3, short3, op>, device::fmx<ushort4, ushort4, int4, short4, op>  },
                    { device::fmx<ushort, ushort, int, int, op>, device::fmx<ushort2, ushort2, int2, int2, op>, device::fmx<ushort3, ushort3, int3, int3, op>, device::fmx<ushort4, ushort4, int4, int4, op>  },
                    { device::fmx<ushort, ushort, int, float, op>, device::fmx<ushort2, ushort2, int2, float2, op>, device::fmx<ushort3, ushort3, int3, float3, op>, device::fmx<ushort4, ushort4, int4, float4, op>  },
                    { device::fmx<ushort, ushort, int, double, op>, device::fmx<ushort2, ushort2, int2, double2, op>, device::fmx<ushort3, ushort3, int3, double3, op>, device::fmx<ushort4, ushort4, int4, double4, op>  },
                },
                {
                    { device::fmx<ushort, ushort, float, uchar, op>, device::fmx<ushort2, ushort2, float2, uchar2, op>, device::fmx<ushort3, ushort3, float3, uchar3, op>, device::fmx<ushort4, ushort4, float4, uchar4, op>  },
                    { device::fmx<ushort, ushort, float, schar, op>, device::fmx<ushort2, ushort2, float2, char2, op>, device::fmx<ushort3, ushort3, float3, char3, op>, device::fmx<ushort4, ushort4, float4, char4, op>  },
                    { device::fmx<ushort, ushort, float, ushort, op>, device::fmx<ushort2, ushort2, float2, ushort2, op>, device::fmx<ushort3, ushort3, float3, ushort3, op>, device::fmx<ushort4, ushort4, float4, ushort4, op>  },
                    { device::fmx<ushort, ushort, float, short, op>, device::fmx<ushort2, ushort2, float2, short2, op>, device::fmx<ushort3, ushort3, float3, short3, op>, device::fmx<ushort4, ushort4, float4, short4, op>  },
                    { device::fmx<ushort, ushort, float, int, op>, device::fmx<ushort2, ushort2, float2, int2, op>, device::fmx<ushort3, ushort3, float3, int3, op>, device::fmx<ushort4, ushort4, float4, int4, op>  },
                    { device::fmx<ushort, ushort, float, float, op>, device::fmx<ushort2, ushort2, float2, float2, op>, device::fmx<ushort3, ushort3, float3, float3, op>, device::fmx<ushort4, ushort4, float4, float4, op>  },
                    { device::fmx<ushort, ushort, float, double, op>, device::fmx<ushort2, ushort2, float2, double2, op>, device::fmx<ushort3, ushort3, float3, double3, op>, device::fmx<ushort4, ushort4, float4, double4, op>  },
                },
                {
                    { device::fmx<ushort, ushort, double, uchar, op>, device::fmx<ushort2, ushort2, double2, uchar2, op>, device::fmx<ushort3, ushort3, double3, uchar3, op>, device::fmx<ushort4, ushort4, double4, uchar4, op>  },
                    { device::fmx<ushort, ushort, double, schar, op>, device::fmx<ushort2, ushort2, double2, char2, op>, device::fmx<ushort3, ushort3, double3, char3, op>, device::fmx<ushort4, ushort4, double4, char4, op>  },
                    { device::fmx<ushort, ushort, double, ushort, op>, device::fmx<ushort2, ushort2, double2, ushort2, op>, device::fmx<ushort3, ushort3, double3, ushort3, op>, device::fmx<ushort4, ushort4, double4, ushort4, op>  },
                    { device::fmx<ushort, ushort, double, short, op>, device::fmx<ushort2, ushort2, double2, short2, op>, device::fmx<ushort3, ushort3, double3, short3, op>, device::fmx<ushort4, ushort4, double4, short4, op>  },
                    { device::fmx<ushort, ushort, double, int, op>, device::fmx<ushort2, ushort2, double2, int2, op>, device::fmx<ushort3, ushort3, double3, int3, op>, device::fmx<ushort4, ushort4, double4, int4, op>  },
                    { device::fmx<ushort, ushort, double, float, op>, device::fmx<ushort2, ushort2, double2, float2, op>, device::fmx<ushort3, ushort3, double3, float3, op>, device::fmx<ushort4, ushort4, double4, float4, op>  },
                    { device::fmx<ushort, ushort, double, double, op>, device::fmx<ushort2, ushort2, double2, double2, op>, device::fmx<ushort3, ushort3, double3, double3, op>, device::fmx<ushort4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<ushort, short, uchar, uchar, op>, device::fmx<ushort2, short2, uchar2, uchar2, op>, device::fmx<ushort3, short3, uchar3, uchar3, op>, device::fmx<ushort4, short4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, short, uchar, schar, op>, device::fmx<ushort2, short2, uchar2, char2, op>, device::fmx<ushort3, short3, uchar3, char3, op>, device::fmx<ushort4, short4, uchar4, char4, op>  },
                    { device::fmx<ushort, short, uchar, ushort, op>, device::fmx<ushort2, short2, uchar2, ushort2, op>, device::fmx<ushort3, short3, uchar3, ushort3, op>, device::fmx<ushort4, short4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, short, uchar, short, op>, device::fmx<ushort2, short2, uchar2, short2, op>, device::fmx<ushort3, short3, uchar3, short3, op>, device::fmx<ushort4, short4, uchar4, short4, op>  },
                    { device::fmx<ushort, short, uchar, int, op>, device::fmx<ushort2, short2, uchar2, int2, op>, device::fmx<ushort3, short3, uchar3, int3, op>, device::fmx<ushort4, short4, uchar4, int4, op>  },
                    { device::fmx<ushort, short, uchar, float, op>, device::fmx<ushort2, short2, uchar2, float2, op>, device::fmx<ushort3, short3, uchar3, float3, op>, device::fmx<ushort4, short4, uchar4, float4, op>  },
                    { device::fmx<ushort, short, uchar, double, op>, device::fmx<ushort2, short2, uchar2, double2, op>, device::fmx<ushort3, short3, uchar3, double3, op>, device::fmx<ushort4, short4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<ushort, short, schar, uchar, op>, device::fmx<ushort2, short2, char2, uchar2, op>, device::fmx<ushort3, short3, char3, uchar3, op>, device::fmx<ushort4, short4, char4, uchar4, op>  },
                    { device::fmx<ushort, short, schar, schar, op>, device::fmx<ushort2, short2, char2, char2, op>, device::fmx<ushort3, short3, char3, char3, op>, device::fmx<ushort4, short4, char4, char4, op>  },
                    { device::fmx<ushort, short, schar, ushort, op>, device::fmx<ushort2, short2, char2, ushort2, op>, device::fmx<ushort3, short3, char3, ushort3, op>, device::fmx<ushort4, short4, char4, ushort4, op>  },
                    { device::fmx<ushort, short, schar, short, op>, device::fmx<ushort2, short2, char2, short2, op>, device::fmx<ushort3, short3, char3, short3, op>, device::fmx<ushort4, short4, char4, short4, op>  },
                    { device::fmx<ushort, short, schar, int, op>, device::fmx<ushort2, short2, char2, int2, op>, device::fmx<ushort3, short3, char3, int3, op>, device::fmx<ushort4, short4, char4, int4, op>  },
                    { device::fmx<ushort, short, schar, float, op>, device::fmx<ushort2, short2, char2, float2, op>, device::fmx<ushort3, short3, char3, float3, op>, device::fmx<ushort4, short4, char4, float4, op>  },
                    { device::fmx<ushort, short, schar, double, op>, device::fmx<ushort2, short2, char2, double2, op>, device::fmx<ushort3, short3, char3, double3, op>, device::fmx<ushort4, short4, char4, double4, op>  },
                },
                {
                    { device::fmx<ushort, short, ushort, uchar, op>, device::fmx<ushort2, short2, ushort2, uchar2, op>, device::fmx<ushort3, short3, ushort3, uchar3, op>, device::fmx<ushort4, short4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, short, ushort, schar, op>, device::fmx<ushort2, short2, ushort2, char2, op>, device::fmx<ushort3, short3, ushort3, char3, op>, device::fmx<ushort4, short4, ushort4, char4, op>  },
                    { device::fmx<ushort, short, ushort, ushort, op>, device::fmx<ushort2, short2, ushort2, ushort2, op>, device::fmx<ushort3, short3, ushort3, ushort3, op>, device::fmx<ushort4, short4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, short, ushort, short, op>, device::fmx<ushort2, short2, ushort2, short2, op>, device::fmx<ushort3, short3, ushort3, short3, op>, device::fmx<ushort4, short4, ushort4, short4, op>  },
                    { device::fmx<ushort, short, ushort, int, op>, device::fmx<ushort2, short2, ushort2, int2, op>, device::fmx<ushort3, short3, ushort3, int3, op>, device::fmx<ushort4, short4, ushort4, int4, op>  },
                    { device::fmx<ushort, short, ushort, float, op>, device::fmx<ushort2, short2, ushort2, float2, op>, device::fmx<ushort3, short3, ushort3, float3, op>, device::fmx<ushort4, short4, ushort4, float4, op>  },
                    { device::fmx<ushort, short, ushort, double, op>, device::fmx<ushort2, short2, ushort2, double2, op>, device::fmx<ushort3, short3, ushort3, double3, op>, device::fmx<ushort4, short4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<ushort, short, short, uchar, op>, device::fmx<ushort2, short2, short2, uchar2, op>, device::fmx<ushort3, short3, short3, uchar3, op>, device::fmx<ushort4, short4, short4, uchar4, op>  },
                    { device::fmx<ushort, short, short, schar, op>, device::fmx<ushort2, short2, short2, char2, op>, device::fmx<ushort3, short3, short3, char3, op>, device::fmx<ushort4, short4, short4, char4, op>  },
                    { device::fmx<ushort, short, short, ushort, op>, device::fmx<ushort2, short2, short2, ushort2, op>, device::fmx<ushort3, short3, short3, ushort3, op>, device::fmx<ushort4, short4, short4, ushort4, op>  },
                    { device::fmx<ushort, short, short, short, op>, device::fmx<ushort2, short2, short2, short2, op>, device::fmx<ushort3, short3, short3, short3, op>, device::fmx<ushort4, short4, short4, short4, op>  },
                    { device::fmx<ushort, short, short, int, op>, device::fmx<ushort2, short2, short2, int2, op>, device::fmx<ushort3, short3, short3, int3, op>, device::fmx<ushort4, short4, short4, int4, op>  },
                    { device::fmx<ushort, short, short, float, op>, device::fmx<ushort2, short2, short2, float2, op>, device::fmx<ushort3, short3, short3, float3, op>, device::fmx<ushort4, short4, short4, float4, op>  },
                    { device::fmx<ushort, short, short, double, op>, device::fmx<ushort2, short2, short2, double2, op>, device::fmx<ushort3, short3, short3, double3, op>, device::fmx<ushort4, short4, short4, double4, op>  },
                },
                {
                    { device::fmx<ushort, short, int, uchar, op>, device::fmx<ushort2, short2, int2, uchar2, op>, device::fmx<ushort3, short3, int3, uchar3, op>, device::fmx<ushort4, short4, int4, uchar4, op>  },
                    { device::fmx<ushort, short, int, schar, op>, device::fmx<ushort2, short2, int2, char2, op>, device::fmx<ushort3, short3, int3, char3, op>, device::fmx<ushort4, short4, int4, char4, op>  },
                    { device::fmx<ushort, short, int, ushort, op>, device::fmx<ushort2, short2, int2, ushort2, op>, device::fmx<ushort3, short3, int3, ushort3, op>, device::fmx<ushort4, short4, int4, ushort4, op>  },
                    { device::fmx<ushort, short, int, short, op>, device::fmx<ushort2, short2, int2, short2, op>, device::fmx<ushort3, short3, int3, short3, op>, device::fmx<ushort4, short4, int4, short4, op>  },
                    { device::fmx<ushort, short, int, int, op>, device::fmx<ushort2, short2, int2, int2, op>, device::fmx<ushort3, short3, int3, int3, op>, device::fmx<ushort4, short4, int4, int4, op>  },
                    { device::fmx<ushort, short, int, float, op>, device::fmx<ushort2, short2, int2, float2, op>, device::fmx<ushort3, short3, int3, float3, op>, device::fmx<ushort4, short4, int4, float4, op>  },
                    { device::fmx<ushort, short, int, double, op>, device::fmx<ushort2, short2, int2, double2, op>, device::fmx<ushort3, short3, int3, double3, op>, device::fmx<ushort4, short4, int4, double4, op>  },
                },
                {
                    { device::fmx<ushort, short, float, uchar, op>, device::fmx<ushort2, short2, float2, uchar2, op>, device::fmx<ushort3, short3, float3, uchar3, op>, device::fmx<ushort4, short4, float4, uchar4, op>  },
                    { device::fmx<ushort, short, float, schar, op>, device::fmx<ushort2, short2, float2, char2, op>, device::fmx<ushort3, short3, float3, char3, op>, device::fmx<ushort4, short4, float4, char4, op>  },
                    { device::fmx<ushort, short, float, ushort, op>, device::fmx<ushort2, short2, float2, ushort2, op>, device::fmx<ushort3, short3, float3, ushort3, op>, device::fmx<ushort4, short4, float4, ushort4, op>  },
                    { device::fmx<ushort, short, float, short, op>, device::fmx<ushort2, short2, float2, short2, op>, device::fmx<ushort3, short3, float3, short3, op>, device::fmx<ushort4, short4, float4, short4, op>  },
                    { device::fmx<ushort, short, float, int, op>, device::fmx<ushort2, short2, float2, int2, op>, device::fmx<ushort3, short3, float3, int3, op>, device::fmx<ushort4, short4, float4, int4, op>  },
                    { device::fmx<ushort, short, float, float, op>, device::fmx<ushort2, short2, float2, float2, op>, device::fmx<ushort3, short3, float3, float3, op>, device::fmx<ushort4, short4, float4, float4, op>  },
                    { device::fmx<ushort, short, float, double, op>, device::fmx<ushort2, short2, float2, double2, op>, device::fmx<ushort3, short3, float3, double3, op>, device::fmx<ushort4, short4, float4, double4, op>  },
                },
                {
                    { device::fmx<ushort, short, double, uchar, op>, device::fmx<ushort2, short2, double2, uchar2, op>, device::fmx<ushort3, short3, double3, uchar3, op>, device::fmx<ushort4, short4, double4, uchar4, op>  },
                    { device::fmx<ushort, short, double, schar, op>, device::fmx<ushort2, short2, double2, char2, op>, device::fmx<ushort3, short3, double3, char3, op>, device::fmx<ushort4, short4, double4, char4, op>  },
                    { device::fmx<ushort, short, double, ushort, op>, device::fmx<ushort2, short2, double2, ushort2, op>, device::fmx<ushort3, short3, double3, ushort3, op>, device::fmx<ushort4, short4, double4, ushort4, op>  },
                    { device::fmx<ushort, short, double, short, op>, device::fmx<ushort2, short2, double2, short2, op>, device::fmx<ushort3, short3, double3, short3, op>, device::fmx<ushort4, short4, double4, short4, op>  },
                    { device::fmx<ushort, short, double, int, op>, device::fmx<ushort2, short2, double2, int2, op>, device::fmx<ushort3, short3, double3, int3, op>, device::fmx<ushort4, short4, double4, int4, op>  },
                    { device::fmx<ushort, short, double, float, op>, device::fmx<ushort2, short2, double2, float2, op>, device::fmx<ushort3, short3, double3, float3, op>, device::fmx<ushort4, short4, double4, float4, op>  },
                    { device::fmx<ushort, short, double, double, op>, device::fmx<ushort2, short2, double2, double2, op>, device::fmx<ushort3, short3, double3, double3, op>, device::fmx<ushort4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<ushort, int, uchar, uchar, op>, device::fmx<ushort2, int2, uchar2, uchar2, op>, device::fmx<ushort3, int3, uchar3, uchar3, op>, device::fmx<ushort4, int4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, int, uchar, schar, op>, device::fmx<ushort2, int2, uchar2, char2, op>, device::fmx<ushort3, int3, uchar3, char3, op>, device::fmx<ushort4, int4, uchar4, char4, op>  },
                    { device::fmx<ushort, int, uchar, ushort, op>, device::fmx<ushort2, int2, uchar2, ushort2, op>, device::fmx<ushort3, int3, uchar3, ushort3, op>, device::fmx<ushort4, int4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, int, uchar, short, op>, device::fmx<ushort2, int2, uchar2, short2, op>, device::fmx<ushort3, int3, uchar3, short3, op>, device::fmx<ushort4, int4, uchar4, short4, op>  },
                    { device::fmx<ushort, int, uchar, int, op>, device::fmx<ushort2, int2, uchar2, int2, op>, device::fmx<ushort3, int3, uchar3, int3, op>, device::fmx<ushort4, int4, uchar4, int4, op>  },
                    { device::fmx<ushort, int, uchar, float, op>, device::fmx<ushort2, int2, uchar2, float2, op>, device::fmx<ushort3, int3, uchar3, float3, op>, device::fmx<ushort4, int4, uchar4, float4, op>  },
                    { device::fmx<ushort, int, uchar, double, op>, device::fmx<ushort2, int2, uchar2, double2, op>, device::fmx<ushort3, int3, uchar3, double3, op>, device::fmx<ushort4, int4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<ushort, int, schar, uchar, op>, device::fmx<ushort2, int2, char2, uchar2, op>, device::fmx<ushort3, int3, char3, uchar3, op>, device::fmx<ushort4, int4, char4, uchar4, op>  },
                    { device::fmx<ushort, int, schar, schar, op>, device::fmx<ushort2, int2, char2, char2, op>, device::fmx<ushort3, int3, char3, char3, op>, device::fmx<ushort4, int4, char4, char4, op>  },
                    { device::fmx<ushort, int, schar, ushort, op>, device::fmx<ushort2, int2, char2, ushort2, op>, device::fmx<ushort3, int3, char3, ushort3, op>, device::fmx<ushort4, int4, char4, ushort4, op>  },
                    { device::fmx<ushort, int, schar, short, op>, device::fmx<ushort2, int2, char2, short2, op>, device::fmx<ushort3, int3, char3, short3, op>, device::fmx<ushort4, int4, char4, short4, op>  },
                    { device::fmx<ushort, int, schar, int, op>, device::fmx<ushort2, int2, char2, int2, op>, device::fmx<ushort3, int3, char3, int3, op>, device::fmx<ushort4, int4, char4, int4, op>  },
                    { device::fmx<ushort, int, schar, float, op>, device::fmx<ushort2, int2, char2, float2, op>, device::fmx<ushort3, int3, char3, float3, op>, device::fmx<ushort4, int4, char4, float4, op>  },
                    { device::fmx<ushort, int, schar, double, op>, device::fmx<ushort2, int2, char2, double2, op>, device::fmx<ushort3, int3, char3, double3, op>, device::fmx<ushort4, int4, char4, double4, op>  },
                },
                {
                    { device::fmx<ushort, int, ushort, uchar, op>, device::fmx<ushort2, int2, ushort2, uchar2, op>, device::fmx<ushort3, int3, ushort3, uchar3, op>, device::fmx<ushort4, int4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, int, ushort, schar, op>, device::fmx<ushort2, int2, ushort2, char2, op>, device::fmx<ushort3, int3, ushort3, char3, op>, device::fmx<ushort4, int4, ushort4, char4, op>  },
                    { device::fmx<ushort, int, ushort, ushort, op>, device::fmx<ushort2, int2, ushort2, ushort2, op>, device::fmx<ushort3, int3, ushort3, ushort3, op>, device::fmx<ushort4, int4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, int, ushort, short, op>, device::fmx<ushort2, int2, ushort2, short2, op>, device::fmx<ushort3, int3, ushort3, short3, op>, device::fmx<ushort4, int4, ushort4, short4, op>  },
                    { device::fmx<ushort, int, ushort, int, op>, device::fmx<ushort2, int2, ushort2, int2, op>, device::fmx<ushort3, int3, ushort3, int3, op>, device::fmx<ushort4, int4, ushort4, int4, op>  },
                    { device::fmx<ushort, int, ushort, float, op>, device::fmx<ushort2, int2, ushort2, float2, op>, device::fmx<ushort3, int3, ushort3, float3, op>, device::fmx<ushort4, int4, ushort4, float4, op>  },
                    { device::fmx<ushort, int, ushort, double, op>, device::fmx<ushort2, int2, ushort2, double2, op>, device::fmx<ushort3, int3, ushort3, double3, op>, device::fmx<ushort4, int4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<ushort, int, short, uchar, op>, device::fmx<ushort2, int2, short2, uchar2, op>, device::fmx<ushort3, int3, short3, uchar3, op>, device::fmx<ushort4, int4, short4, uchar4, op>  },
                    { device::fmx<ushort, int, short, schar, op>, device::fmx<ushort2, int2, short2, char2, op>, device::fmx<ushort3, int3, short3, char3, op>, device::fmx<ushort4, int4, short4, char4, op>  },
                    { device::fmx<ushort, int, short, ushort, op>, device::fmx<ushort2, int2, short2, ushort2, op>, device::fmx<ushort3, int3, short3, ushort3, op>, device::fmx<ushort4, int4, short4, ushort4, op>  },
                    { device::fmx<ushort, int, short, short, op>, device::fmx<ushort2, int2, short2, short2, op>, device::fmx<ushort3, int3, short3, short3, op>, device::fmx<ushort4, int4, short4, short4, op>  },
                    { device::fmx<ushort, int, short, int, op>, device::fmx<ushort2, int2, short2, int2, op>, device::fmx<ushort3, int3, short3, int3, op>, device::fmx<ushort4, int4, short4, int4, op>  },
                    { device::fmx<ushort, int, short, float, op>, device::fmx<ushort2, int2, short2, float2, op>, device::fmx<ushort3, int3, short3, float3, op>, device::fmx<ushort4, int4, short4, float4, op>  },
                    { device::fmx<ushort, int, short, double, op>, device::fmx<ushort2, int2, short2, double2, op>, device::fmx<ushort3, int3, short3, double3, op>, device::fmx<ushort4, int4, short4, double4, op>  },
                },
                {
                    { device::fmx<ushort, int, int, uchar, op>, device::fmx<ushort2, int2, int2, uchar2, op>, device::fmx<ushort3, int3, int3, uchar3, op>, device::fmx<ushort4, int4, int4, uchar4, op>  },
                    { device::fmx<ushort, int, int, schar, op>, device::fmx<ushort2, int2, int2, char2, op>, device::fmx<ushort3, int3, int3, char3, op>, device::fmx<ushort4, int4, int4, char4, op>  },
                    { device::fmx<ushort, int, int, ushort, op>, device::fmx<ushort2, int2, int2, ushort2, op>, device::fmx<ushort3, int3, int3, ushort3, op>, device::fmx<ushort4, int4, int4, ushort4, op>  },
                    { device::fmx<ushort, int, int, short, op>, device::fmx<ushort2, int2, int2, short2, op>, device::fmx<ushort3, int3, int3, short3, op>, device::fmx<ushort4, int4, int4, short4, op>  },
                    { device::fmx<ushort, int, int, int, op>, device::fmx<ushort2, int2, int2, int2, op>, device::fmx<ushort3, int3, int3, int3, op>, device::fmx<ushort4, int4, int4, int4, op>  },
                    { device::fmx<ushort, int, int, float, op>, device::fmx<ushort2, int2, int2, float2, op>, device::fmx<ushort3, int3, int3, float3, op>, device::fmx<ushort4, int4, int4, float4, op>  },
                    { device::fmx<ushort, int, int, double, op>, device::fmx<ushort2, int2, int2, double2, op>, device::fmx<ushort3, int3, int3, double3, op>, device::fmx<ushort4, int4, int4, double4, op>  },
                },
                {
                    { device::fmx<ushort, int, float, uchar, op>, device::fmx<ushort2, int2, float2, uchar2, op>, device::fmx<ushort3, int3, float3, uchar3, op>, device::fmx<ushort4, int4, float4, uchar4, op>  },
                    { device::fmx<ushort, int, float, schar, op>, device::fmx<ushort2, int2, float2, char2, op>, device::fmx<ushort3, int3, float3, char3, op>, device::fmx<ushort4, int4, float4, char4, op>  },
                    { device::fmx<ushort, int, float, ushort, op>, device::fmx<ushort2, int2, float2, ushort2, op>, device::fmx<ushort3, int3, float3, ushort3, op>, device::fmx<ushort4, int4, float4, ushort4, op>  },
                    { device::fmx<ushort, int, float, short, op>, device::fmx<ushort2, int2, float2, short2, op>, device::fmx<ushort3, int3, float3, short3, op>, device::fmx<ushort4, int4, float4, short4, op>  },
                    { device::fmx<ushort, int, float, int, op>, device::fmx<ushort2, int2, float2, int2, op>, device::fmx<ushort3, int3, float3, int3, op>, device::fmx<ushort4, int4, float4, int4, op>  },
                    { device::fmx<ushort, int, float, float, op>, device::fmx<ushort2, int2, float2, float2, op>, device::fmx<ushort3, int3, float3, float3, op>, device::fmx<ushort4, int4, float4, float4, op>  },
                    { device::fmx<ushort, int, float, double, op>, device::fmx<ushort2, int2, float2, double2, op>, device::fmx<ushort3, int3, float3, double3, op>, device::fmx<ushort4, int4, float4, double4, op>  },
                },
                {
                    { device::fmx<ushort, int, double, uchar, op>, device::fmx<ushort2, int2, double2, uchar2, op>, device::fmx<ushort3, int3, double3, uchar3, op>, device::fmx<ushort4, int4, double4, uchar4, op>  },
                    { device::fmx<ushort, int, double, schar, op>, device::fmx<ushort2, int2, double2, char2, op>, device::fmx<ushort3, int3, double3, char3, op>, device::fmx<ushort4, int4, double4, char4, op>  },
                    { device::fmx<ushort, int, double, ushort, op>, device::fmx<ushort2, int2, double2, ushort2, op>, device::fmx<ushort3, int3, double3, ushort3, op>, device::fmx<ushort4, int4, double4, ushort4, op>  },
                    { device::fmx<ushort, int, double, short, op>, device::fmx<ushort2, int2, double2, short2, op>, device::fmx<ushort3, int3, double3, short3, op>, device::fmx<ushort4, int4, double4, short4, op>  },
                    { device::fmx<ushort, int, double, int, op>, device::fmx<ushort2, int2, double2, int2, op>, device::fmx<ushort3, int3, double3, int3, op>, device::fmx<ushort4, int4, double4, int4, op>  },
                    { device::fmx<ushort, int, double, float, op>, device::fmx<ushort2, int2, double2, float2, op>, device::fmx<ushort3, int3, double3, float3, op>, device::fmx<ushort4, int4, double4, float4, op>  },
                    { device::fmx<ushort, int, double, double, op>, device::fmx<ushort2, int2, double2, double2, op>, device::fmx<ushort3, int3, double3, double3, op>, device::fmx<ushort4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<ushort, float, uchar, uchar, op>, device::fmx<ushort2, float2, uchar2, uchar2, op>, device::fmx<ushort3, float3, uchar3, uchar3, op>, device::fmx<ushort4, float4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, float, uchar, schar, op>, device::fmx<ushort2, float2, uchar2, char2, op>, device::fmx<ushort3, float3, uchar3, char3, op>, device::fmx<ushort4, float4, uchar4, char4, op>  },
                    { device::fmx<ushort, float, uchar, ushort, op>, device::fmx<ushort2, float2, uchar2, ushort2, op>, device::fmx<ushort3, float3, uchar3, ushort3, op>, device::fmx<ushort4, float4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, float, uchar, short, op>, device::fmx<ushort2, float2, uchar2, short2, op>, device::fmx<ushort3, float3, uchar3, short3, op>, device::fmx<ushort4, float4, uchar4, short4, op>  },
                    { device::fmx<ushort, float, uchar, int, op>, device::fmx<ushort2, float2, uchar2, int2, op>, device::fmx<ushort3, float3, uchar3, int3, op>, device::fmx<ushort4, float4, uchar4, int4, op>  },
                    { device::fmx<ushort, float, uchar, float, op>, device::fmx<ushort2, float2, uchar2, float2, op>, device::fmx<ushort3, float3, uchar3, float3, op>, device::fmx<ushort4, float4, uchar4, float4, op>  },
                    { device::fmx<ushort, float, uchar, double, op>, device::fmx<ushort2, float2, uchar2, double2, op>, device::fmx<ushort3, float3, uchar3, double3, op>, device::fmx<ushort4, float4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<ushort, float, schar, uchar, op>, device::fmx<ushort2, float2, char2, uchar2, op>, device::fmx<ushort3, float3, char3, uchar3, op>, device::fmx<ushort4, float4, char4, uchar4, op>  },
                    { device::fmx<ushort, float, schar, schar, op>, device::fmx<ushort2, float2, char2, char2, op>, device::fmx<ushort3, float3, char3, char3, op>, device::fmx<ushort4, float4, char4, char4, op>  },
                    { device::fmx<ushort, float, schar, ushort, op>, device::fmx<ushort2, float2, char2, ushort2, op>, device::fmx<ushort3, float3, char3, ushort3, op>, device::fmx<ushort4, float4, char4, ushort4, op>  },
                    { device::fmx<ushort, float, schar, short, op>, device::fmx<ushort2, float2, char2, short2, op>, device::fmx<ushort3, float3, char3, short3, op>, device::fmx<ushort4, float4, char4, short4, op>  },
                    { device::fmx<ushort, float, schar, int, op>, device::fmx<ushort2, float2, char2, int2, op>, device::fmx<ushort3, float3, char3, int3, op>, device::fmx<ushort4, float4, char4, int4, op>  },
                    { device::fmx<ushort, float, schar, float, op>, device::fmx<ushort2, float2, char2, float2, op>, device::fmx<ushort3, float3, char3, float3, op>, device::fmx<ushort4, float4, char4, float4, op>  },
                    { device::fmx<ushort, float, schar, double, op>, device::fmx<ushort2, float2, char2, double2, op>, device::fmx<ushort3, float3, char3, double3, op>, device::fmx<ushort4, float4, char4, double4, op>  },
                },
                {
                    { device::fmx<ushort, float, ushort, uchar, op>, device::fmx<ushort2, float2, ushort2, uchar2, op>, device::fmx<ushort3, float3, ushort3, uchar3, op>, device::fmx<ushort4, float4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, float, ushort, schar, op>, device::fmx<ushort2, float2, ushort2, char2, op>, device::fmx<ushort3, float3, ushort3, char3, op>, device::fmx<ushort4, float4, ushort4, char4, op>  },
                    { device::fmx<ushort, float, ushort, ushort, op>, device::fmx<ushort2, float2, ushort2, ushort2, op>, device::fmx<ushort3, float3, ushort3, ushort3, op>, device::fmx<ushort4, float4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, float, ushort, short, op>, device::fmx<ushort2, float2, ushort2, short2, op>, device::fmx<ushort3, float3, ushort3, short3, op>, device::fmx<ushort4, float4, ushort4, short4, op>  },
                    { device::fmx<ushort, float, ushort, int, op>, device::fmx<ushort2, float2, ushort2, int2, op>, device::fmx<ushort3, float3, ushort3, int3, op>, device::fmx<ushort4, float4, ushort4, int4, op>  },
                    { device::fmx<ushort, float, ushort, float, op>, device::fmx<ushort2, float2, ushort2, float2, op>, device::fmx<ushort3, float3, ushort3, float3, op>, device::fmx<ushort4, float4, ushort4, float4, op>  },
                    { device::fmx<ushort, float, ushort, double, op>, device::fmx<ushort2, float2, ushort2, double2, op>, device::fmx<ushort3, float3, ushort3, double3, op>, device::fmx<ushort4, float4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<ushort, float, short, uchar, op>, device::fmx<ushort2, float2, short2, uchar2, op>, device::fmx<ushort3, float3, short3, uchar3, op>, device::fmx<ushort4, float4, short4, uchar4, op>  },
                    { device::fmx<ushort, float, short, schar, op>, device::fmx<ushort2, float2, short2, char2, op>, device::fmx<ushort3, float3, short3, char3, op>, device::fmx<ushort4, float4, short4, char4, op>  },
                    { device::fmx<ushort, float, short, ushort, op>, device::fmx<ushort2, float2, short2, ushort2, op>, device::fmx<ushort3, float3, short3, ushort3, op>, device::fmx<ushort4, float4, short4, ushort4, op>  },
                    { device::fmx<ushort, float, short, short, op>, device::fmx<ushort2, float2, short2, short2, op>, device::fmx<ushort3, float3, short3, short3, op>, device::fmx<ushort4, float4, short4, short4, op>  },
                    { device::fmx<ushort, float, short, int, op>, device::fmx<ushort2, float2, short2, int2, op>, device::fmx<ushort3, float3, short3, int3, op>, device::fmx<ushort4, float4, short4, int4, op>  },
                    { device::fmx<ushort, float, short, float, op>, device::fmx<ushort2, float2, short2, float2, op>, device::fmx<ushort3, float3, short3, float3, op>, device::fmx<ushort4, float4, short4, float4, op>  },
                    { device::fmx<ushort, float, short, double, op>, device::fmx<ushort2, float2, short2, double2, op>, device::fmx<ushort3, float3, short3, double3, op>, device::fmx<ushort4, float4, short4, double4, op>  },
                },
                {
                    { device::fmx<ushort, float, int, uchar, op>, device::fmx<ushort2, float2, int2, uchar2, op>, device::fmx<ushort3, float3, int3, uchar3, op>, device::fmx<ushort4, float4, int4, uchar4, op>  },
                    { device::fmx<ushort, float, int, schar, op>, device::fmx<ushort2, float2, int2, char2, op>, device::fmx<ushort3, float3, int3, char3, op>, device::fmx<ushort4, float4, int4, char4, op>  },
                    { device::fmx<ushort, float, int, ushort, op>, device::fmx<ushort2, float2, int2, ushort2, op>, device::fmx<ushort3, float3, int3, ushort3, op>, device::fmx<ushort4, float4, int4, ushort4, op>  },
                    { device::fmx<ushort, float, int, short, op>, device::fmx<ushort2, float2, int2, short2, op>, device::fmx<ushort3, float3, int3, short3, op>, device::fmx<ushort4, float4, int4, short4, op>  },
                    { device::fmx<ushort, float, int, int, op>, device::fmx<ushort2, float2, int2, int2, op>, device::fmx<ushort3, float3, int3, int3, op>, device::fmx<ushort4, float4, int4, int4, op>  },
                    { device::fmx<ushort, float, int, float, op>, device::fmx<ushort2, float2, int2, float2, op>, device::fmx<ushort3, float3, int3, float3, op>, device::fmx<ushort4, float4, int4, float4, op>  },
                    { device::fmx<ushort, float, int, double, op>, device::fmx<ushort2, float2, int2, double2, op>, device::fmx<ushort3, float3, int3, double3, op>, device::fmx<ushort4, float4, int4, double4, op>  },
                },
                {
                    { device::fmx<ushort, float, float, uchar, op>, device::fmx<ushort2, float2, float2, uchar2, op>, device::fmx<ushort3, float3, float3, uchar3, op>, device::fmx<ushort4, float4, float4, uchar4, op>  },
                    { device::fmx<ushort, float, float, schar, op>, device::fmx<ushort2, float2, float2, char2, op>, device::fmx<ushort3, float3, float3, char3, op>, device::fmx<ushort4, float4, float4, char4, op>  },
                    { device::fmx<ushort, float, float, ushort, op>, device::fmx<ushort2, float2, float2, ushort2, op>, device::fmx<ushort3, float3, float3, ushort3, op>, device::fmx<ushort4, float4, float4, ushort4, op>  },
                    { device::fmx<ushort, float, float, short, op>, device::fmx<ushort2, float2, float2, short2, op>, device::fmx<ushort3, float3, float3, short3, op>, device::fmx<ushort4, float4, float4, short4, op>  },
                    { device::fmx<ushort, float, float, int, op>, device::fmx<ushort2, float2, float2, int2, op>, device::fmx<ushort3, float3, float3, int3, op>, device::fmx<ushort4, float4, float4, int4, op>  },
                    { device::fmx<ushort, float, float, float, op>, device::fmx<ushort2, float2, float2, float2, op>, device::fmx<ushort3, float3, float3, float3, op>, device::fmx<ushort4, float4, float4, float4, op>  },
                    { device::fmx<ushort, float, float, double, op>, device::fmx<ushort2, float2, float2, double2, op>, device::fmx<ushort3, float3, float3, double3, op>, device::fmx<ushort4, float4, float4, double4, op>  },
                },
                {
                    { device::fmx<ushort, float, double, uchar, op>, device::fmx<ushort2, float2, double2, uchar2, op>, device::fmx<ushort3, float3, double3, uchar3, op>, device::fmx<ushort4, float4, double4, uchar4, op>  },
                    { device::fmx<ushort, float, double, schar, op>, device::fmx<ushort2, float2, double2, char2, op>, device::fmx<ushort3, float3, double3, char3, op>, device::fmx<ushort4, float4, double4, char4, op>  },
                    { device::fmx<ushort, float, double, ushort, op>, device::fmx<ushort2, float2, double2, ushort2, op>, device::fmx<ushort3, float3, double3, ushort3, op>, device::fmx<ushort4, float4, double4, ushort4, op>  },
                    { device::fmx<ushort, float, double, short, op>, device::fmx<ushort2, float2, double2, short2, op>, device::fmx<ushort3, float3, double3, short3, op>, device::fmx<ushort4, float4, double4, short4, op>  },
                    { device::fmx<ushort, float, double, int, op>, device::fmx<ushort2, float2, double2, int2, op>, device::fmx<ushort3, float3, double3, int3, op>, device::fmx<ushort4, float4, double4, int4, op>  },
                    { device::fmx<ushort, float, double, float, op>, device::fmx<ushort2, float2, double2, float2, op>, device::fmx<ushort3, float3, double3, float3, op>, device::fmx<ushort4, float4, double4, float4, op>  },
                    { device::fmx<ushort, float, double, double, op>, device::fmx<ushort2, float2, double2, double2, op>, device::fmx<ushort3, float3, double3, double3, op>, device::fmx<ushort4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<ushort, double, uchar, uchar, op>, device::fmx<ushort2, double2, uchar2, uchar2, op>, device::fmx<ushort3, double3, uchar3, uchar3, op>, device::fmx<ushort4, double4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, double, uchar, schar, op>, device::fmx<ushort2, double2, uchar2, char2, op>, device::fmx<ushort3, double3, uchar3, char3, op>, device::fmx<ushort4, double4, uchar4, char4, op>  },
                    { device::fmx<ushort, double, uchar, ushort, op>, device::fmx<ushort2, double2, uchar2, ushort2, op>, device::fmx<ushort3, double3, uchar3, ushort3, op>, device::fmx<ushort4, double4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, double, uchar, short, op>, device::fmx<ushort2, double2, uchar2, short2, op>, device::fmx<ushort3, double3, uchar3, short3, op>, device::fmx<ushort4, double4, uchar4, short4, op>  },
                    { device::fmx<ushort, double, uchar, int, op>, device::fmx<ushort2, double2, uchar2, int2, op>, device::fmx<ushort3, double3, uchar3, int3, op>, device::fmx<ushort4, double4, uchar4, int4, op>  },
                    { device::fmx<ushort, double, uchar, float, op>, device::fmx<ushort2, double2, uchar2, float2, op>, device::fmx<ushort3, double3, uchar3, float3, op>, device::fmx<ushort4, double4, uchar4, float4, op>  },
                    { device::fmx<ushort, double, uchar, double, op>, device::fmx<ushort2, double2, uchar2, double2, op>, device::fmx<ushort3, double3, uchar3, double3, op>, device::fmx<ushort4, double4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<ushort, double, schar, uchar, op>, device::fmx<ushort2, double2, char2, uchar2, op>, device::fmx<ushort3, double3, char3, uchar3, op>, device::fmx<ushort4, double4, char4, uchar4, op>  },
                    { device::fmx<ushort, double, schar, schar, op>, device::fmx<ushort2, double2, char2, char2, op>, device::fmx<ushort3, double3, char3, char3, op>, device::fmx<ushort4, double4, char4, char4, op>  },
                    { device::fmx<ushort, double, schar, ushort, op>, device::fmx<ushort2, double2, char2, ushort2, op>, device::fmx<ushort3, double3, char3, ushort3, op>, device::fmx<ushort4, double4, char4, ushort4, op>  },
                    { device::fmx<ushort, double, schar, short, op>, device::fmx<ushort2, double2, char2, short2, op>, device::fmx<ushort3, double3, char3, short3, op>, device::fmx<ushort4, double4, char4, short4, op>  },
                    { device::fmx<ushort, double, schar, int, op>, device::fmx<ushort2, double2, char2, int2, op>, device::fmx<ushort3, double3, char3, int3, op>, device::fmx<ushort4, double4, char4, int4, op>  },
                    { device::fmx<ushort, double, schar, float, op>, device::fmx<ushort2, double2, char2, float2, op>, device::fmx<ushort3, double3, char3, float3, op>, device::fmx<ushort4, double4, char4, float4, op>  },
                    { device::fmx<ushort, double, schar, double, op>, device::fmx<ushort2, double2, char2, double2, op>, device::fmx<ushort3, double3, char3, double3, op>, device::fmx<ushort4, double4, char4, double4, op>  },
                },
                {
                    { device::fmx<ushort, double, ushort, uchar, op>, device::fmx<ushort2, double2, ushort2, uchar2, op>, device::fmx<ushort3, double3, ushort3, uchar3, op>, device::fmx<ushort4, double4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, double, ushort, schar, op>, device::fmx<ushort2, double2, ushort2, char2, op>, device::fmx<ushort3, double3, ushort3, char3, op>, device::fmx<ushort4, double4, ushort4, char4, op>  },
                    { device::fmx<ushort, double, ushort, ushort, op>, device::fmx<ushort2, double2, ushort2, ushort2, op>, device::fmx<ushort3, double3, ushort3, ushort3, op>, device::fmx<ushort4, double4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, double, ushort, short, op>, device::fmx<ushort2, double2, ushort2, short2, op>, device::fmx<ushort3, double3, ushort3, short3, op>, device::fmx<ushort4, double4, ushort4, short4, op>  },
                    { device::fmx<ushort, double, ushort, int, op>, device::fmx<ushort2, double2, ushort2, int2, op>, device::fmx<ushort3, double3, ushort3, int3, op>, device::fmx<ushort4, double4, ushort4, int4, op>  },
                    { device::fmx<ushort, double, ushort, float, op>, device::fmx<ushort2, double2, ushort2, float2, op>, device::fmx<ushort3, double3, ushort3, float3, op>, device::fmx<ushort4, double4, ushort4, float4, op>  },
                    { device::fmx<ushort, double, ushort, double, op>, device::fmx<ushort2, double2, ushort2, double2, op>, device::fmx<ushort3, double3, ushort3, double3, op>, device::fmx<ushort4, double4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<ushort, double, short, uchar, op>, device::fmx<ushort2, double2, short2, uchar2, op>, device::fmx<ushort3, double3, short3, uchar3, op>, device::fmx<ushort4, double4, short4, uchar4, op>  },
                    { device::fmx<ushort, double, short, schar, op>, device::fmx<ushort2, double2, short2, char2, op>, device::fmx<ushort3, double3, short3, char3, op>, device::fmx<ushort4, double4, short4, char4, op>  },
                    { device::fmx<ushort, double, short, ushort, op>, device::fmx<ushort2, double2, short2, ushort2, op>, device::fmx<ushort3, double3, short3, ushort3, op>, device::fmx<ushort4, double4, short4, ushort4, op>  },
                    { device::fmx<ushort, double, short, short, op>, device::fmx<ushort2, double2, short2, short2, op>, device::fmx<ushort3, double3, short3, short3, op>, device::fmx<ushort4, double4, short4, short4, op>  },
                    { device::fmx<ushort, double, short, int, op>, device::fmx<ushort2, double2, short2, int2, op>, device::fmx<ushort3, double3, short3, int3, op>, device::fmx<ushort4, double4, short4, int4, op>  },
                    { device::fmx<ushort, double, short, float, op>, device::fmx<ushort2, double2, short2, float2, op>, device::fmx<ushort3, double3, short3, float3, op>, device::fmx<ushort4, double4, short4, float4, op>  },
                    { device::fmx<ushort, double, short, double, op>, device::fmx<ushort2, double2, short2, double2, op>, device::fmx<ushort3, double3, short3, double3, op>, device::fmx<ushort4, double4, short4, double4, op>  },
                },
                {
                    { device::fmx<ushort, double, int, uchar, op>, device::fmx<ushort2, double2, int2, uchar2, op>, device::fmx<ushort3, double3, int3, uchar3, op>, device::fmx<ushort4, double4, int4, uchar4, op>  },
                    { device::fmx<ushort, double, int, schar, op>, device::fmx<ushort2, double2, int2, char2, op>, device::fmx<ushort3, double3, int3, char3, op>, device::fmx<ushort4, double4, int4, char4, op>  },
                    { device::fmx<ushort, double, int, ushort, op>, device::fmx<ushort2, double2, int2, ushort2, op>, device::fmx<ushort3, double3, int3, ushort3, op>, device::fmx<ushort4, double4, int4, ushort4, op>  },
                    { device::fmx<ushort, double, int, short, op>, device::fmx<ushort2, double2, int2, short2, op>, device::fmx<ushort3, double3, int3, short3, op>, device::fmx<ushort4, double4, int4, short4, op>  },
                    { device::fmx<ushort, double, int, int, op>, device::fmx<ushort2, double2, int2, int2, op>, device::fmx<ushort3, double3, int3, int3, op>, device::fmx<ushort4, double4, int4, int4, op>  },
                    { device::fmx<ushort, double, int, float, op>, device::fmx<ushort2, double2, int2, float2, op>, device::fmx<ushort3, double3, int3, float3, op>, device::fmx<ushort4, double4, int4, float4, op>  },
                    { device::fmx<ushort, double, int, double, op>, device::fmx<ushort2, double2, int2, double2, op>, device::fmx<ushort3, double3, int3, double3, op>, device::fmx<ushort4, double4, int4, double4, op>  },
                },
                {
                    { device::fmx<ushort, double, float, uchar, op>, device::fmx<ushort2, double2, float2, uchar2, op>, device::fmx<ushort3, double3, float3, uchar3, op>, device::fmx<ushort4, double4, float4, uchar4, op>  },
                    { device::fmx<ushort, double, float, schar, op>, device::fmx<ushort2, double2, float2, char2, op>, device::fmx<ushort3, double3, float3, char3, op>, device::fmx<ushort4, double4, float4, char4, op>  },
                    { device::fmx<ushort, double, float, ushort, op>, device::fmx<ushort2, double2, float2, ushort2, op>, device::fmx<ushort3, double3, float3, ushort3, op>, device::fmx<ushort4, double4, float4, ushort4, op>  },
                    { device::fmx<ushort, double, float, short, op>, device::fmx<ushort2, double2, float2, short2, op>, device::fmx<ushort3, double3, float3, short3, op>, device::fmx<ushort4, double4, float4, short4, op>  },
                    { device::fmx<ushort, double, float, int, op>, device::fmx<ushort2, double2, float2, int2, op>, device::fmx<ushort3, double3, float3, int3, op>, device::fmx<ushort4, double4, float4, int4, op>  },
                    { device::fmx<ushort, double, float, float, op>, device::fmx<ushort2, double2, float2, float2, op>, device::fmx<ushort3, double3, float3, float3, op>, device::fmx<ushort4, double4, float4, float4, op>  },
                    { device::fmx<ushort, double, float, double, op>, device::fmx<ushort2, double2, float2, double2, op>, device::fmx<ushort3, double3, float3, double3, op>, device::fmx<ushort4, double4, float4, double4, op>  },
                },
                {
                    { device::fmx<ushort, double, double, uchar, op>, device::fmx<ushort2, double2, double2, uchar2, op>, device::fmx<ushort3, double3, double3, uchar3, op>, device::fmx<ushort4, double4, double4, uchar4, op>  },
                    { device::fmx<ushort, double, double, schar, op>, device::fmx<ushort2, double2, double2, char2, op>, device::fmx<ushort3, double3, double3, char3, op>, device::fmx<ushort4, double4, double4, char4, op>  },
                    { device::fmx<ushort, double, double, ushort, op>, device::fmx<ushort2, double2, double2, ushort2, op>, device::fmx<ushort3, double3, double3, ushort3, op>, device::fmx<ushort4, double4, double4, ushort4, op>  },
                    { device::fmx<ushort, double, double, short, op>, device::fmx<ushort2, double2, double2, short2, op>, device::fmx<ushort3, double3, double3, short3, op>, device::fmx<ushort4, double4, double4, short4, op>  },
                    { device::fmx<ushort, double, double, int, op>, device::fmx<ushort2, double2, double2, int2, op>, device::fmx<ushort3, double3, double3, int3, op>, device::fmx<ushort4, double4, double4, int4, op>  },
                    { device::fmx<ushort, double, double, float, op>, device::fmx<ushort2, double2, double2, float2, op>, device::fmx<ushort3, double3, double3, float3, op>, device::fmx<ushort4, double4, double4, float4, op>  },
                    { device::fmx<ushort, double, double, double, op>, device::fmx<ushort2, double2, double2, double2, op>, device::fmx<ushort3, double3, double3, double3, op>, device::fmx<ushort4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { device::fmx<short, uchar, uchar, uchar, op>, device::fmx<short2, uchar2, uchar2, uchar2, op>, device::fmx<short3, uchar3, uchar3, uchar3, op>, device::fmx<short4, uchar4, uchar4, uchar4, op>  },
                    { device::fmx<short, uchar, uchar, schar, op>, device::fmx<short2, uchar2, uchar2, char2, op>, device::fmx<short3, uchar3, uchar3, char3, op>, device::fmx<short4, uchar4, uchar4, char4, op>  },
                    { device::fmx<short, uchar, uchar, ushort, op>, device::fmx<short2, uchar2, uchar2, ushort2, op>, device::fmx<short3, uchar3, uchar3, ushort3, op>, device::fmx<short4, uchar4, uchar4, ushort4, op>  },
                    { device::fmx<short, uchar, uchar, short, op>, device::fmx<short2, uchar2, uchar2, short2, op>, device::fmx<short3, uchar3, uchar3, short3, op>, device::fmx<short4, uchar4, uchar4, short4, op>  },
                    { device::fmx<short, uchar, uchar, int, op>, device::fmx<short2, uchar2, uchar2, int2, op>, device::fmx<short3, uchar3, uchar3, int3, op>, device::fmx<short4, uchar4, uchar4, int4, op>  },
                    { device::fmx<short, uchar, uchar, float, op>, device::fmx<short2, uchar2, uchar2, float2, op>, device::fmx<short3, uchar3, uchar3, float3, op>, device::fmx<short4, uchar4, uchar4, float4, op>  },
                    { device::fmx<short, uchar, uchar, double, op>, device::fmx<short2, uchar2, uchar2, double2, op>, device::fmx<short3, uchar3, uchar3, double3, op>, device::fmx<short4, uchar4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<short, uchar, schar, uchar, op>, device::fmx<short2, uchar2, char2, uchar2, op>, device::fmx<short3, uchar3, char3, uchar3, op>, device::fmx<short4, uchar4, char4, uchar4, op>  },
                    { device::fmx<short, uchar, schar, schar, op>, device::fmx<short2, uchar2, char2, char2, op>, device::fmx<short3, uchar3, char3, char3, op>, device::fmx<short4, uchar4, char4, char4, op>  },
                    { device::fmx<short, uchar, schar, ushort, op>, device::fmx<short2, uchar2, char2, ushort2, op>, device::fmx<short3, uchar3, char3, ushort3, op>, device::fmx<short4, uchar4, char4, ushort4, op>  },
                    { device::fmx<short, uchar, schar, short, op>, device::fmx<short2, uchar2, char2, short2, op>, device::fmx<short3, uchar3, char3, short3, op>, device::fmx<short4, uchar4, char4, short4, op>  },
                    { device::fmx<short, uchar, schar, int, op>, device::fmx<short2, uchar2, char2, int2, op>, device::fmx<short3, uchar3, char3, int3, op>, device::fmx<short4, uchar4, char4, int4, op>  },
                    { device::fmx<short, uchar, schar, float, op>, device::fmx<short2, uchar2, char2, float2, op>, device::fmx<short3, uchar3, char3, float3, op>, device::fmx<short4, uchar4, char4, float4, op>  },
                    { device::fmx<short, uchar, schar, double, op>, device::fmx<short2, uchar2, char2, double2, op>, device::fmx<short3, uchar3, char3, double3, op>, device::fmx<short4, uchar4, char4, double4, op>  },
                },
                {
                    { device::fmx<short, uchar, ushort, uchar, op>, device::fmx<short2, uchar2, ushort2, uchar2, op>, device::fmx<short3, uchar3, ushort3, uchar3, op>, device::fmx<short4, uchar4, ushort4, uchar4, op>  },
                    { device::fmx<short, uchar, ushort, schar, op>, device::fmx<short2, uchar2, ushort2, char2, op>, device::fmx<short3, uchar3, ushort3, char3, op>, device::fmx<short4, uchar4, ushort4, char4, op>  },
                    { device::fmx<short, uchar, ushort, ushort, op>, device::fmx<short2, uchar2, ushort2, ushort2, op>, device::fmx<short3, uchar3, ushort3, ushort3, op>, device::fmx<short4, uchar4, ushort4, ushort4, op>  },
                    { device::fmx<short, uchar, ushort, short, op>, device::fmx<short2, uchar2, ushort2, short2, op>, device::fmx<short3, uchar3, ushort3, short3, op>, device::fmx<short4, uchar4, ushort4, short4, op>  },
                    { device::fmx<short, uchar, ushort, int, op>, device::fmx<short2, uchar2, ushort2, int2, op>, device::fmx<short3, uchar3, ushort3, int3, op>, device::fmx<short4, uchar4, ushort4, int4, op>  },
                    { device::fmx<short, uchar, ushort, float, op>, device::fmx<short2, uchar2, ushort2, float2, op>, device::fmx<short3, uchar3, ushort3, float3, op>, device::fmx<short4, uchar4, ushort4, float4, op>  },
                    { device::fmx<short, uchar, ushort, double, op>, device::fmx<short2, uchar2, ushort2, double2, op>, device::fmx<short3, uchar3, ushort3, double3, op>, device::fmx<short4, uchar4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<short, uchar, short, uchar, op>, device::fmx<short2, uchar2, short2, uchar2, op>, device::fmx<short3, uchar3, short3, uchar3, op>, device::fmx<short4, uchar4, short4, uchar4, op>  },
                    { device::fmx<short, uchar, short, schar, op>, device::fmx<short2, uchar2, short2, char2, op>, device::fmx<short3, uchar3, short3, char3, op>, device::fmx<short4, uchar4, short4, char4, op>  },
                    { device::fmx<short, uchar, short, ushort, op>, device::fmx<short2, uchar2, short2, ushort2, op>, device::fmx<short3, uchar3, short3, ushort3, op>, device::fmx<short4, uchar4, short4, ushort4, op>  },
                    { device::fmx<short, uchar, short, short, op>, device::fmx<short2, uchar2, short2, short2, op>, device::fmx<short3, uchar3, short3, short3, op>, device::fmx<short4, uchar4, short4, short4, op>  },
                    { device::fmx<short, uchar, short, int, op>, device::fmx<short2, uchar2, short2, int2, op>, device::fmx<short3, uchar3, short3, int3, op>, device::fmx<short4, uchar4, short4, int4, op>  },
                    { device::fmx<short, uchar, short, float, op>, device::fmx<short2, uchar2, short2, float2, op>, device::fmx<short3, uchar3, short3, float3, op>, device::fmx<short4, uchar4, short4, float4, op>  },
                    { device::fmx<short, uchar, short, double, op>, device::fmx<short2, uchar2, short2, double2, op>, device::fmx<short3, uchar3, short3, double3, op>, device::fmx<short4, uchar4, short4, double4, op>  },
                },
                {
                    { device::fmx<short, uchar, int, uchar, op>, device::fmx<short2, uchar2, int2, uchar2, op>, device::fmx<short3, uchar3, int3, uchar3, op>, device::fmx<short4, uchar4, int4, uchar4, op>  },
                    { device::fmx<short, uchar, int, schar, op>, device::fmx<short2, uchar2, int2, char2, op>, device::fmx<short3, uchar3, int3, char3, op>, device::fmx<short4, uchar4, int4, char4, op>  },
                    { device::fmx<short, uchar, int, ushort, op>, device::fmx<short2, uchar2, int2, ushort2, op>, device::fmx<short3, uchar3, int3, ushort3, op>, device::fmx<short4, uchar4, int4, ushort4, op>  },
                    { device::fmx<short, uchar, int, short, op>, device::fmx<short2, uchar2, int2, short2, op>, device::fmx<short3, uchar3, int3, short3, op>, device::fmx<short4, uchar4, int4, short4, op>  },
                    { device::fmx<short, uchar, int, int, op>, device::fmx<short2, uchar2, int2, int2, op>, device::fmx<short3, uchar3, int3, int3, op>, device::fmx<short4, uchar4, int4, int4, op>  },
                    { device::fmx<short, uchar, int, float, op>, device::fmx<short2, uchar2, int2, float2, op>, device::fmx<short3, uchar3, int3, float3, op>, device::fmx<short4, uchar4, int4, float4, op>  },
                    { device::fmx<short, uchar, int, double, op>, device::fmx<short2, uchar2, int2, double2, op>, device::fmx<short3, uchar3, int3, double3, op>, device::fmx<short4, uchar4, int4, double4, op>  },
                },
                {
                    { device::fmx<short, uchar, float, uchar, op>, device::fmx<short2, uchar2, float2, uchar2, op>, device::fmx<short3, uchar3, float3, uchar3, op>, device::fmx<short4, uchar4, float4, uchar4, op>  },
                    { device::fmx<short, uchar, float, schar, op>, device::fmx<short2, uchar2, float2, char2, op>, device::fmx<short3, uchar3, float3, char3, op>, device::fmx<short4, uchar4, float4, char4, op>  },
                    { device::fmx<short, uchar, float, ushort, op>, device::fmx<short2, uchar2, float2, ushort2, op>, device::fmx<short3, uchar3, float3, ushort3, op>, device::fmx<short4, uchar4, float4, ushort4, op>  },
                    { device::fmx<short, uchar, float, short, op>, device::fmx<short2, uchar2, float2, short2, op>, device::fmx<short3, uchar3, float3, short3, op>, device::fmx<short4, uchar4, float4, short4, op>  },
                    { device::fmx<short, uchar, float, int, op>, device::fmx<short2, uchar2, float2, int2, op>, device::fmx<short3, uchar3, float3, int3, op>, device::fmx<short4, uchar4, float4, int4, op>  },
                    { device::fmx<short, uchar, float, float, op>, device::fmx<short2, uchar2, float2, float2, op>, device::fmx<short3, uchar3, float3, float3, op>, device::fmx<short4, uchar4, float4, float4, op>  },
                    { device::fmx<short, uchar, float, double, op>, device::fmx<short2, uchar2, float2, double2, op>, device::fmx<short3, uchar3, float3, double3, op>, device::fmx<short4, uchar4, float4, double4, op>  },
                },
                {
                    { device::fmx<short, uchar, double, uchar, op>, device::fmx<short2, uchar2, double2, uchar2, op>, device::fmx<short3, uchar3, double3, uchar3, op>, device::fmx<short4, uchar4, double4, uchar4, op>  },
                    { device::fmx<short, uchar, double, schar, op>, device::fmx<short2, uchar2, double2, char2, op>, device::fmx<short3, uchar3, double3, char3, op>, device::fmx<short4, uchar4, double4, char4, op>  },
                    { device::fmx<short, uchar, double, ushort, op>, device::fmx<short2, uchar2, double2, ushort2, op>, device::fmx<short3, uchar3, double3, ushort3, op>, device::fmx<short4, uchar4, double4, ushort4, op>  },
                    { device::fmx<short, uchar, double, short, op>, device::fmx<short2, uchar2, double2, short2, op>, device::fmx<short3, uchar3, double3, short3, op>, device::fmx<short4, uchar4, double4, short4, op>  },
                    { device::fmx<short, uchar, double, int, op>, device::fmx<short2, uchar2, double2, int2, op>, device::fmx<short3, uchar3, double3, int3, op>, device::fmx<short4, uchar4, double4, int4, op>  },
                    { device::fmx<short, uchar, double, float, op>, device::fmx<short2, uchar2, double2, float2, op>, device::fmx<short3, uchar3, double3, float3, op>, device::fmx<short4, uchar4, double4, float4, op>  },
                    { device::fmx<short, uchar, double, double, op>, device::fmx<short2, uchar2, double2, double2, op>, device::fmx<short3, uchar3, double3, double3, op>, device::fmx<short4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<short, schar, uchar, uchar, op>, device::fmx<short2, char2, uchar2, uchar2, op>, device::fmx<short3, char3, uchar3, uchar3, op>, device::fmx<short4, char4, uchar4, uchar4, op>  },
                    { device::fmx<short, schar, uchar, schar, op>, device::fmx<short2, char2, uchar2, char2, op>, device::fmx<short3, char3, uchar3, char3, op>, device::fmx<short4, char4, uchar4, char4, op>  },
                    { device::fmx<short, schar, uchar, ushort, op>, device::fmx<short2, char2, uchar2, ushort2, op>, device::fmx<short3, char3, uchar3, ushort3, op>, device::fmx<short4, char4, uchar4, ushort4, op>  },
                    { device::fmx<short, schar, uchar, short, op>, device::fmx<short2, char2, uchar2, short2, op>, device::fmx<short3, char3, uchar3, short3, op>, device::fmx<short4, char4, uchar4, short4, op>  },
                    { device::fmx<short, schar, uchar, int, op>, device::fmx<short2, char2, uchar2, int2, op>, device::fmx<short3, char3, uchar3, int3, op>, device::fmx<short4, char4, uchar4, int4, op>  },
                    { device::fmx<short, schar, uchar, float, op>, device::fmx<short2, char2, uchar2, float2, op>, device::fmx<short3, char3, uchar3, float3, op>, device::fmx<short4, char4, uchar4, float4, op>  },
                    { device::fmx<short, schar, uchar, double, op>, device::fmx<short2, char2, uchar2, double2, op>, device::fmx<short3, char3, uchar3, double3, op>, device::fmx<short4, char4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<short, schar, schar, uchar, op>, device::fmx<short2, char2, char2, uchar2, op>, device::fmx<short3, char3, char3, uchar3, op>, device::fmx<short4, char4, char4, uchar4, op>  },
                    { device::fmx<short, schar, schar, schar, op>, device::fmx<short2, char2, char2, char2, op>, device::fmx<short3, char3, char3, char3, op>, device::fmx<short4, char4, char4, char4, op>  },
                    { device::fmx<short, schar, schar, ushort, op>, device::fmx<short2, char2, char2, ushort2, op>, device::fmx<short3, char3, char3, ushort3, op>, device::fmx<short4, char4, char4, ushort4, op>  },
                    { device::fmx<short, schar, schar, short, op>, device::fmx<short2, char2, char2, short2, op>, device::fmx<short3, char3, char3, short3, op>, device::fmx<short4, char4, char4, short4, op>  },
                    { device::fmx<short, schar, schar, int, op>, device::fmx<short2, char2, char2, int2, op>, device::fmx<short3, char3, char3, int3, op>, device::fmx<short4, char4, char4, int4, op>  },
                    { device::fmx<short, schar, schar, float, op>, device::fmx<short2, char2, char2, float2, op>, device::fmx<short3, char3, char3, float3, op>, device::fmx<short4, char4, char4, float4, op>  },
                    { device::fmx<short, schar, schar, double, op>, device::fmx<short2, char2, char2, double2, op>, device::fmx<short3, char3, char3, double3, op>, device::fmx<short4, char4, char4, double4, op>  },
                },
                {
                    { device::fmx<short, schar, ushort, uchar, op>, device::fmx<short2, char2, ushort2, uchar2, op>, device::fmx<short3, char3, ushort3, uchar3, op>, device::fmx<short4, char4, ushort4, uchar4, op>  },
                    { device::fmx<short, schar, ushort, schar, op>, device::fmx<short2, char2, ushort2, char2, op>, device::fmx<short3, char3, ushort3, char3, op>, device::fmx<short4, char4, ushort4, char4, op>  },
                    { device::fmx<short, schar, ushort, ushort, op>, device::fmx<short2, char2, ushort2, ushort2, op>, device::fmx<short3, char3, ushort3, ushort3, op>, device::fmx<short4, char4, ushort4, ushort4, op>  },
                    { device::fmx<short, schar, ushort, short, op>, device::fmx<short2, char2, ushort2, short2, op>, device::fmx<short3, char3, ushort3, short3, op>, device::fmx<short4, char4, ushort4, short4, op>  },
                    { device::fmx<short, schar, ushort, int, op>, device::fmx<short2, char2, ushort2, int2, op>, device::fmx<short3, char3, ushort3, int3, op>, device::fmx<short4, char4, ushort4, int4, op>  },
                    { device::fmx<short, schar, ushort, float, op>, device::fmx<short2, char2, ushort2, float2, op>, device::fmx<short3, char3, ushort3, float3, op>, device::fmx<short4, char4, ushort4, float4, op>  },
                    { device::fmx<short, schar, ushort, double, op>, device::fmx<short2, char2, ushort2, double2, op>, device::fmx<short3, char3, ushort3, double3, op>, device::fmx<short4, char4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<short, schar, short, uchar, op>, device::fmx<short2, char2, short2, uchar2, op>, device::fmx<short3, char3, short3, uchar3, op>, device::fmx<short4, char4, short4, uchar4, op>  },
                    { device::fmx<short, schar, short, schar, op>, device::fmx<short2, char2, short2, char2, op>, device::fmx<short3, char3, short3, char3, op>, device::fmx<short4, char4, short4, char4, op>  },
                    { device::fmx<short, schar, short, ushort, op>, device::fmx<short2, char2, short2, ushort2, op>, device::fmx<short3, char3, short3, ushort3, op>, device::fmx<short4, char4, short4, ushort4, op>  },
                    { device::fmx<short, schar, short, short, op>, device::fmx<short2, char2, short2, short2, op>, device::fmx<short3, char3, short3, short3, op>, device::fmx<short4, char4, short4, short4, op>  },
                    { device::fmx<short, schar, short, int, op>, device::fmx<short2, char2, short2, int2, op>, device::fmx<short3, char3, short3, int3, op>, device::fmx<short4, char4, short4, int4, op>  },
                    { device::fmx<short, schar, short, float, op>, device::fmx<short2, char2, short2, float2, op>, device::fmx<short3, char3, short3, float3, op>, device::fmx<short4, char4, short4, float4, op>  },
                    { device::fmx<short, schar, short, double, op>, device::fmx<short2, char2, short2, double2, op>, device::fmx<short3, char3, short3, double3, op>, device::fmx<short4, char4, short4, double4, op>  },
                },
                {
                    { device::fmx<short, schar, int, uchar, op>, device::fmx<short2, char2, int2, uchar2, op>, device::fmx<short3, char3, int3, uchar3, op>, device::fmx<short4, char4, int4, uchar4, op>  },
                    { device::fmx<short, schar, int, schar, op>, device::fmx<short2, char2, int2, char2, op>, device::fmx<short3, char3, int3, char3, op>, device::fmx<short4, char4, int4, char4, op>  },
                    { device::fmx<short, schar, int, ushort, op>, device::fmx<short2, char2, int2, ushort2, op>, device::fmx<short3, char3, int3, ushort3, op>, device::fmx<short4, char4, int4, ushort4, op>  },
                    { device::fmx<short, schar, int, short, op>, device::fmx<short2, char2, int2, short2, op>, device::fmx<short3, char3, int3, short3, op>, device::fmx<short4, char4, int4, short4, op>  },
                    { device::fmx<short, schar, int, int, op>, device::fmx<short2, char2, int2, int2, op>, device::fmx<short3, char3, int3, int3, op>, device::fmx<short4, char4, int4, int4, op>  },
                    { device::fmx<short, schar, int, float, op>, device::fmx<short2, char2, int2, float2, op>, device::fmx<short3, char3, int3, float3, op>, device::fmx<short4, char4, int4, float4, op>  },
                    { device::fmx<short, schar, int, double, op>, device::fmx<short2, char2, int2, double2, op>, device::fmx<short3, char3, int3, double3, op>, device::fmx<short4, char4, int4, double4, op>  },
                },
                {
                    { device::fmx<short, schar, float, uchar, op>, device::fmx<short2, char2, float2, uchar2, op>, device::fmx<short3, char3, float3, uchar3, op>, device::fmx<short4, char4, float4, uchar4, op>  },
                    { device::fmx<short, schar, float, schar, op>, device::fmx<short2, char2, float2, char2, op>, device::fmx<short3, char3, float3, char3, op>, device::fmx<short4, char4, float4, char4, op>  },
                    { device::fmx<short, schar, float, ushort, op>, device::fmx<short2, char2, float2, ushort2, op>, device::fmx<short3, char3, float3, ushort3, op>, device::fmx<short4, char4, float4, ushort4, op>  },
                    { device::fmx<short, schar, float, short, op>, device::fmx<short2, char2, float2, short2, op>, device::fmx<short3, char3, float3, short3, op>, device::fmx<short4, char4, float4, short4, op>  },
                    { device::fmx<short, schar, float, int, op>, device::fmx<short2, char2, float2, int2, op>, device::fmx<short3, char3, float3, int3, op>, device::fmx<short4, char4, float4, int4, op>  },
                    { device::fmx<short, schar, float, float, op>, device::fmx<short2, char2, float2, float2, op>, device::fmx<short3, char3, float3, float3, op>, device::fmx<short4, char4, float4, float4, op>  },
                    { device::fmx<short, schar, float, double, op>, device::fmx<short2, char2, float2, double2, op>, device::fmx<short3, char3, float3, double3, op>, device::fmx<short4, char4, float4, double4, op>  },
                },
                {
                    { device::fmx<short, schar, double, uchar, op>, device::fmx<short2, char2, double2, uchar2, op>, device::fmx<short3, char3, double3, uchar3, op>, device::fmx<short4, char4, double4, uchar4, op>  },
                    { device::fmx<short, schar, double, schar, op>, device::fmx<short2, char2, double2, char2, op>, device::fmx<short3, char3, double3, char3, op>, device::fmx<short4, char4, double4, char4, op>  },
                    { device::fmx<short, schar, double, ushort, op>, device::fmx<short2, char2, double2, ushort2, op>, device::fmx<short3, char3, double3, ushort3, op>, device::fmx<short4, char4, double4, ushort4, op>  },
                    { device::fmx<short, schar, double, short, op>, device::fmx<short2, char2, double2, short2, op>, device::fmx<short3, char3, double3, short3, op>, device::fmx<short4, char4, double4, short4, op>  },
                    { device::fmx<short, schar, double, int, op>, device::fmx<short2, char2, double2, int2, op>, device::fmx<short3, char3, double3, int3, op>, device::fmx<short4, char4, double4, int4, op>  },
                    { device::fmx<short, schar, double, float, op>, device::fmx<short2, char2, double2, float2, op>, device::fmx<short3, char3, double3, float3, op>, device::fmx<short4, char4, double4, float4, op>  },
                    { device::fmx<short, schar, double, double, op>, device::fmx<short2, char2, double2, double2, op>, device::fmx<short3, char3, double3, double3, op>, device::fmx<short4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<short, ushort, uchar, uchar, op>, device::fmx<short2, ushort2, uchar2, uchar2, op>, device::fmx<short3, ushort3, uchar3, uchar3, op>, device::fmx<short4, ushort4, uchar4, uchar4, op>  },
                    { device::fmx<short, ushort, uchar, schar, op>, device::fmx<short2, ushort2, uchar2, char2, op>, device::fmx<short3, ushort3, uchar3, char3, op>, device::fmx<short4, ushort4, uchar4, char4, op>  },
                    { device::fmx<short, ushort, uchar, ushort, op>, device::fmx<short2, ushort2, uchar2, ushort2, op>, device::fmx<short3, ushort3, uchar3, ushort3, op>, device::fmx<short4, ushort4, uchar4, ushort4, op>  },
                    { device::fmx<short, ushort, uchar, short, op>, device::fmx<short2, ushort2, uchar2, short2, op>, device::fmx<short3, ushort3, uchar3, short3, op>, device::fmx<short4, ushort4, uchar4, short4, op>  },
                    { device::fmx<short, ushort, uchar, int, op>, device::fmx<short2, ushort2, uchar2, int2, op>, device::fmx<short3, ushort3, uchar3, int3, op>, device::fmx<short4, ushort4, uchar4, int4, op>  },
                    { device::fmx<short, ushort, uchar, float, op>, device::fmx<short2, ushort2, uchar2, float2, op>, device::fmx<short3, ushort3, uchar3, float3, op>, device::fmx<short4, ushort4, uchar4, float4, op>  },
                    { device::fmx<short, ushort, uchar, double, op>, device::fmx<short2, ushort2, uchar2, double2, op>, device::fmx<short3, ushort3, uchar3, double3, op>, device::fmx<short4, ushort4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<short, ushort, schar, uchar, op>, device::fmx<short2, ushort2, char2, uchar2, op>, device::fmx<short3, ushort3, char3, uchar3, op>, device::fmx<short4, ushort4, char4, uchar4, op>  },
                    { device::fmx<short, ushort, schar, schar, op>, device::fmx<short2, ushort2, char2, char2, op>, device::fmx<short3, ushort3, char3, char3, op>, device::fmx<short4, ushort4, char4, char4, op>  },
                    { device::fmx<short, ushort, schar, ushort, op>, device::fmx<short2, ushort2, char2, ushort2, op>, device::fmx<short3, ushort3, char3, ushort3, op>, device::fmx<short4, ushort4, char4, ushort4, op>  },
                    { device::fmx<short, ushort, schar, short, op>, device::fmx<short2, ushort2, char2, short2, op>, device::fmx<short3, ushort3, char3, short3, op>, device::fmx<short4, ushort4, char4, short4, op>  },
                    { device::fmx<short, ushort, schar, int, op>, device::fmx<short2, ushort2, char2, int2, op>, device::fmx<short3, ushort3, char3, int3, op>, device::fmx<short4, ushort4, char4, int4, op>  },
                    { device::fmx<short, ushort, schar, float, op>, device::fmx<short2, ushort2, char2, float2, op>, device::fmx<short3, ushort3, char3, float3, op>, device::fmx<short4, ushort4, char4, float4, op>  },
                    { device::fmx<short, ushort, schar, double, op>, device::fmx<short2, ushort2, char2, double2, op>, device::fmx<short3, ushort3, char3, double3, op>, device::fmx<short4, ushort4, char4, double4, op>  },
                },
                {
                    { device::fmx<short, ushort, ushort, uchar, op>, device::fmx<short2, ushort2, ushort2, uchar2, op>, device::fmx<short3, ushort3, ushort3, uchar3, op>, device::fmx<short4, ushort4, ushort4, uchar4, op>  },
                    { device::fmx<short, ushort, ushort, schar, op>, device::fmx<short2, ushort2, ushort2, char2, op>, device::fmx<short3, ushort3, ushort3, char3, op>, device::fmx<short4, ushort4, ushort4, char4, op>  },
                    { device::fmx<short, ushort, ushort, ushort, op>, device::fmx<short2, ushort2, ushort2, ushort2, op>, device::fmx<short3, ushort3, ushort3, ushort3, op>, device::fmx<short4, ushort4, ushort4, ushort4, op>  },
                    { device::fmx<short, ushort, ushort, short, op>, device::fmx<short2, ushort2, ushort2, short2, op>, device::fmx<short3, ushort3, ushort3, short3, op>, device::fmx<short4, ushort4, ushort4, short4, op>  },
                    { device::fmx<short, ushort, ushort, int, op>, device::fmx<short2, ushort2, ushort2, int2, op>, device::fmx<short3, ushort3, ushort3, int3, op>, device::fmx<short4, ushort4, ushort4, int4, op>  },
                    { device::fmx<short, ushort, ushort, float, op>, device::fmx<short2, ushort2, ushort2, float2, op>, device::fmx<short3, ushort3, ushort3, float3, op>, device::fmx<short4, ushort4, ushort4, float4, op>  },
                    { device::fmx<short, ushort, ushort, double, op>, device::fmx<short2, ushort2, ushort2, double2, op>, device::fmx<short3, ushort3, ushort3, double3, op>, device::fmx<short4, ushort4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<short, ushort, short, uchar, op>, device::fmx<short2, ushort2, short2, uchar2, op>, device::fmx<short3, ushort3, short3, uchar3, op>, device::fmx<short4, ushort4, short4, uchar4, op>  },
                    { device::fmx<short, ushort, short, schar, op>, device::fmx<short2, ushort2, short2, char2, op>, device::fmx<short3, ushort3, short3, char3, op>, device::fmx<short4, ushort4, short4, char4, op>  },
                    { device::fmx<short, ushort, short, ushort, op>, device::fmx<short2, ushort2, short2, ushort2, op>, device::fmx<short3, ushort3, short3, ushort3, op>, device::fmx<short4, ushort4, short4, ushort4, op>  },
                    { device::fmx<short, ushort, short, short, op>, device::fmx<short2, ushort2, short2, short2, op>, device::fmx<short3, ushort3, short3, short3, op>, device::fmx<short4, ushort4, short4, short4, op>  },
                    { device::fmx<short, ushort, short, int, op>, device::fmx<short2, ushort2, short2, int2, op>, device::fmx<short3, ushort3, short3, int3, op>, device::fmx<short4, ushort4, short4, int4, op>  },
                    { device::fmx<short, ushort, short, float, op>, device::fmx<short2, ushort2, short2, float2, op>, device::fmx<short3, ushort3, short3, float3, op>, device::fmx<short4, ushort4, short4, float4, op>  },
                    { device::fmx<short, ushort, short, double, op>, device::fmx<short2, ushort2, short2, double2, op>, device::fmx<short3, ushort3, short3, double3, op>, device::fmx<short4, ushort4, short4, double4, op>  },
                },
                {
                    { device::fmx<short, ushort, int, uchar, op>, device::fmx<short2, ushort2, int2, uchar2, op>, device::fmx<short3, ushort3, int3, uchar3, op>, device::fmx<short4, ushort4, int4, uchar4, op>  },
                    { device::fmx<short, ushort, int, schar, op>, device::fmx<short2, ushort2, int2, char2, op>, device::fmx<short3, ushort3, int3, char3, op>, device::fmx<short4, ushort4, int4, char4, op>  },
                    { device::fmx<short, ushort, int, ushort, op>, device::fmx<short2, ushort2, int2, ushort2, op>, device::fmx<short3, ushort3, int3, ushort3, op>, device::fmx<short4, ushort4, int4, ushort4, op>  },
                    { device::fmx<short, ushort, int, short, op>, device::fmx<short2, ushort2, int2, short2, op>, device::fmx<short3, ushort3, int3, short3, op>, device::fmx<short4, ushort4, int4, short4, op>  },
                    { device::fmx<short, ushort, int, int, op>, device::fmx<short2, ushort2, int2, int2, op>, device::fmx<short3, ushort3, int3, int3, op>, device::fmx<short4, ushort4, int4, int4, op>  },
                    { device::fmx<short, ushort, int, float, op>, device::fmx<short2, ushort2, int2, float2, op>, device::fmx<short3, ushort3, int3, float3, op>, device::fmx<short4, ushort4, int4, float4, op>  },
                    { device::fmx<short, ushort, int, double, op>, device::fmx<short2, ushort2, int2, double2, op>, device::fmx<short3, ushort3, int3, double3, op>, device::fmx<short4, ushort4, int4, double4, op>  },
                },
                {
                    { device::fmx<short, ushort, float, uchar, op>, device::fmx<short2, ushort2, float2, uchar2, op>, device::fmx<short3, ushort3, float3, uchar3, op>, device::fmx<short4, ushort4, float4, uchar4, op>  },
                    { device::fmx<short, ushort, float, schar, op>, device::fmx<short2, ushort2, float2, char2, op>, device::fmx<short3, ushort3, float3, char3, op>, device::fmx<short4, ushort4, float4, char4, op>  },
                    { device::fmx<short, ushort, float, ushort, op>, device::fmx<short2, ushort2, float2, ushort2, op>, device::fmx<short3, ushort3, float3, ushort3, op>, device::fmx<short4, ushort4, float4, ushort4, op>  },
                    { device::fmx<short, ushort, float, short, op>, device::fmx<short2, ushort2, float2, short2, op>, device::fmx<short3, ushort3, float3, short3, op>, device::fmx<short4, ushort4, float4, short4, op>  },
                    { device::fmx<short, ushort, float, int, op>, device::fmx<short2, ushort2, float2, int2, op>, device::fmx<short3, ushort3, float3, int3, op>, device::fmx<short4, ushort4, float4, int4, op>  },
                    { device::fmx<short, ushort, float, float, op>, device::fmx<short2, ushort2, float2, float2, op>, device::fmx<short3, ushort3, float3, float3, op>, device::fmx<short4, ushort4, float4, float4, op>  },
                    { device::fmx<short, ushort, float, double, op>, device::fmx<short2, ushort2, float2, double2, op>, device::fmx<short3, ushort3, float3, double3, op>, device::fmx<short4, ushort4, float4, double4, op>  },
                },
                {
                    { device::fmx<short, ushort, double, uchar, op>, device::fmx<short2, ushort2, double2, uchar2, op>, device::fmx<short3, ushort3, double3, uchar3, op>, device::fmx<short4, ushort4, double4, uchar4, op>  },
                    { device::fmx<short, ushort, double, schar, op>, device::fmx<short2, ushort2, double2, char2, op>, device::fmx<short3, ushort3, double3, char3, op>, device::fmx<short4, ushort4, double4, char4, op>  },
                    { device::fmx<short, ushort, double, ushort, op>, device::fmx<short2, ushort2, double2, ushort2, op>, device::fmx<short3, ushort3, double3, ushort3, op>, device::fmx<short4, ushort4, double4, ushort4, op>  },
                    { device::fmx<short, ushort, double, short, op>, device::fmx<short2, ushort2, double2, short2, op>, device::fmx<short3, ushort3, double3, short3, op>, device::fmx<short4, ushort4, double4, short4, op>  },
                    { device::fmx<short, ushort, double, int, op>, device::fmx<short2, ushort2, double2, int2, op>, device::fmx<short3, ushort3, double3, int3, op>, device::fmx<short4, ushort4, double4, int4, op>  },
                    { device::fmx<short, ushort, double, float, op>, device::fmx<short2, ushort2, double2, float2, op>, device::fmx<short3, ushort3, double3, float3, op>, device::fmx<short4, ushort4, double4, float4, op>  },
                    { device::fmx<short, ushort, double, double, op>, device::fmx<short2, ushort2, double2, double2, op>, device::fmx<short3, ushort3, double3, double3, op>, device::fmx<short4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<short, short, uchar, uchar, op>, device::fmx<short2, short2, uchar2, uchar2, op>, device::fmx<short3, short3, uchar3, uchar3, op>, device::fmx<short4, short4, uchar4, uchar4, op>  },
                    { device::fmx<short, short, uchar, schar, op>, device::fmx<short2, short2, uchar2, char2, op>, device::fmx<short3, short3, uchar3, char3, op>, device::fmx<short4, short4, uchar4, char4, op>  },
                    { device::fmx<short, short, uchar, ushort, op>, device::fmx<short2, short2, uchar2, ushort2, op>, device::fmx<short3, short3, uchar3, ushort3, op>, device::fmx<short4, short4, uchar4, ushort4, op>  },
                    { device::fmx<short, short, uchar, short, op>, device::fmx<short2, short2, uchar2, short2, op>, device::fmx<short3, short3, uchar3, short3, op>, device::fmx<short4, short4, uchar4, short4, op>  },
                    { device::fmx<short, short, uchar, int, op>, device::fmx<short2, short2, uchar2, int2, op>, device::fmx<short3, short3, uchar3, int3, op>, device::fmx<short4, short4, uchar4, int4, op>  },
                    { device::fmx<short, short, uchar, float, op>, device::fmx<short2, short2, uchar2, float2, op>, device::fmx<short3, short3, uchar3, float3, op>, device::fmx<short4, short4, uchar4, float4, op>  },
                    { device::fmx<short, short, uchar, double, op>, device::fmx<short2, short2, uchar2, double2, op>, device::fmx<short3, short3, uchar3, double3, op>, device::fmx<short4, short4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<short, short, schar, uchar, op>, device::fmx<short2, short2, char2, uchar2, op>, device::fmx<short3, short3, char3, uchar3, op>, device::fmx<short4, short4, char4, uchar4, op>  },
                    { device::fmx<short, short, schar, schar, op>, device::fmx<short2, short2, char2, char2, op>, device::fmx<short3, short3, char3, char3, op>, device::fmx<short4, short4, char4, char4, op>  },
                    { device::fmx<short, short, schar, ushort, op>, device::fmx<short2, short2, char2, ushort2, op>, device::fmx<short3, short3, char3, ushort3, op>, device::fmx<short4, short4, char4, ushort4, op>  },
                    { device::fmx<short, short, schar, short, op>, device::fmx<short2, short2, char2, short2, op>, device::fmx<short3, short3, char3, short3, op>, device::fmx<short4, short4, char4, short4, op>  },
                    { device::fmx<short, short, schar, int, op>, device::fmx<short2, short2, char2, int2, op>, device::fmx<short3, short3, char3, int3, op>, device::fmx<short4, short4, char4, int4, op>  },
                    { device::fmx<short, short, schar, float, op>, device::fmx<short2, short2, char2, float2, op>, device::fmx<short3, short3, char3, float3, op>, device::fmx<short4, short4, char4, float4, op>  },
                    { device::fmx<short, short, schar, double, op>, device::fmx<short2, short2, char2, double2, op>, device::fmx<short3, short3, char3, double3, op>, device::fmx<short4, short4, char4, double4, op>  },
                },
                {
                    { device::fmx<short, short, ushort, uchar, op>, device::fmx<short2, short2, ushort2, uchar2, op>, device::fmx<short3, short3, ushort3, uchar3, op>, device::fmx<short4, short4, ushort4, uchar4, op>  },
                    { device::fmx<short, short, ushort, schar, op>, device::fmx<short2, short2, ushort2, char2, op>, device::fmx<short3, short3, ushort3, char3, op>, device::fmx<short4, short4, ushort4, char4, op>  },
                    { device::fmx<short, short, ushort, ushort, op>, device::fmx<short2, short2, ushort2, ushort2, op>, device::fmx<short3, short3, ushort3, ushort3, op>, device::fmx<short4, short4, ushort4, ushort4, op>  },
                    { device::fmx<short, short, ushort, short, op>, device::fmx<short2, short2, ushort2, short2, op>, device::fmx<short3, short3, ushort3, short3, op>, device::fmx<short4, short4, ushort4, short4, op>  },
                    { device::fmx<short, short, ushort, int, op>, device::fmx<short2, short2, ushort2, int2, op>, device::fmx<short3, short3, ushort3, int3, op>, device::fmx<short4, short4, ushort4, int4, op>  },
                    { device::fmx<short, short, ushort, float, op>, device::fmx<short2, short2, ushort2, float2, op>, device::fmx<short3, short3, ushort3, float3, op>, device::fmx<short4, short4, ushort4, float4, op>  },
                    { device::fmx<short, short, ushort, double, op>, device::fmx<short2, short2, ushort2, double2, op>, device::fmx<short3, short3, ushort3, double3, op>, device::fmx<short4, short4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<short, short, short, uchar, op>, device::fmx<short2, short2, short2, uchar2, op>, device::fmx<short3, short3, short3, uchar3, op>, device::fmx<short4, short4, short4, uchar4, op>  },
                    { device::fmx<short, short, short, schar, op>, device::fmx<short2, short2, short2, char2, op>, device::fmx<short3, short3, short3, char3, op>, device::fmx<short4, short4, short4, char4, op>  },
                    { device::fmx<short, short, short, ushort, op>, device::fmx<short2, short2, short2, ushort2, op>, device::fmx<short3, short3, short3, ushort3, op>, device::fmx<short4, short4, short4, ushort4, op>  },
                    { device::fmx<short, short, short, short, op>, device::fmx<short2, short2, short2, short2, op>, device::fmx<short3, short3, short3, short3, op>, device::fmx<short4, short4, short4, short4, op>  },
                    { device::fmx<short, short, short, int, op>, device::fmx<short2, short2, short2, int2, op>, device::fmx<short3, short3, short3, int3, op>, device::fmx<short4, short4, short4, int4, op>  },
                    { device::fmx<short, short, short, float, op>, device::fmx<short2, short2, short2, float2, op>, device::fmx<short3, short3, short3, float3, op>, device::fmx<short4, short4, short4, float4, op>  },
                    { device::fmx<short, short, short, double, op>, device::fmx<short2, short2, short2, double2, op>, device::fmx<short3, short3, short3, double3, op>, device::fmx<short4, short4, short4, double4, op>  },
                },
                {
                    { device::fmx<short, short, int, uchar, op>, device::fmx<short2, short2, int2, uchar2, op>, device::fmx<short3, short3, int3, uchar3, op>, device::fmx<short4, short4, int4, uchar4, op>  },
                    { device::fmx<short, short, int, schar, op>, device::fmx<short2, short2, int2, char2, op>, device::fmx<short3, short3, int3, char3, op>, device::fmx<short4, short4, int4, char4, op>  },
                    { device::fmx<short, short, int, ushort, op>, device::fmx<short2, short2, int2, ushort2, op>, device::fmx<short3, short3, int3, ushort3, op>, device::fmx<short4, short4, int4, ushort4, op>  },
                    { device::fmx<short, short, int, short, op>, device::fmx<short2, short2, int2, short2, op>, device::fmx<short3, short3, int3, short3, op>, device::fmx<short4, short4, int4, short4, op>  },
                    { device::fmx<short, short, int, int, op>, device::fmx<short2, short2, int2, int2, op>, device::fmx<short3, short3, int3, int3, op>, device::fmx<short4, short4, int4, int4, op>  },
                    { device::fmx<short, short, int, float, op>, device::fmx<short2, short2, int2, float2, op>, device::fmx<short3, short3, int3, float3, op>, device::fmx<short4, short4, int4, float4, op>  },
                    { device::fmx<short, short, int, double, op>, device::fmx<short2, short2, int2, double2, op>, device::fmx<short3, short3, int3, double3, op>, device::fmx<short4, short4, int4, double4, op>  },
                },
                {
                    { device::fmx<short, short, float, uchar, op>, device::fmx<short2, short2, float2, uchar2, op>, device::fmx<short3, short3, float3, uchar3, op>, device::fmx<short4, short4, float4, uchar4, op>  },
                    { device::fmx<short, short, float, schar, op>, device::fmx<short2, short2, float2, char2, op>, device::fmx<short3, short3, float3, char3, op>, device::fmx<short4, short4, float4, char4, op>  },
                    { device::fmx<short, short, float, ushort, op>, device::fmx<short2, short2, float2, ushort2, op>, device::fmx<short3, short3, float3, ushort3, op>, device::fmx<short4, short4, float4, ushort4, op>  },
                    { device::fmx<short, short, float, short, op>, device::fmx<short2, short2, float2, short2, op>, device::fmx<short3, short3, float3, short3, op>, device::fmx<short4, short4, float4, short4, op>  },
                    { device::fmx<short, short, float, int, op>, device::fmx<short2, short2, float2, int2, op>, device::fmx<short3, short3, float3, int3, op>, device::fmx<short4, short4, float4, int4, op>  },
                    { device::fmx<short, short, float, float, op>, device::fmx<short2, short2, float2, float2, op>, device::fmx<short3, short3, float3, float3, op>, device::fmx<short4, short4, float4, float4, op>  },
                    { device::fmx<short, short, float, double, op>, device::fmx<short2, short2, float2, double2, op>, device::fmx<short3, short3, float3, double3, op>, device::fmx<short4, short4, float4, double4, op>  },
                },
                {
                    { device::fmx<short, short, double, uchar, op>, device::fmx<short2, short2, double2, uchar2, op>, device::fmx<short3, short3, double3, uchar3, op>, device::fmx<short4, short4, double4, uchar4, op>  },
                    { device::fmx<short, short, double, schar, op>, device::fmx<short2, short2, double2, char2, op>, device::fmx<short3, short3, double3, char3, op>, device::fmx<short4, short4, double4, char4, op>  },
                    { device::fmx<short, short, double, ushort, op>, device::fmx<short2, short2, double2, ushort2, op>, device::fmx<short3, short3, double3, ushort3, op>, device::fmx<short4, short4, double4, ushort4, op>  },
                    { device::fmx<short, short, double, short, op>, device::fmx<short2, short2, double2, short2, op>, device::fmx<short3, short3, double3, short3, op>, device::fmx<short4, short4, double4, short4, op>  },
                    { device::fmx<short, short, double, int, op>, device::fmx<short2, short2, double2, int2, op>, device::fmx<short3, short3, double3, int3, op>, device::fmx<short4, short4, double4, int4, op>  },
                    { device::fmx<short, short, double, float, op>, device::fmx<short2, short2, double2, float2, op>, device::fmx<short3, short3, double3, float3, op>, device::fmx<short4, short4, double4, float4, op>  },
                    { device::fmx<short, short, double, double, op>, device::fmx<short2, short2, double2, double2, op>, device::fmx<short3, short3, double3, double3, op>, device::fmx<short4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<short, int, uchar, uchar, op>, device::fmx<short2, int2, uchar2, uchar2, op>, device::fmx<short3, int3, uchar3, uchar3, op>, device::fmx<short4, int4, uchar4, uchar4, op>  },
                    { device::fmx<short, int, uchar, schar, op>, device::fmx<short2, int2, uchar2, char2, op>, device::fmx<short3, int3, uchar3, char3, op>, device::fmx<short4, int4, uchar4, char4, op>  },
                    { device::fmx<short, int, uchar, ushort, op>, device::fmx<short2, int2, uchar2, ushort2, op>, device::fmx<short3, int3, uchar3, ushort3, op>, device::fmx<short4, int4, uchar4, ushort4, op>  },
                    { device::fmx<short, int, uchar, short, op>, device::fmx<short2, int2, uchar2, short2, op>, device::fmx<short3, int3, uchar3, short3, op>, device::fmx<short4, int4, uchar4, short4, op>  },
                    { device::fmx<short, int, uchar, int, op>, device::fmx<short2, int2, uchar2, int2, op>, device::fmx<short3, int3, uchar3, int3, op>, device::fmx<short4, int4, uchar4, int4, op>  },
                    { device::fmx<short, int, uchar, float, op>, device::fmx<short2, int2, uchar2, float2, op>, device::fmx<short3, int3, uchar3, float3, op>, device::fmx<short4, int4, uchar4, float4, op>  },
                    { device::fmx<short, int, uchar, double, op>, device::fmx<short2, int2, uchar2, double2, op>, device::fmx<short3, int3, uchar3, double3, op>, device::fmx<short4, int4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<short, int, schar, uchar, op>, device::fmx<short2, int2, char2, uchar2, op>, device::fmx<short3, int3, char3, uchar3, op>, device::fmx<short4, int4, char4, uchar4, op>  },
                    { device::fmx<short, int, schar, schar, op>, device::fmx<short2, int2, char2, char2, op>, device::fmx<short3, int3, char3, char3, op>, device::fmx<short4, int4, char4, char4, op>  },
                    { device::fmx<short, int, schar, ushort, op>, device::fmx<short2, int2, char2, ushort2, op>, device::fmx<short3, int3, char3, ushort3, op>, device::fmx<short4, int4, char4, ushort4, op>  },
                    { device::fmx<short, int, schar, short, op>, device::fmx<short2, int2, char2, short2, op>, device::fmx<short3, int3, char3, short3, op>, device::fmx<short4, int4, char4, short4, op>  },
                    { device::fmx<short, int, schar, int, op>, device::fmx<short2, int2, char2, int2, op>, device::fmx<short3, int3, char3, int3, op>, device::fmx<short4, int4, char4, int4, op>  },
                    { device::fmx<short, int, schar, float, op>, device::fmx<short2, int2, char2, float2, op>, device::fmx<short3, int3, char3, float3, op>, device::fmx<short4, int4, char4, float4, op>  },
                    { device::fmx<short, int, schar, double, op>, device::fmx<short2, int2, char2, double2, op>, device::fmx<short3, int3, char3, double3, op>, device::fmx<short4, int4, char4, double4, op>  },
                },
                {
                    { device::fmx<short, int, ushort, uchar, op>, device::fmx<short2, int2, ushort2, uchar2, op>, device::fmx<short3, int3, ushort3, uchar3, op>, device::fmx<short4, int4, ushort4, uchar4, op>  },
                    { device::fmx<short, int, ushort, schar, op>, device::fmx<short2, int2, ushort2, char2, op>, device::fmx<short3, int3, ushort3, char3, op>, device::fmx<short4, int4, ushort4, char4, op>  },
                    { device::fmx<short, int, ushort, ushort, op>, device::fmx<short2, int2, ushort2, ushort2, op>, device::fmx<short3, int3, ushort3, ushort3, op>, device::fmx<short4, int4, ushort4, ushort4, op>  },
                    { device::fmx<short, int, ushort, short, op>, device::fmx<short2, int2, ushort2, short2, op>, device::fmx<short3, int3, ushort3, short3, op>, device::fmx<short4, int4, ushort4, short4, op>  },
                    { device::fmx<short, int, ushort, int, op>, device::fmx<short2, int2, ushort2, int2, op>, device::fmx<short3, int3, ushort3, int3, op>, device::fmx<short4, int4, ushort4, int4, op>  },
                    { device::fmx<short, int, ushort, float, op>, device::fmx<short2, int2, ushort2, float2, op>, device::fmx<short3, int3, ushort3, float3, op>, device::fmx<short4, int4, ushort4, float4, op>  },
                    { device::fmx<short, int, ushort, double, op>, device::fmx<short2, int2, ushort2, double2, op>, device::fmx<short3, int3, ushort3, double3, op>, device::fmx<short4, int4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<short, int, short, uchar, op>, device::fmx<short2, int2, short2, uchar2, op>, device::fmx<short3, int3, short3, uchar3, op>, device::fmx<short4, int4, short4, uchar4, op>  },
                    { device::fmx<short, int, short, schar, op>, device::fmx<short2, int2, short2, char2, op>, device::fmx<short3, int3, short3, char3, op>, device::fmx<short4, int4, short4, char4, op>  },
                    { device::fmx<short, int, short, ushort, op>, device::fmx<short2, int2, short2, ushort2, op>, device::fmx<short3, int3, short3, ushort3, op>, device::fmx<short4, int4, short4, ushort4, op>  },
                    { device::fmx<short, int, short, short, op>, device::fmx<short2, int2, short2, short2, op>, device::fmx<short3, int3, short3, short3, op>, device::fmx<short4, int4, short4, short4, op>  },
                    { device::fmx<short, int, short, int, op>, device::fmx<short2, int2, short2, int2, op>, device::fmx<short3, int3, short3, int3, op>, device::fmx<short4, int4, short4, int4, op>  },
                    { device::fmx<short, int, short, float, op>, device::fmx<short2, int2, short2, float2, op>, device::fmx<short3, int3, short3, float3, op>, device::fmx<short4, int4, short4, float4, op>  },
                    { device::fmx<short, int, short, double, op>, device::fmx<short2, int2, short2, double2, op>, device::fmx<short3, int3, short3, double3, op>, device::fmx<short4, int4, short4, double4, op>  },
                },
                {
                    { device::fmx<short, int, int, uchar, op>, device::fmx<short2, int2, int2, uchar2, op>, device::fmx<short3, int3, int3, uchar3, op>, device::fmx<short4, int4, int4, uchar4, op>  },
                    { device::fmx<short, int, int, schar, op>, device::fmx<short2, int2, int2, char2, op>, device::fmx<short3, int3, int3, char3, op>, device::fmx<short4, int4, int4, char4, op>  },
                    { device::fmx<short, int, int, ushort, op>, device::fmx<short2, int2, int2, ushort2, op>, device::fmx<short3, int3, int3, ushort3, op>, device::fmx<short4, int4, int4, ushort4, op>  },
                    { device::fmx<short, int, int, short, op>, device::fmx<short2, int2, int2, short2, op>, device::fmx<short3, int3, int3, short3, op>, device::fmx<short4, int4, int4, short4, op>  },
                    { device::fmx<short, int, int, int, op>, device::fmx<short2, int2, int2, int2, op>, device::fmx<short3, int3, int3, int3, op>, device::fmx<short4, int4, int4, int4, op>  },
                    { device::fmx<short, int, int, float, op>, device::fmx<short2, int2, int2, float2, op>, device::fmx<short3, int3, int3, float3, op>, device::fmx<short4, int4, int4, float4, op>  },
                    { device::fmx<short, int, int, double, op>, device::fmx<short2, int2, int2, double2, op>, device::fmx<short3, int3, int3, double3, op>, device::fmx<short4, int4, int4, double4, op>  },
                },
                {
                    { device::fmx<short, int, float, uchar, op>, device::fmx<short2, int2, float2, uchar2, op>, device::fmx<short3, int3, float3, uchar3, op>, device::fmx<short4, int4, float4, uchar4, op>  },
                    { device::fmx<short, int, float, schar, op>, device::fmx<short2, int2, float2, char2, op>, device::fmx<short3, int3, float3, char3, op>, device::fmx<short4, int4, float4, char4, op>  },
                    { device::fmx<short, int, float, ushort, op>, device::fmx<short2, int2, float2, ushort2, op>, device::fmx<short3, int3, float3, ushort3, op>, device::fmx<short4, int4, float4, ushort4, op>  },
                    { device::fmx<short, int, float, short, op>, device::fmx<short2, int2, float2, short2, op>, device::fmx<short3, int3, float3, short3, op>, device::fmx<short4, int4, float4, short4, op>  },
                    { device::fmx<short, int, float, int, op>, device::fmx<short2, int2, float2, int2, op>, device::fmx<short3, int3, float3, int3, op>, device::fmx<short4, int4, float4, int4, op>  },
                    { device::fmx<short, int, float, float, op>, device::fmx<short2, int2, float2, float2, op>, device::fmx<short3, int3, float3, float3, op>, device::fmx<short4, int4, float4, float4, op>  },
                    { device::fmx<short, int, float, double, op>, device::fmx<short2, int2, float2, double2, op>, device::fmx<short3, int3, float3, double3, op>, device::fmx<short4, int4, float4, double4, op>  },
                },
                {
                    { device::fmx<short, int, double, uchar, op>, device::fmx<short2, int2, double2, uchar2, op>, device::fmx<short3, int3, double3, uchar3, op>, device::fmx<short4, int4, double4, uchar4, op>  },
                    { device::fmx<short, int, double, schar, op>, device::fmx<short2, int2, double2, char2, op>, device::fmx<short3, int3, double3, char3, op>, device::fmx<short4, int4, double4, char4, op>  },
                    { device::fmx<short, int, double, ushort, op>, device::fmx<short2, int2, double2, ushort2, op>, device::fmx<short3, int3, double3, ushort3, op>, device::fmx<short4, int4, double4, ushort4, op>  },
                    { device::fmx<short, int, double, short, op>, device::fmx<short2, int2, double2, short2, op>, device::fmx<short3, int3, double3, short3, op>, device::fmx<short4, int4, double4, short4, op>  },
                    { device::fmx<short, int, double, int, op>, device::fmx<short2, int2, double2, int2, op>, device::fmx<short3, int3, double3, int3, op>, device::fmx<short4, int4, double4, int4, op>  },
                    { device::fmx<short, int, double, float, op>, device::fmx<short2, int2, double2, float2, op>, device::fmx<short3, int3, double3, float3, op>, device::fmx<short4, int4, double4, float4, op>  },
                    { device::fmx<short, int, double, double, op>, device::fmx<short2, int2, double2, double2, op>, device::fmx<short3, int3, double3, double3, op>, device::fmx<short4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<short, float, uchar, uchar, op>, device::fmx<short2, float2, uchar2, uchar2, op>, device::fmx<short3, float3, uchar3, uchar3, op>, device::fmx<short4, float4, uchar4, uchar4, op>  },
                    { device::fmx<short, float, uchar, schar, op>, device::fmx<short2, float2, uchar2, char2, op>, device::fmx<short3, float3, uchar3, char3, op>, device::fmx<short4, float4, uchar4, char4, op>  },
                    { device::fmx<short, float, uchar, ushort, op>, device::fmx<short2, float2, uchar2, ushort2, op>, device::fmx<short3, float3, uchar3, ushort3, op>, device::fmx<short4, float4, uchar4, ushort4, op>  },
                    { device::fmx<short, float, uchar, short, op>, device::fmx<short2, float2, uchar2, short2, op>, device::fmx<short3, float3, uchar3, short3, op>, device::fmx<short4, float4, uchar4, short4, op>  },
                    { device::fmx<short, float, uchar, int, op>, device::fmx<short2, float2, uchar2, int2, op>, device::fmx<short3, float3, uchar3, int3, op>, device::fmx<short4, float4, uchar4, int4, op>  },
                    { device::fmx<short, float, uchar, float, op>, device::fmx<short2, float2, uchar2, float2, op>, device::fmx<short3, float3, uchar3, float3, op>, device::fmx<short4, float4, uchar4, float4, op>  },
                    { device::fmx<short, float, uchar, double, op>, device::fmx<short2, float2, uchar2, double2, op>, device::fmx<short3, float3, uchar3, double3, op>, device::fmx<short4, float4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<short, float, schar, uchar, op>, device::fmx<short2, float2, char2, uchar2, op>, device::fmx<short3, float3, char3, uchar3, op>, device::fmx<short4, float4, char4, uchar4, op>  },
                    { device::fmx<short, float, schar, schar, op>, device::fmx<short2, float2, char2, char2, op>, device::fmx<short3, float3, char3, char3, op>, device::fmx<short4, float4, char4, char4, op>  },
                    { device::fmx<short, float, schar, ushort, op>, device::fmx<short2, float2, char2, ushort2, op>, device::fmx<short3, float3, char3, ushort3, op>, device::fmx<short4, float4, char4, ushort4, op>  },
                    { device::fmx<short, float, schar, short, op>, device::fmx<short2, float2, char2, short2, op>, device::fmx<short3, float3, char3, short3, op>, device::fmx<short4, float4, char4, short4, op>  },
                    { device::fmx<short, float, schar, int, op>, device::fmx<short2, float2, char2, int2, op>, device::fmx<short3, float3, char3, int3, op>, device::fmx<short4, float4, char4, int4, op>  },
                    { device::fmx<short, float, schar, float, op>, device::fmx<short2, float2, char2, float2, op>, device::fmx<short3, float3, char3, float3, op>, device::fmx<short4, float4, char4, float4, op>  },
                    { device::fmx<short, float, schar, double, op>, device::fmx<short2, float2, char2, double2, op>, device::fmx<short3, float3, char3, double3, op>, device::fmx<short4, float4, char4, double4, op>  },
                },
                {
                    { device::fmx<short, float, ushort, uchar, op>, device::fmx<short2, float2, ushort2, uchar2, op>, device::fmx<short3, float3, ushort3, uchar3, op>, device::fmx<short4, float4, ushort4, uchar4, op>  },
                    { device::fmx<short, float, ushort, schar, op>, device::fmx<short2, float2, ushort2, char2, op>, device::fmx<short3, float3, ushort3, char3, op>, device::fmx<short4, float4, ushort4, char4, op>  },
                    { device::fmx<short, float, ushort, ushort, op>, device::fmx<short2, float2, ushort2, ushort2, op>, device::fmx<short3, float3, ushort3, ushort3, op>, device::fmx<short4, float4, ushort4, ushort4, op>  },
                    { device::fmx<short, float, ushort, short, op>, device::fmx<short2, float2, ushort2, short2, op>, device::fmx<short3, float3, ushort3, short3, op>, device::fmx<short4, float4, ushort4, short4, op>  },
                    { device::fmx<short, float, ushort, int, op>, device::fmx<short2, float2, ushort2, int2, op>, device::fmx<short3, float3, ushort3, int3, op>, device::fmx<short4, float4, ushort4, int4, op>  },
                    { device::fmx<short, float, ushort, float, op>, device::fmx<short2, float2, ushort2, float2, op>, device::fmx<short3, float3, ushort3, float3, op>, device::fmx<short4, float4, ushort4, float4, op>  },
                    { device::fmx<short, float, ushort, double, op>, device::fmx<short2, float2, ushort2, double2, op>, device::fmx<short3, float3, ushort3, double3, op>, device::fmx<short4, float4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<short, float, short, uchar, op>, device::fmx<short2, float2, short2, uchar2, op>, device::fmx<short3, float3, short3, uchar3, op>, device::fmx<short4, float4, short4, uchar4, op>  },
                    { device::fmx<short, float, short, schar, op>, device::fmx<short2, float2, short2, char2, op>, device::fmx<short3, float3, short3, char3, op>, device::fmx<short4, float4, short4, char4, op>  },
                    { device::fmx<short, float, short, ushort, op>, device::fmx<short2, float2, short2, ushort2, op>, device::fmx<short3, float3, short3, ushort3, op>, device::fmx<short4, float4, short4, ushort4, op>  },
                    { device::fmx<short, float, short, short, op>, device::fmx<short2, float2, short2, short2, op>, device::fmx<short3, float3, short3, short3, op>, device::fmx<short4, float4, short4, short4, op>  },
                    { device::fmx<short, float, short, int, op>, device::fmx<short2, float2, short2, int2, op>, device::fmx<short3, float3, short3, int3, op>, device::fmx<short4, float4, short4, int4, op>  },
                    { device::fmx<short, float, short, float, op>, device::fmx<short2, float2, short2, float2, op>, device::fmx<short3, float3, short3, float3, op>, device::fmx<short4, float4, short4, float4, op>  },
                    { device::fmx<short, float, short, double, op>, device::fmx<short2, float2, short2, double2, op>, device::fmx<short3, float3, short3, double3, op>, device::fmx<short4, float4, short4, double4, op>  },
                },
                {
                    { device::fmx<short, float, int, uchar, op>, device::fmx<short2, float2, int2, uchar2, op>, device::fmx<short3, float3, int3, uchar3, op>, device::fmx<short4, float4, int4, uchar4, op>  },
                    { device::fmx<short, float, int, schar, op>, device::fmx<short2, float2, int2, char2, op>, device::fmx<short3, float3, int3, char3, op>, device::fmx<short4, float4, int4, char4, op>  },
                    { device::fmx<short, float, int, ushort, op>, device::fmx<short2, float2, int2, ushort2, op>, device::fmx<short3, float3, int3, ushort3, op>, device::fmx<short4, float4, int4, ushort4, op>  },
                    { device::fmx<short, float, int, short, op>, device::fmx<short2, float2, int2, short2, op>, device::fmx<short3, float3, int3, short3, op>, device::fmx<short4, float4, int4, short4, op>  },
                    { device::fmx<short, float, int, int, op>, device::fmx<short2, float2, int2, int2, op>, device::fmx<short3, float3, int3, int3, op>, device::fmx<short4, float4, int4, int4, op>  },
                    { device::fmx<short, float, int, float, op>, device::fmx<short2, float2, int2, float2, op>, device::fmx<short3, float3, int3, float3, op>, device::fmx<short4, float4, int4, float4, op>  },
                    { device::fmx<short, float, int, double, op>, device::fmx<short2, float2, int2, double2, op>, device::fmx<short3, float3, int3, double3, op>, device::fmx<short4, float4, int4, double4, op>  },
                },
                {
                    { device::fmx<short, float, float, uchar, op>, device::fmx<short2, float2, float2, uchar2, op>, device::fmx<short3, float3, float3, uchar3, op>, device::fmx<short4, float4, float4, uchar4, op>  },
                    { device::fmx<short, float, float, schar, op>, device::fmx<short2, float2, float2, char2, op>, device::fmx<short3, float3, float3, char3, op>, device::fmx<short4, float4, float4, char4, op>  },
                    { device::fmx<short, float, float, ushort, op>, device::fmx<short2, float2, float2, ushort2, op>, device::fmx<short3, float3, float3, ushort3, op>, device::fmx<short4, float4, float4, ushort4, op>  },
                    { device::fmx<short, float, float, short, op>, device::fmx<short2, float2, float2, short2, op>, device::fmx<short3, float3, float3, short3, op>, device::fmx<short4, float4, float4, short4, op>  },
                    { device::fmx<short, float, float, int, op>, device::fmx<short2, float2, float2, int2, op>, device::fmx<short3, float3, float3, int3, op>, device::fmx<short4, float4, float4, int4, op>  },
                    { device::fmx<short, float, float, float, op>, device::fmx<short2, float2, float2, float2, op>, device::fmx<short3, float3, float3, float3, op>, device::fmx<short4, float4, float4, float4, op>  },
                    { device::fmx<short, float, float, double, op>, device::fmx<short2, float2, float2, double2, op>, device::fmx<short3, float3, float3, double3, op>, device::fmx<short4, float4, float4, double4, op>  },
                },
                {
                    { device::fmx<short, float, double, uchar, op>, device::fmx<short2, float2, double2, uchar2, op>, device::fmx<short3, float3, double3, uchar3, op>, device::fmx<short4, float4, double4, uchar4, op>  },
                    { device::fmx<short, float, double, schar, op>, device::fmx<short2, float2, double2, char2, op>, device::fmx<short3, float3, double3, char3, op>, device::fmx<short4, float4, double4, char4, op>  },
                    { device::fmx<short, float, double, ushort, op>, device::fmx<short2, float2, double2, ushort2, op>, device::fmx<short3, float3, double3, ushort3, op>, device::fmx<short4, float4, double4, ushort4, op>  },
                    { device::fmx<short, float, double, short, op>, device::fmx<short2, float2, double2, short2, op>, device::fmx<short3, float3, double3, short3, op>, device::fmx<short4, float4, double4, short4, op>  },
                    { device::fmx<short, float, double, int, op>, device::fmx<short2, float2, double2, int2, op>, device::fmx<short3, float3, double3, int3, op>, device::fmx<short4, float4, double4, int4, op>  },
                    { device::fmx<short, float, double, float, op>, device::fmx<short2, float2, double2, float2, op>, device::fmx<short3, float3, double3, float3, op>, device::fmx<short4, float4, double4, float4, op>  },
                    { device::fmx<short, float, double, double, op>, device::fmx<short2, float2, double2, double2, op>, device::fmx<short3, float3, double3, double3, op>, device::fmx<short4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<short, double, uchar, uchar, op>, device::fmx<short2, double2, uchar2, uchar2, op>, device::fmx<short3, double3, uchar3, uchar3, op>, device::fmx<short4, double4, uchar4, uchar4, op>  },
                    { device::fmx<short, double, uchar, schar, op>, device::fmx<short2, double2, uchar2, char2, op>, device::fmx<short3, double3, uchar3, char3, op>, device::fmx<short4, double4, uchar4, char4, op>  },
                    { device::fmx<short, double, uchar, ushort, op>, device::fmx<short2, double2, uchar2, ushort2, op>, device::fmx<short3, double3, uchar3, ushort3, op>, device::fmx<short4, double4, uchar4, ushort4, op>  },
                    { device::fmx<short, double, uchar, short, op>, device::fmx<short2, double2, uchar2, short2, op>, device::fmx<short3, double3, uchar3, short3, op>, device::fmx<short4, double4, uchar4, short4, op>  },
                    { device::fmx<short, double, uchar, int, op>, device::fmx<short2, double2, uchar2, int2, op>, device::fmx<short3, double3, uchar3, int3, op>, device::fmx<short4, double4, uchar4, int4, op>  },
                    { device::fmx<short, double, uchar, float, op>, device::fmx<short2, double2, uchar2, float2, op>, device::fmx<short3, double3, uchar3, float3, op>, device::fmx<short4, double4, uchar4, float4, op>  },
                    { device::fmx<short, double, uchar, double, op>, device::fmx<short2, double2, uchar2, double2, op>, device::fmx<short3, double3, uchar3, double3, op>, device::fmx<short4, double4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<short, double, schar, uchar, op>, device::fmx<short2, double2, char2, uchar2, op>, device::fmx<short3, double3, char3, uchar3, op>, device::fmx<short4, double4, char4, uchar4, op>  },
                    { device::fmx<short, double, schar, schar, op>, device::fmx<short2, double2, char2, char2, op>, device::fmx<short3, double3, char3, char3, op>, device::fmx<short4, double4, char4, char4, op>  },
                    { device::fmx<short, double, schar, ushort, op>, device::fmx<short2, double2, char2, ushort2, op>, device::fmx<short3, double3, char3, ushort3, op>, device::fmx<short4, double4, char4, ushort4, op>  },
                    { device::fmx<short, double, schar, short, op>, device::fmx<short2, double2, char2, short2, op>, device::fmx<short3, double3, char3, short3, op>, device::fmx<short4, double4, char4, short4, op>  },
                    { device::fmx<short, double, schar, int, op>, device::fmx<short2, double2, char2, int2, op>, device::fmx<short3, double3, char3, int3, op>, device::fmx<short4, double4, char4, int4, op>  },
                    { device::fmx<short, double, schar, float, op>, device::fmx<short2, double2, char2, float2, op>, device::fmx<short3, double3, char3, float3, op>, device::fmx<short4, double4, char4, float4, op>  },
                    { device::fmx<short, double, schar, double, op>, device::fmx<short2, double2, char2, double2, op>, device::fmx<short3, double3, char3, double3, op>, device::fmx<short4, double4, char4, double4, op>  },
                },
                {
                    { device::fmx<short, double, ushort, uchar, op>, device::fmx<short2, double2, ushort2, uchar2, op>, device::fmx<short3, double3, ushort3, uchar3, op>, device::fmx<short4, double4, ushort4, uchar4, op>  },
                    { device::fmx<short, double, ushort, schar, op>, device::fmx<short2, double2, ushort2, char2, op>, device::fmx<short3, double3, ushort3, char3, op>, device::fmx<short4, double4, ushort4, char4, op>  },
                    { device::fmx<short, double, ushort, ushort, op>, device::fmx<short2, double2, ushort2, ushort2, op>, device::fmx<short3, double3, ushort3, ushort3, op>, device::fmx<short4, double4, ushort4, ushort4, op>  },
                    { device::fmx<short, double, ushort, short, op>, device::fmx<short2, double2, ushort2, short2, op>, device::fmx<short3, double3, ushort3, short3, op>, device::fmx<short4, double4, ushort4, short4, op>  },
                    { device::fmx<short, double, ushort, int, op>, device::fmx<short2, double2, ushort2, int2, op>, device::fmx<short3, double3, ushort3, int3, op>, device::fmx<short4, double4, ushort4, int4, op>  },
                    { device::fmx<short, double, ushort, float, op>, device::fmx<short2, double2, ushort2, float2, op>, device::fmx<short3, double3, ushort3, float3, op>, device::fmx<short4, double4, ushort4, float4, op>  },
                    { device::fmx<short, double, ushort, double, op>, device::fmx<short2, double2, ushort2, double2, op>, device::fmx<short3, double3, ushort3, double3, op>, device::fmx<short4, double4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<short, double, short, uchar, op>, device::fmx<short2, double2, short2, uchar2, op>, device::fmx<short3, double3, short3, uchar3, op>, device::fmx<short4, double4, short4, uchar4, op>  },
                    { device::fmx<short, double, short, schar, op>, device::fmx<short2, double2, short2, char2, op>, device::fmx<short3, double3, short3, char3, op>, device::fmx<short4, double4, short4, char4, op>  },
                    { device::fmx<short, double, short, ushort, op>, device::fmx<short2, double2, short2, ushort2, op>, device::fmx<short3, double3, short3, ushort3, op>, device::fmx<short4, double4, short4, ushort4, op>  },
                    { device::fmx<short, double, short, short, op>, device::fmx<short2, double2, short2, short2, op>, device::fmx<short3, double3, short3, short3, op>, device::fmx<short4, double4, short4, short4, op>  },
                    { device::fmx<short, double, short, int, op>, device::fmx<short2, double2, short2, int2, op>, device::fmx<short3, double3, short3, int3, op>, device::fmx<short4, double4, short4, int4, op>  },
                    { device::fmx<short, double, short, float, op>, device::fmx<short2, double2, short2, float2, op>, device::fmx<short3, double3, short3, float3, op>, device::fmx<short4, double4, short4, float4, op>  },
                    { device::fmx<short, double, short, double, op>, device::fmx<short2, double2, short2, double2, op>, device::fmx<short3, double3, short3, double3, op>, device::fmx<short4, double4, short4, double4, op>  },
                },
                {
                    { device::fmx<short, double, int, uchar, op>, device::fmx<short2, double2, int2, uchar2, op>, device::fmx<short3, double3, int3, uchar3, op>, device::fmx<short4, double4, int4, uchar4, op>  },
                    { device::fmx<short, double, int, schar, op>, device::fmx<short2, double2, int2, char2, op>, device::fmx<short3, double3, int3, char3, op>, device::fmx<short4, double4, int4, char4, op>  },
                    { device::fmx<short, double, int, ushort, op>, device::fmx<short2, double2, int2, ushort2, op>, device::fmx<short3, double3, int3, ushort3, op>, device::fmx<short4, double4, int4, ushort4, op>  },
                    { device::fmx<short, double, int, short, op>, device::fmx<short2, double2, int2, short2, op>, device::fmx<short3, double3, int3, short3, op>, device::fmx<short4, double4, int4, short4, op>  },
                    { device::fmx<short, double, int, int, op>, device::fmx<short2, double2, int2, int2, op>, device::fmx<short3, double3, int3, int3, op>, device::fmx<short4, double4, int4, int4, op>  },
                    { device::fmx<short, double, int, float, op>, device::fmx<short2, double2, int2, float2, op>, device::fmx<short3, double3, int3, float3, op>, device::fmx<short4, double4, int4, float4, op>  },
                    { device::fmx<short, double, int, double, op>, device::fmx<short2, double2, int2, double2, op>, device::fmx<short3, double3, int3, double3, op>, device::fmx<short4, double4, int4, double4, op>  },
                },
                {
                    { device::fmx<short, double, float, uchar, op>, device::fmx<short2, double2, float2, uchar2, op>, device::fmx<short3, double3, float3, uchar3, op>, device::fmx<short4, double4, float4, uchar4, op>  },
                    { device::fmx<short, double, float, schar, op>, device::fmx<short2, double2, float2, char2, op>, device::fmx<short3, double3, float3, char3, op>, device::fmx<short4, double4, float4, char4, op>  },
                    { device::fmx<short, double, float, ushort, op>, device::fmx<short2, double2, float2, ushort2, op>, device::fmx<short3, double3, float3, ushort3, op>, device::fmx<short4, double4, float4, ushort4, op>  },
                    { device::fmx<short, double, float, short, op>, device::fmx<short2, double2, float2, short2, op>, device::fmx<short3, double3, float3, short3, op>, device::fmx<short4, double4, float4, short4, op>  },
                    { device::fmx<short, double, float, int, op>, device::fmx<short2, double2, float2, int2, op>, device::fmx<short3, double3, float3, int3, op>, device::fmx<short4, double4, float4, int4, op>  },
                    { device::fmx<short, double, float, float, op>, device::fmx<short2, double2, float2, float2, op>, device::fmx<short3, double3, float3, float3, op>, device::fmx<short4, double4, float4, float4, op>  },
                    { device::fmx<short, double, float, double, op>, device::fmx<short2, double2, float2, double2, op>, device::fmx<short3, double3, float3, double3, op>, device::fmx<short4, double4, float4, double4, op>  },
                },
                {
                    { device::fmx<short, double, double, uchar, op>, device::fmx<short2, double2, double2, uchar2, op>, device::fmx<short3, double3, double3, uchar3, op>, device::fmx<short4, double4, double4, uchar4, op>  },
                    { device::fmx<short, double, double, schar, op>, device::fmx<short2, double2, double2, char2, op>, device::fmx<short3, double3, double3, char3, op>, device::fmx<short4, double4, double4, char4, op>  },
                    { device::fmx<short, double, double, ushort, op>, device::fmx<short2, double2, double2, ushort2, op>, device::fmx<short3, double3, double3, ushort3, op>, device::fmx<short4, double4, double4, ushort4, op>  },
                    { device::fmx<short, double, double, short, op>, device::fmx<short2, double2, double2, short2, op>, device::fmx<short3, double3, double3, short3, op>, device::fmx<short4, double4, double4, short4, op>  },
                    { device::fmx<short, double, double, int, op>, device::fmx<short2, double2, double2, int2, op>, device::fmx<short3, double3, double3, int3, op>, device::fmx<short4, double4, double4, int4, op>  },
                    { device::fmx<short, double, double, float, op>, device::fmx<short2, double2, double2, float2, op>, device::fmx<short3, double3, double3, float3, op>, device::fmx<short4, double4, double4, float4, op>  },
                    { device::fmx<short, double, double, double, op>, device::fmx<short2, double2, double2, double2, op>, device::fmx<short3, double3, double3, double3, op>, device::fmx<short4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { device::fmx<int, uchar, uchar, uchar, op>, device::fmx<int2, uchar2, uchar2, uchar2, op>, device::fmx<int3, uchar3, uchar3, uchar3, op>, device::fmx<int4, uchar4, uchar4, uchar4, op>  },
                    { device::fmx<int, uchar, uchar, schar, op>, device::fmx<int2, uchar2, uchar2, char2, op>, device::fmx<int3, uchar3, uchar3, char3, op>, device::fmx<int4, uchar4, uchar4, char4, op>  },
                    { device::fmx<int, uchar, uchar, ushort, op>, device::fmx<int2, uchar2, uchar2, ushort2, op>, device::fmx<int3, uchar3, uchar3, ushort3, op>, device::fmx<int4, uchar4, uchar4, ushort4, op>  },
                    { device::fmx<int, uchar, uchar, short, op>, device::fmx<int2, uchar2, uchar2, short2, op>, device::fmx<int3, uchar3, uchar3, short3, op>, device::fmx<int4, uchar4, uchar4, short4, op>  },
                    { device::fmx<int, uchar, uchar, int, op>, device::fmx<int2, uchar2, uchar2, int2, op>, device::fmx<int3, uchar3, uchar3, int3, op>, device::fmx<int4, uchar4, uchar4, int4, op>  },
                    { device::fmx<int, uchar, uchar, float, op>, device::fmx<int2, uchar2, uchar2, float2, op>, device::fmx<int3, uchar3, uchar3, float3, op>, device::fmx<int4, uchar4, uchar4, float4, op>  },
                    { device::fmx<int, uchar, uchar, double, op>, device::fmx<int2, uchar2, uchar2, double2, op>, device::fmx<int3, uchar3, uchar3, double3, op>, device::fmx<int4, uchar4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<int, uchar, schar, uchar, op>, device::fmx<int2, uchar2, char2, uchar2, op>, device::fmx<int3, uchar3, char3, uchar3, op>, device::fmx<int4, uchar4, char4, uchar4, op>  },
                    { device::fmx<int, uchar, schar, schar, op>, device::fmx<int2, uchar2, char2, char2, op>, device::fmx<int3, uchar3, char3, char3, op>, device::fmx<int4, uchar4, char4, char4, op>  },
                    { device::fmx<int, uchar, schar, ushort, op>, device::fmx<int2, uchar2, char2, ushort2, op>, device::fmx<int3, uchar3, char3, ushort3, op>, device::fmx<int4, uchar4, char4, ushort4, op>  },
                    { device::fmx<int, uchar, schar, short, op>, device::fmx<int2, uchar2, char2, short2, op>, device::fmx<int3, uchar3, char3, short3, op>, device::fmx<int4, uchar4, char4, short4, op>  },
                    { device::fmx<int, uchar, schar, int, op>, device::fmx<int2, uchar2, char2, int2, op>, device::fmx<int3, uchar3, char3, int3, op>, device::fmx<int4, uchar4, char4, int4, op>  },
                    { device::fmx<int, uchar, schar, float, op>, device::fmx<int2, uchar2, char2, float2, op>, device::fmx<int3, uchar3, char3, float3, op>, device::fmx<int4, uchar4, char4, float4, op>  },
                    { device::fmx<int, uchar, schar, double, op>, device::fmx<int2, uchar2, char2, double2, op>, device::fmx<int3, uchar3, char3, double3, op>, device::fmx<int4, uchar4, char4, double4, op>  },
                },
                {
                    { device::fmx<int, uchar, ushort, uchar, op>, device::fmx<int2, uchar2, ushort2, uchar2, op>, device::fmx<int3, uchar3, ushort3, uchar3, op>, device::fmx<int4, uchar4, ushort4, uchar4, op>  },
                    { device::fmx<int, uchar, ushort, schar, op>, device::fmx<int2, uchar2, ushort2, char2, op>, device::fmx<int3, uchar3, ushort3, char3, op>, device::fmx<int4, uchar4, ushort4, char4, op>  },
                    { device::fmx<int, uchar, ushort, ushort, op>, device::fmx<int2, uchar2, ushort2, ushort2, op>, device::fmx<int3, uchar3, ushort3, ushort3, op>, device::fmx<int4, uchar4, ushort4, ushort4, op>  },
                    { device::fmx<int, uchar, ushort, short, op>, device::fmx<int2, uchar2, ushort2, short2, op>, device::fmx<int3, uchar3, ushort3, short3, op>, device::fmx<int4, uchar4, ushort4, short4, op>  },
                    { device::fmx<int, uchar, ushort, int, op>, device::fmx<int2, uchar2, ushort2, int2, op>, device::fmx<int3, uchar3, ushort3, int3, op>, device::fmx<int4, uchar4, ushort4, int4, op>  },
                    { device::fmx<int, uchar, ushort, float, op>, device::fmx<int2, uchar2, ushort2, float2, op>, device::fmx<int3, uchar3, ushort3, float3, op>, device::fmx<int4, uchar4, ushort4, float4, op>  },
                    { device::fmx<int, uchar, ushort, double, op>, device::fmx<int2, uchar2, ushort2, double2, op>, device::fmx<int3, uchar3, ushort3, double3, op>, device::fmx<int4, uchar4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<int, uchar, short, uchar, op>, device::fmx<int2, uchar2, short2, uchar2, op>, device::fmx<int3, uchar3, short3, uchar3, op>, device::fmx<int4, uchar4, short4, uchar4, op>  },
                    { device::fmx<int, uchar, short, schar, op>, device::fmx<int2, uchar2, short2, char2, op>, device::fmx<int3, uchar3, short3, char3, op>, device::fmx<int4, uchar4, short4, char4, op>  },
                    { device::fmx<int, uchar, short, ushort, op>, device::fmx<int2, uchar2, short2, ushort2, op>, device::fmx<int3, uchar3, short3, ushort3, op>, device::fmx<int4, uchar4, short4, ushort4, op>  },
                    { device::fmx<int, uchar, short, short, op>, device::fmx<int2, uchar2, short2, short2, op>, device::fmx<int3, uchar3, short3, short3, op>, device::fmx<int4, uchar4, short4, short4, op>  },
                    { device::fmx<int, uchar, short, int, op>, device::fmx<int2, uchar2, short2, int2, op>, device::fmx<int3, uchar3, short3, int3, op>, device::fmx<int4, uchar4, short4, int4, op>  },
                    { device::fmx<int, uchar, short, float, op>, device::fmx<int2, uchar2, short2, float2, op>, device::fmx<int3, uchar3, short3, float3, op>, device::fmx<int4, uchar4, short4, float4, op>  },
                    { device::fmx<int, uchar, short, double, op>, device::fmx<int2, uchar2, short2, double2, op>, device::fmx<int3, uchar3, short3, double3, op>, device::fmx<int4, uchar4, short4, double4, op>  },
                },
                {
                    { device::fmx<int, uchar, int, uchar, op>, device::fmx<int2, uchar2, int2, uchar2, op>, device::fmx<int3, uchar3, int3, uchar3, op>, device::fmx<int4, uchar4, int4, uchar4, op>  },
                    { device::fmx<int, uchar, int, schar, op>, device::fmx<int2, uchar2, int2, char2, op>, device::fmx<int3, uchar3, int3, char3, op>, device::fmx<int4, uchar4, int4, char4, op>  },
                    { device::fmx<int, uchar, int, ushort, op>, device::fmx<int2, uchar2, int2, ushort2, op>, device::fmx<int3, uchar3, int3, ushort3, op>, device::fmx<int4, uchar4, int4, ushort4, op>  },
                    { device::fmx<int, uchar, int, short, op>, device::fmx<int2, uchar2, int2, short2, op>, device::fmx<int3, uchar3, int3, short3, op>, device::fmx<int4, uchar4, int4, short4, op>  },
                    { device::fmx<int, uchar, int, int, op>, device::fmx<int2, uchar2, int2, int2, op>, device::fmx<int3, uchar3, int3, int3, op>, device::fmx<int4, uchar4, int4, int4, op>  },
                    { device::fmx<int, uchar, int, float, op>, device::fmx<int2, uchar2, int2, float2, op>, device::fmx<int3, uchar3, int3, float3, op>, device::fmx<int4, uchar4, int4, float4, op>  },
                    { device::fmx<int, uchar, int, double, op>, device::fmx<int2, uchar2, int2, double2, op>, device::fmx<int3, uchar3, int3, double3, op>, device::fmx<int4, uchar4, int4, double4, op>  },
                },
                {
                    { device::fmx<int, uchar, float, uchar, op>, device::fmx<int2, uchar2, float2, uchar2, op>, device::fmx<int3, uchar3, float3, uchar3, op>, device::fmx<int4, uchar4, float4, uchar4, op>  },
                    { device::fmx<int, uchar, float, schar, op>, device::fmx<int2, uchar2, float2, char2, op>, device::fmx<int3, uchar3, float3, char3, op>, device::fmx<int4, uchar4, float4, char4, op>  },
                    { device::fmx<int, uchar, float, ushort, op>, device::fmx<int2, uchar2, float2, ushort2, op>, device::fmx<int3, uchar3, float3, ushort3, op>, device::fmx<int4, uchar4, float4, ushort4, op>  },
                    { device::fmx<int, uchar, float, short, op>, device::fmx<int2, uchar2, float2, short2, op>, device::fmx<int3, uchar3, float3, short3, op>, device::fmx<int4, uchar4, float4, short4, op>  },
                    { device::fmx<int, uchar, float, int, op>, device::fmx<int2, uchar2, float2, int2, op>, device::fmx<int3, uchar3, float3, int3, op>, device::fmx<int4, uchar4, float4, int4, op>  },
                    { device::fmx<int, uchar, float, float, op>, device::fmx<int2, uchar2, float2, float2, op>, device::fmx<int3, uchar3, float3, float3, op>, device::fmx<int4, uchar4, float4, float4, op>  },
                    { device::fmx<int, uchar, float, double, op>, device::fmx<int2, uchar2, float2, double2, op>, device::fmx<int3, uchar3, float3, double3, op>, device::fmx<int4, uchar4, float4, double4, op>  },
                },
                {
                    { device::fmx<int, uchar, double, uchar, op>, device::fmx<int2, uchar2, double2, uchar2, op>, device::fmx<int3, uchar3, double3, uchar3, op>, device::fmx<int4, uchar4, double4, uchar4, op>  },
                    { device::fmx<int, uchar, double, schar, op>, device::fmx<int2, uchar2, double2, char2, op>, device::fmx<int3, uchar3, double3, char3, op>, device::fmx<int4, uchar4, double4, char4, op>  },
                    { device::fmx<int, uchar, double, ushort, op>, device::fmx<int2, uchar2, double2, ushort2, op>, device::fmx<int3, uchar3, double3, ushort3, op>, device::fmx<int4, uchar4, double4, ushort4, op>  },
                    { device::fmx<int, uchar, double, short, op>, device::fmx<int2, uchar2, double2, short2, op>, device::fmx<int3, uchar3, double3, short3, op>, device::fmx<int4, uchar4, double4, short4, op>  },
                    { device::fmx<int, uchar, double, int, op>, device::fmx<int2, uchar2, double2, int2, op>, device::fmx<int3, uchar3, double3, int3, op>, device::fmx<int4, uchar4, double4, int4, op>  },
                    { device::fmx<int, uchar, double, float, op>, device::fmx<int2, uchar2, double2, float2, op>, device::fmx<int3, uchar3, double3, float3, op>, device::fmx<int4, uchar4, double4, float4, op>  },
                    { device::fmx<int, uchar, double, double, op>, device::fmx<int2, uchar2, double2, double2, op>, device::fmx<int3, uchar3, double3, double3, op>, device::fmx<int4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<int, schar, uchar, uchar, op>, device::fmx<int2, char2, uchar2, uchar2, op>, device::fmx<int3, char3, uchar3, uchar3, op>, device::fmx<int4, char4, uchar4, uchar4, op>  },
                    { device::fmx<int, schar, uchar, schar, op>, device::fmx<int2, char2, uchar2, char2, op>, device::fmx<int3, char3, uchar3, char3, op>, device::fmx<int4, char4, uchar4, char4, op>  },
                    { device::fmx<int, schar, uchar, ushort, op>, device::fmx<int2, char2, uchar2, ushort2, op>, device::fmx<int3, char3, uchar3, ushort3, op>, device::fmx<int4, char4, uchar4, ushort4, op>  },
                    { device::fmx<int, schar, uchar, short, op>, device::fmx<int2, char2, uchar2, short2, op>, device::fmx<int3, char3, uchar3, short3, op>, device::fmx<int4, char4, uchar4, short4, op>  },
                    { device::fmx<int, schar, uchar, int, op>, device::fmx<int2, char2, uchar2, int2, op>, device::fmx<int3, char3, uchar3, int3, op>, device::fmx<int4, char4, uchar4, int4, op>  },
                    { device::fmx<int, schar, uchar, float, op>, device::fmx<int2, char2, uchar2, float2, op>, device::fmx<int3, char3, uchar3, float3, op>, device::fmx<int4, char4, uchar4, float4, op>  },
                    { device::fmx<int, schar, uchar, double, op>, device::fmx<int2, char2, uchar2, double2, op>, device::fmx<int3, char3, uchar3, double3, op>, device::fmx<int4, char4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<int, schar, schar, uchar, op>, device::fmx<int2, char2, char2, uchar2, op>, device::fmx<int3, char3, char3, uchar3, op>, device::fmx<int4, char4, char4, uchar4, op>  },
                    { device::fmx<int, schar, schar, schar, op>, device::fmx<int2, char2, char2, char2, op>, device::fmx<int3, char3, char3, char3, op>, device::fmx<int4, char4, char4, char4, op>  },
                    { device::fmx<int, schar, schar, ushort, op>, device::fmx<int2, char2, char2, ushort2, op>, device::fmx<int3, char3, char3, ushort3, op>, device::fmx<int4, char4, char4, ushort4, op>  },
                    { device::fmx<int, schar, schar, short, op>, device::fmx<int2, char2, char2, short2, op>, device::fmx<int3, char3, char3, short3, op>, device::fmx<int4, char4, char4, short4, op>  },
                    { device::fmx<int, schar, schar, int, op>, device::fmx<int2, char2, char2, int2, op>, device::fmx<int3, char3, char3, int3, op>, device::fmx<int4, char4, char4, int4, op>  },
                    { device::fmx<int, schar, schar, float, op>, device::fmx<int2, char2, char2, float2, op>, device::fmx<int3, char3, char3, float3, op>, device::fmx<int4, char4, char4, float4, op>  },
                    { device::fmx<int, schar, schar, double, op>, device::fmx<int2, char2, char2, double2, op>, device::fmx<int3, char3, char3, double3, op>, device::fmx<int4, char4, char4, double4, op>  },
                },
                {
                    { device::fmx<int, schar, ushort, uchar, op>, device::fmx<int2, char2, ushort2, uchar2, op>, device::fmx<int3, char3, ushort3, uchar3, op>, device::fmx<int4, char4, ushort4, uchar4, op>  },
                    { device::fmx<int, schar, ushort, schar, op>, device::fmx<int2, char2, ushort2, char2, op>, device::fmx<int3, char3, ushort3, char3, op>, device::fmx<int4, char4, ushort4, char4, op>  },
                    { device::fmx<int, schar, ushort, ushort, op>, device::fmx<int2, char2, ushort2, ushort2, op>, device::fmx<int3, char3, ushort3, ushort3, op>, device::fmx<int4, char4, ushort4, ushort4, op>  },
                    { device::fmx<int, schar, ushort, short, op>, device::fmx<int2, char2, ushort2, short2, op>, device::fmx<int3, char3, ushort3, short3, op>, device::fmx<int4, char4, ushort4, short4, op>  },
                    { device::fmx<int, schar, ushort, int, op>, device::fmx<int2, char2, ushort2, int2, op>, device::fmx<int3, char3, ushort3, int3, op>, device::fmx<int4, char4, ushort4, int4, op>  },
                    { device::fmx<int, schar, ushort, float, op>, device::fmx<int2, char2, ushort2, float2, op>, device::fmx<int3, char3, ushort3, float3, op>, device::fmx<int4, char4, ushort4, float4, op>  },
                    { device::fmx<int, schar, ushort, double, op>, device::fmx<int2, char2, ushort2, double2, op>, device::fmx<int3, char3, ushort3, double3, op>, device::fmx<int4, char4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<int, schar, short, uchar, op>, device::fmx<int2, char2, short2, uchar2, op>, device::fmx<int3, char3, short3, uchar3, op>, device::fmx<int4, char4, short4, uchar4, op>  },
                    { device::fmx<int, schar, short, schar, op>, device::fmx<int2, char2, short2, char2, op>, device::fmx<int3, char3, short3, char3, op>, device::fmx<int4, char4, short4, char4, op>  },
                    { device::fmx<int, schar, short, ushort, op>, device::fmx<int2, char2, short2, ushort2, op>, device::fmx<int3, char3, short3, ushort3, op>, device::fmx<int4, char4, short4, ushort4, op>  },
                    { device::fmx<int, schar, short, short, op>, device::fmx<int2, char2, short2, short2, op>, device::fmx<int3, char3, short3, short3, op>, device::fmx<int4, char4, short4, short4, op>  },
                    { device::fmx<int, schar, short, int, op>, device::fmx<int2, char2, short2, int2, op>, device::fmx<int3, char3, short3, int3, op>, device::fmx<int4, char4, short4, int4, op>  },
                    { device::fmx<int, schar, short, float, op>, device::fmx<int2, char2, short2, float2, op>, device::fmx<int3, char3, short3, float3, op>, device::fmx<int4, char4, short4, float4, op>  },
                    { device::fmx<int, schar, short, double, op>, device::fmx<int2, char2, short2, double2, op>, device::fmx<int3, char3, short3, double3, op>, device::fmx<int4, char4, short4, double4, op>  },
                },
                {
                    { device::fmx<int, schar, int, uchar, op>, device::fmx<int2, char2, int2, uchar2, op>, device::fmx<int3, char3, int3, uchar3, op>, device::fmx<int4, char4, int4, uchar4, op>  },
                    { device::fmx<int, schar, int, schar, op>, device::fmx<int2, char2, int2, char2, op>, device::fmx<int3, char3, int3, char3, op>, device::fmx<int4, char4, int4, char4, op>  },
                    { device::fmx<int, schar, int, ushort, op>, device::fmx<int2, char2, int2, ushort2, op>, device::fmx<int3, char3, int3, ushort3, op>, device::fmx<int4, char4, int4, ushort4, op>  },
                    { device::fmx<int, schar, int, short, op>, device::fmx<int2, char2, int2, short2, op>, device::fmx<int3, char3, int3, short3, op>, device::fmx<int4, char4, int4, short4, op>  },
                    { device::fmx<int, schar, int, int, op>, device::fmx<int2, char2, int2, int2, op>, device::fmx<int3, char3, int3, int3, op>, device::fmx<int4, char4, int4, int4, op>  },
                    { device::fmx<int, schar, int, float, op>, device::fmx<int2, char2, int2, float2, op>, device::fmx<int3, char3, int3, float3, op>, device::fmx<int4, char4, int4, float4, op>  },
                    { device::fmx<int, schar, int, double, op>, device::fmx<int2, char2, int2, double2, op>, device::fmx<int3, char3, int3, double3, op>, device::fmx<int4, char4, int4, double4, op>  },
                },
                {
                    { device::fmx<int, schar, float, uchar, op>, device::fmx<int2, char2, float2, uchar2, op>, device::fmx<int3, char3, float3, uchar3, op>, device::fmx<int4, char4, float4, uchar4, op>  },
                    { device::fmx<int, schar, float, schar, op>, device::fmx<int2, char2, float2, char2, op>, device::fmx<int3, char3, float3, char3, op>, device::fmx<int4, char4, float4, char4, op>  },
                    { device::fmx<int, schar, float, ushort, op>, device::fmx<int2, char2, float2, ushort2, op>, device::fmx<int3, char3, float3, ushort3, op>, device::fmx<int4, char4, float4, ushort4, op>  },
                    { device::fmx<int, schar, float, short, op>, device::fmx<int2, char2, float2, short2, op>, device::fmx<int3, char3, float3, short3, op>, device::fmx<int4, char4, float4, short4, op>  },
                    { device::fmx<int, schar, float, int, op>, device::fmx<int2, char2, float2, int2, op>, device::fmx<int3, char3, float3, int3, op>, device::fmx<int4, char4, float4, int4, op>  },
                    { device::fmx<int, schar, float, float, op>, device::fmx<int2, char2, float2, float2, op>, device::fmx<int3, char3, float3, float3, op>, device::fmx<int4, char4, float4, float4, op>  },
                    { device::fmx<int, schar, float, double, op>, device::fmx<int2, char2, float2, double2, op>, device::fmx<int3, char3, float3, double3, op>, device::fmx<int4, char4, float4, double4, op>  },
                },
                {
                    { device::fmx<int, schar, double, uchar, op>, device::fmx<int2, char2, double2, uchar2, op>, device::fmx<int3, char3, double3, uchar3, op>, device::fmx<int4, char4, double4, uchar4, op>  },
                    { device::fmx<int, schar, double, schar, op>, device::fmx<int2, char2, double2, char2, op>, device::fmx<int3, char3, double3, char3, op>, device::fmx<int4, char4, double4, char4, op>  },
                    { device::fmx<int, schar, double, ushort, op>, device::fmx<int2, char2, double2, ushort2, op>, device::fmx<int3, char3, double3, ushort3, op>, device::fmx<int4, char4, double4, ushort4, op>  },
                    { device::fmx<int, schar, double, short, op>, device::fmx<int2, char2, double2, short2, op>, device::fmx<int3, char3, double3, short3, op>, device::fmx<int4, char4, double4, short4, op>  },
                    { device::fmx<int, schar, double, int, op>, device::fmx<int2, char2, double2, int2, op>, device::fmx<int3, char3, double3, int3, op>, device::fmx<int4, char4, double4, int4, op>  },
                    { device::fmx<int, schar, double, float, op>, device::fmx<int2, char2, double2, float2, op>, device::fmx<int3, char3, double3, float3, op>, device::fmx<int4, char4, double4, float4, op>  },
                    { device::fmx<int, schar, double, double, op>, device::fmx<int2, char2, double2, double2, op>, device::fmx<int3, char3, double3, double3, op>, device::fmx<int4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<int, ushort, uchar, uchar, op>, device::fmx<int2, ushort2, uchar2, uchar2, op>, device::fmx<int3, ushort3, uchar3, uchar3, op>, device::fmx<int4, ushort4, uchar4, uchar4, op>  },
                    { device::fmx<int, ushort, uchar, schar, op>, device::fmx<int2, ushort2, uchar2, char2, op>, device::fmx<int3, ushort3, uchar3, char3, op>, device::fmx<int4, ushort4, uchar4, char4, op>  },
                    { device::fmx<int, ushort, uchar, ushort, op>, device::fmx<int2, ushort2, uchar2, ushort2, op>, device::fmx<int3, ushort3, uchar3, ushort3, op>, device::fmx<int4, ushort4, uchar4, ushort4, op>  },
                    { device::fmx<int, ushort, uchar, short, op>, device::fmx<int2, ushort2, uchar2, short2, op>, device::fmx<int3, ushort3, uchar3, short3, op>, device::fmx<int4, ushort4, uchar4, short4, op>  },
                    { device::fmx<int, ushort, uchar, int, op>, device::fmx<int2, ushort2, uchar2, int2, op>, device::fmx<int3, ushort3, uchar3, int3, op>, device::fmx<int4, ushort4, uchar4, int4, op>  },
                    { device::fmx<int, ushort, uchar, float, op>, device::fmx<int2, ushort2, uchar2, float2, op>, device::fmx<int3, ushort3, uchar3, float3, op>, device::fmx<int4, ushort4, uchar4, float4, op>  },
                    { device::fmx<int, ushort, uchar, double, op>, device::fmx<int2, ushort2, uchar2, double2, op>, device::fmx<int3, ushort3, uchar3, double3, op>, device::fmx<int4, ushort4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<int, ushort, schar, uchar, op>, device::fmx<int2, ushort2, char2, uchar2, op>, device::fmx<int3, ushort3, char3, uchar3, op>, device::fmx<int4, ushort4, char4, uchar4, op>  },
                    { device::fmx<int, ushort, schar, schar, op>, device::fmx<int2, ushort2, char2, char2, op>, device::fmx<int3, ushort3, char3, char3, op>, device::fmx<int4, ushort4, char4, char4, op>  },
                    { device::fmx<int, ushort, schar, ushort, op>, device::fmx<int2, ushort2, char2, ushort2, op>, device::fmx<int3, ushort3, char3, ushort3, op>, device::fmx<int4, ushort4, char4, ushort4, op>  },
                    { device::fmx<int, ushort, schar, short, op>, device::fmx<int2, ushort2, char2, short2, op>, device::fmx<int3, ushort3, char3, short3, op>, device::fmx<int4, ushort4, char4, short4, op>  },
                    { device::fmx<int, ushort, schar, int, op>, device::fmx<int2, ushort2, char2, int2, op>, device::fmx<int3, ushort3, char3, int3, op>, device::fmx<int4, ushort4, char4, int4, op>  },
                    { device::fmx<int, ushort, schar, float, op>, device::fmx<int2, ushort2, char2, float2, op>, device::fmx<int3, ushort3, char3, float3, op>, device::fmx<int4, ushort4, char4, float4, op>  },
                    { device::fmx<int, ushort, schar, double, op>, device::fmx<int2, ushort2, char2, double2, op>, device::fmx<int3, ushort3, char3, double3, op>, device::fmx<int4, ushort4, char4, double4, op>  },
                },
                {
                    { device::fmx<int, ushort, ushort, uchar, op>, device::fmx<int2, ushort2, ushort2, uchar2, op>, device::fmx<int3, ushort3, ushort3, uchar3, op>, device::fmx<int4, ushort4, ushort4, uchar4, op>  },
                    { device::fmx<int, ushort, ushort, schar, op>, device::fmx<int2, ushort2, ushort2, char2, op>, device::fmx<int3, ushort3, ushort3, char3, op>, device::fmx<int4, ushort4, ushort4, char4, op>  },
                    { device::fmx<int, ushort, ushort, ushort, op>, device::fmx<int2, ushort2, ushort2, ushort2, op>, device::fmx<int3, ushort3, ushort3, ushort3, op>, device::fmx<int4, ushort4, ushort4, ushort4, op>  },
                    { device::fmx<int, ushort, ushort, short, op>, device::fmx<int2, ushort2, ushort2, short2, op>, device::fmx<int3, ushort3, ushort3, short3, op>, device::fmx<int4, ushort4, ushort4, short4, op>  },
                    { device::fmx<int, ushort, ushort, int, op>, device::fmx<int2, ushort2, ushort2, int2, op>, device::fmx<int3, ushort3, ushort3, int3, op>, device::fmx<int4, ushort4, ushort4, int4, op>  },
                    { device::fmx<int, ushort, ushort, float, op>, device::fmx<int2, ushort2, ushort2, float2, op>, device::fmx<int3, ushort3, ushort3, float3, op>, device::fmx<int4, ushort4, ushort4, float4, op>  },
                    { device::fmx<int, ushort, ushort, double, op>, device::fmx<int2, ushort2, ushort2, double2, op>, device::fmx<int3, ushort3, ushort3, double3, op>, device::fmx<int4, ushort4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<int, ushort, short, uchar, op>, device::fmx<int2, ushort2, short2, uchar2, op>, device::fmx<int3, ushort3, short3, uchar3, op>, device::fmx<int4, ushort4, short4, uchar4, op>  },
                    { device::fmx<int, ushort, short, schar, op>, device::fmx<int2, ushort2, short2, char2, op>, device::fmx<int3, ushort3, short3, char3, op>, device::fmx<int4, ushort4, short4, char4, op>  },
                    { device::fmx<int, ushort, short, ushort, op>, device::fmx<int2, ushort2, short2, ushort2, op>, device::fmx<int3, ushort3, short3, ushort3, op>, device::fmx<int4, ushort4, short4, ushort4, op>  },
                    { device::fmx<int, ushort, short, short, op>, device::fmx<int2, ushort2, short2, short2, op>, device::fmx<int3, ushort3, short3, short3, op>, device::fmx<int4, ushort4, short4, short4, op>  },
                    { device::fmx<int, ushort, short, int, op>, device::fmx<int2, ushort2, short2, int2, op>, device::fmx<int3, ushort3, short3, int3, op>, device::fmx<int4, ushort4, short4, int4, op>  },
                    { device::fmx<int, ushort, short, float, op>, device::fmx<int2, ushort2, short2, float2, op>, device::fmx<int3, ushort3, short3, float3, op>, device::fmx<int4, ushort4, short4, float4, op>  },
                    { device::fmx<int, ushort, short, double, op>, device::fmx<int2, ushort2, short2, double2, op>, device::fmx<int3, ushort3, short3, double3, op>, device::fmx<int4, ushort4, short4, double4, op>  },
                },
                {
                    { device::fmx<int, ushort, int, uchar, op>, device::fmx<int2, ushort2, int2, uchar2, op>, device::fmx<int3, ushort3, int3, uchar3, op>, device::fmx<int4, ushort4, int4, uchar4, op>  },
                    { device::fmx<int, ushort, int, schar, op>, device::fmx<int2, ushort2, int2, char2, op>, device::fmx<int3, ushort3, int3, char3, op>, device::fmx<int4, ushort4, int4, char4, op>  },
                    { device::fmx<int, ushort, int, ushort, op>, device::fmx<int2, ushort2, int2, ushort2, op>, device::fmx<int3, ushort3, int3, ushort3, op>, device::fmx<int4, ushort4, int4, ushort4, op>  },
                    { device::fmx<int, ushort, int, short, op>, device::fmx<int2, ushort2, int2, short2, op>, device::fmx<int3, ushort3, int3, short3, op>, device::fmx<int4, ushort4, int4, short4, op>  },
                    { device::fmx<int, ushort, int, int, op>, device::fmx<int2, ushort2, int2, int2, op>, device::fmx<int3, ushort3, int3, int3, op>, device::fmx<int4, ushort4, int4, int4, op>  },
                    { device::fmx<int, ushort, int, float, op>, device::fmx<int2, ushort2, int2, float2, op>, device::fmx<int3, ushort3, int3, float3, op>, device::fmx<int4, ushort4, int4, float4, op>  },
                    { device::fmx<int, ushort, int, double, op>, device::fmx<int2, ushort2, int2, double2, op>, device::fmx<int3, ushort3, int3, double3, op>, device::fmx<int4, ushort4, int4, double4, op>  },
                },
                {
                    { device::fmx<int, ushort, float, uchar, op>, device::fmx<int2, ushort2, float2, uchar2, op>, device::fmx<int3, ushort3, float3, uchar3, op>, device::fmx<int4, ushort4, float4, uchar4, op>  },
                    { device::fmx<int, ushort, float, schar, op>, device::fmx<int2, ushort2, float2, char2, op>, device::fmx<int3, ushort3, float3, char3, op>, device::fmx<int4, ushort4, float4, char4, op>  },
                    { device::fmx<int, ushort, float, ushort, op>, device::fmx<int2, ushort2, float2, ushort2, op>, device::fmx<int3, ushort3, float3, ushort3, op>, device::fmx<int4, ushort4, float4, ushort4, op>  },
                    { device::fmx<int, ushort, float, short, op>, device::fmx<int2, ushort2, float2, short2, op>, device::fmx<int3, ushort3, float3, short3, op>, device::fmx<int4, ushort4, float4, short4, op>  },
                    { device::fmx<int, ushort, float, int, op>, device::fmx<int2, ushort2, float2, int2, op>, device::fmx<int3, ushort3, float3, int3, op>, device::fmx<int4, ushort4, float4, int4, op>  },
                    { device::fmx<int, ushort, float, float, op>, device::fmx<int2, ushort2, float2, float2, op>, device::fmx<int3, ushort3, float3, float3, op>, device::fmx<int4, ushort4, float4, float4, op>  },
                    { device::fmx<int, ushort, float, double, op>, device::fmx<int2, ushort2, float2, double2, op>, device::fmx<int3, ushort3, float3, double3, op>, device::fmx<int4, ushort4, float4, double4, op>  },
                },
                {
                    { device::fmx<int, ushort, double, uchar, op>, device::fmx<int2, ushort2, double2, uchar2, op>, device::fmx<int3, ushort3, double3, uchar3, op>, device::fmx<int4, ushort4, double4, uchar4, op>  },
                    { device::fmx<int, ushort, double, schar, op>, device::fmx<int2, ushort2, double2, char2, op>, device::fmx<int3, ushort3, double3, char3, op>, device::fmx<int4, ushort4, double4, char4, op>  },
                    { device::fmx<int, ushort, double, ushort, op>, device::fmx<int2, ushort2, double2, ushort2, op>, device::fmx<int3, ushort3, double3, ushort3, op>, device::fmx<int4, ushort4, double4, ushort4, op>  },
                    { device::fmx<int, ushort, double, short, op>, device::fmx<int2, ushort2, double2, short2, op>, device::fmx<int3, ushort3, double3, short3, op>, device::fmx<int4, ushort4, double4, short4, op>  },
                    { device::fmx<int, ushort, double, int, op>, device::fmx<int2, ushort2, double2, int2, op>, device::fmx<int3, ushort3, double3, int3, op>, device::fmx<int4, ushort4, double4, int4, op>  },
                    { device::fmx<int, ushort, double, float, op>, device::fmx<int2, ushort2, double2, float2, op>, device::fmx<int3, ushort3, double3, float3, op>, device::fmx<int4, ushort4, double4, float4, op>  },
                    { device::fmx<int, ushort, double, double, op>, device::fmx<int2, ushort2, double2, double2, op>, device::fmx<int3, ushort3, double3, double3, op>, device::fmx<int4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<int, short, uchar, uchar, op>, device::fmx<int2, short2, uchar2, uchar2, op>, device::fmx<int3, short3, uchar3, uchar3, op>, device::fmx<int4, short4, uchar4, uchar4, op>  },
                    { device::fmx<int, short, uchar, schar, op>, device::fmx<int2, short2, uchar2, char2, op>, device::fmx<int3, short3, uchar3, char3, op>, device::fmx<int4, short4, uchar4, char4, op>  },
                    { device::fmx<int, short, uchar, ushort, op>, device::fmx<int2, short2, uchar2, ushort2, op>, device::fmx<int3, short3, uchar3, ushort3, op>, device::fmx<int4, short4, uchar4, ushort4, op>  },
                    { device::fmx<int, short, uchar, short, op>, device::fmx<int2, short2, uchar2, short2, op>, device::fmx<int3, short3, uchar3, short3, op>, device::fmx<int4, short4, uchar4, short4, op>  },
                    { device::fmx<int, short, uchar, int, op>, device::fmx<int2, short2, uchar2, int2, op>, device::fmx<int3, short3, uchar3, int3, op>, device::fmx<int4, short4, uchar4, int4, op>  },
                    { device::fmx<int, short, uchar, float, op>, device::fmx<int2, short2, uchar2, float2, op>, device::fmx<int3, short3, uchar3, float3, op>, device::fmx<int4, short4, uchar4, float4, op>  },
                    { device::fmx<int, short, uchar, double, op>, device::fmx<int2, short2, uchar2, double2, op>, device::fmx<int3, short3, uchar3, double3, op>, device::fmx<int4, short4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<int, short, schar, uchar, op>, device::fmx<int2, short2, char2, uchar2, op>, device::fmx<int3, short3, char3, uchar3, op>, device::fmx<int4, short4, char4, uchar4, op>  },
                    { device::fmx<int, short, schar, schar, op>, device::fmx<int2, short2, char2, char2, op>, device::fmx<int3, short3, char3, char3, op>, device::fmx<int4, short4, char4, char4, op>  },
                    { device::fmx<int, short, schar, ushort, op>, device::fmx<int2, short2, char2, ushort2, op>, device::fmx<int3, short3, char3, ushort3, op>, device::fmx<int4, short4, char4, ushort4, op>  },
                    { device::fmx<int, short, schar, short, op>, device::fmx<int2, short2, char2, short2, op>, device::fmx<int3, short3, char3, short3, op>, device::fmx<int4, short4, char4, short4, op>  },
                    { device::fmx<int, short, schar, int, op>, device::fmx<int2, short2, char2, int2, op>, device::fmx<int3, short3, char3, int3, op>, device::fmx<int4, short4, char4, int4, op>  },
                    { device::fmx<int, short, schar, float, op>, device::fmx<int2, short2, char2, float2, op>, device::fmx<int3, short3, char3, float3, op>, device::fmx<int4, short4, char4, float4, op>  },
                    { device::fmx<int, short, schar, double, op>, device::fmx<int2, short2, char2, double2, op>, device::fmx<int3, short3, char3, double3, op>, device::fmx<int4, short4, char4, double4, op>  },
                },
                {
                    { device::fmx<int, short, ushort, uchar, op>, device::fmx<int2, short2, ushort2, uchar2, op>, device::fmx<int3, short3, ushort3, uchar3, op>, device::fmx<int4, short4, ushort4, uchar4, op>  },
                    { device::fmx<int, short, ushort, schar, op>, device::fmx<int2, short2, ushort2, char2, op>, device::fmx<int3, short3, ushort3, char3, op>, device::fmx<int4, short4, ushort4, char4, op>  },
                    { device::fmx<int, short, ushort, ushort, op>, device::fmx<int2, short2, ushort2, ushort2, op>, device::fmx<int3, short3, ushort3, ushort3, op>, device::fmx<int4, short4, ushort4, ushort4, op>  },
                    { device::fmx<int, short, ushort, short, op>, device::fmx<int2, short2, ushort2, short2, op>, device::fmx<int3, short3, ushort3, short3, op>, device::fmx<int4, short4, ushort4, short4, op>  },
                    { device::fmx<int, short, ushort, int, op>, device::fmx<int2, short2, ushort2, int2, op>, device::fmx<int3, short3, ushort3, int3, op>, device::fmx<int4, short4, ushort4, int4, op>  },
                    { device::fmx<int, short, ushort, float, op>, device::fmx<int2, short2, ushort2, float2, op>, device::fmx<int3, short3, ushort3, float3, op>, device::fmx<int4, short4, ushort4, float4, op>  },
                    { device::fmx<int, short, ushort, double, op>, device::fmx<int2, short2, ushort2, double2, op>, device::fmx<int3, short3, ushort3, double3, op>, device::fmx<int4, short4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<int, short, short, uchar, op>, device::fmx<int2, short2, short2, uchar2, op>, device::fmx<int3, short3, short3, uchar3, op>, device::fmx<int4, short4, short4, uchar4, op>  },
                    { device::fmx<int, short, short, schar, op>, device::fmx<int2, short2, short2, char2, op>, device::fmx<int3, short3, short3, char3, op>, device::fmx<int4, short4, short4, char4, op>  },
                    { device::fmx<int, short, short, ushort, op>, device::fmx<int2, short2, short2, ushort2, op>, device::fmx<int3, short3, short3, ushort3, op>, device::fmx<int4, short4, short4, ushort4, op>  },
                    { device::fmx<int, short, short, short, op>, device::fmx<int2, short2, short2, short2, op>, device::fmx<int3, short3, short3, short3, op>, device::fmx<int4, short4, short4, short4, op>  },
                    { device::fmx<int, short, short, int, op>, device::fmx<int2, short2, short2, int2, op>, device::fmx<int3, short3, short3, int3, op>, device::fmx<int4, short4, short4, int4, op>  },
                    { device::fmx<int, short, short, float, op>, device::fmx<int2, short2, short2, float2, op>, device::fmx<int3, short3, short3, float3, op>, device::fmx<int4, short4, short4, float4, op>  },
                    { device::fmx<int, short, short, double, op>, device::fmx<int2, short2, short2, double2, op>, device::fmx<int3, short3, short3, double3, op>, device::fmx<int4, short4, short4, double4, op>  },
                },
                {
                    { device::fmx<int, short, int, uchar, op>, device::fmx<int2, short2, int2, uchar2, op>, device::fmx<int3, short3, int3, uchar3, op>, device::fmx<int4, short4, int4, uchar4, op>  },
                    { device::fmx<int, short, int, schar, op>, device::fmx<int2, short2, int2, char2, op>, device::fmx<int3, short3, int3, char3, op>, device::fmx<int4, short4, int4, char4, op>  },
                    { device::fmx<int, short, int, ushort, op>, device::fmx<int2, short2, int2, ushort2, op>, device::fmx<int3, short3, int3, ushort3, op>, device::fmx<int4, short4, int4, ushort4, op>  },
                    { device::fmx<int, short, int, short, op>, device::fmx<int2, short2, int2, short2, op>, device::fmx<int3, short3, int3, short3, op>, device::fmx<int4, short4, int4, short4, op>  },
                    { device::fmx<int, short, int, int, op>, device::fmx<int2, short2, int2, int2, op>, device::fmx<int3, short3, int3, int3, op>, device::fmx<int4, short4, int4, int4, op>  },
                    { device::fmx<int, short, int, float, op>, device::fmx<int2, short2, int2, float2, op>, device::fmx<int3, short3, int3, float3, op>, device::fmx<int4, short4, int4, float4, op>  },
                    { device::fmx<int, short, int, double, op>, device::fmx<int2, short2, int2, double2, op>, device::fmx<int3, short3, int3, double3, op>, device::fmx<int4, short4, int4, double4, op>  },
                },
                {
                    { device::fmx<int, short, float, uchar, op>, device::fmx<int2, short2, float2, uchar2, op>, device::fmx<int3, short3, float3, uchar3, op>, device::fmx<int4, short4, float4, uchar4, op>  },
                    { device::fmx<int, short, float, schar, op>, device::fmx<int2, short2, float2, char2, op>, device::fmx<int3, short3, float3, char3, op>, device::fmx<int4, short4, float4, char4, op>  },
                    { device::fmx<int, short, float, ushort, op>, device::fmx<int2, short2, float2, ushort2, op>, device::fmx<int3, short3, float3, ushort3, op>, device::fmx<int4, short4, float4, ushort4, op>  },
                    { device::fmx<int, short, float, short, op>, device::fmx<int2, short2, float2, short2, op>, device::fmx<int3, short3, float3, short3, op>, device::fmx<int4, short4, float4, short4, op>  },
                    { device::fmx<int, short, float, int, op>, device::fmx<int2, short2, float2, int2, op>, device::fmx<int3, short3, float3, int3, op>, device::fmx<int4, short4, float4, int4, op>  },
                    { device::fmx<int, short, float, float, op>, device::fmx<int2, short2, float2, float2, op>, device::fmx<int3, short3, float3, float3, op>, device::fmx<int4, short4, float4, float4, op>  },
                    { device::fmx<int, short, float, double, op>, device::fmx<int2, short2, float2, double2, op>, device::fmx<int3, short3, float3, double3, op>, device::fmx<int4, short4, float4, double4, op>  },
                },
                {
                    { device::fmx<int, short, double, uchar, op>, device::fmx<int2, short2, double2, uchar2, op>, device::fmx<int3, short3, double3, uchar3, op>, device::fmx<int4, short4, double4, uchar4, op>  },
                    { device::fmx<int, short, double, schar, op>, device::fmx<int2, short2, double2, char2, op>, device::fmx<int3, short3, double3, char3, op>, device::fmx<int4, short4, double4, char4, op>  },
                    { device::fmx<int, short, double, ushort, op>, device::fmx<int2, short2, double2, ushort2, op>, device::fmx<int3, short3, double3, ushort3, op>, device::fmx<int4, short4, double4, ushort4, op>  },
                    { device::fmx<int, short, double, short, op>, device::fmx<int2, short2, double2, short2, op>, device::fmx<int3, short3, double3, short3, op>, device::fmx<int4, short4, double4, short4, op>  },
                    { device::fmx<int, short, double, int, op>, device::fmx<int2, short2, double2, int2, op>, device::fmx<int3, short3, double3, int3, op>, device::fmx<int4, short4, double4, int4, op>  },
                    { device::fmx<int, short, double, float, op>, device::fmx<int2, short2, double2, float2, op>, device::fmx<int3, short3, double3, float3, op>, device::fmx<int4, short4, double4, float4, op>  },
                    { device::fmx<int, short, double, double, op>, device::fmx<int2, short2, double2, double2, op>, device::fmx<int3, short3, double3, double3, op>, device::fmx<int4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<int, int, uchar, uchar, op>, device::fmx<int2, int2, uchar2, uchar2, op>, device::fmx<int3, int3, uchar3, uchar3, op>, device::fmx<int4, int4, uchar4, uchar4, op>  },
                    { device::fmx<int, int, uchar, schar, op>, device::fmx<int2, int2, uchar2, char2, op>, device::fmx<int3, int3, uchar3, char3, op>, device::fmx<int4, int4, uchar4, char4, op>  },
                    { device::fmx<int, int, uchar, ushort, op>, device::fmx<int2, int2, uchar2, ushort2, op>, device::fmx<int3, int3, uchar3, ushort3, op>, device::fmx<int4, int4, uchar4, ushort4, op>  },
                    { device::fmx<int, int, uchar, short, op>, device::fmx<int2, int2, uchar2, short2, op>, device::fmx<int3, int3, uchar3, short3, op>, device::fmx<int4, int4, uchar4, short4, op>  },
                    { device::fmx<int, int, uchar, int, op>, device::fmx<int2, int2, uchar2, int2, op>, device::fmx<int3, int3, uchar3, int3, op>, device::fmx<int4, int4, uchar4, int4, op>  },
                    { device::fmx<int, int, uchar, float, op>, device::fmx<int2, int2, uchar2, float2, op>, device::fmx<int3, int3, uchar3, float3, op>, device::fmx<int4, int4, uchar4, float4, op>  },
                    { device::fmx<int, int, uchar, double, op>, device::fmx<int2, int2, uchar2, double2, op>, device::fmx<int3, int3, uchar3, double3, op>, device::fmx<int4, int4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<int, int, schar, uchar, op>, device::fmx<int2, int2, char2, uchar2, op>, device::fmx<int3, int3, char3, uchar3, op>, device::fmx<int4, int4, char4, uchar4, op>  },
                    { device::fmx<int, int, schar, schar, op>, device::fmx<int2, int2, char2, char2, op>, device::fmx<int3, int3, char3, char3, op>, device::fmx<int4, int4, char4, char4, op>  },
                    { device::fmx<int, int, schar, ushort, op>, device::fmx<int2, int2, char2, ushort2, op>, device::fmx<int3, int3, char3, ushort3, op>, device::fmx<int4, int4, char4, ushort4, op>  },
                    { device::fmx<int, int, schar, short, op>, device::fmx<int2, int2, char2, short2, op>, device::fmx<int3, int3, char3, short3, op>, device::fmx<int4, int4, char4, short4, op>  },
                    { device::fmx<int, int, schar, int, op>, device::fmx<int2, int2, char2, int2, op>, device::fmx<int3, int3, char3, int3, op>, device::fmx<int4, int4, char4, int4, op>  },
                    { device::fmx<int, int, schar, float, op>, device::fmx<int2, int2, char2, float2, op>, device::fmx<int3, int3, char3, float3, op>, device::fmx<int4, int4, char4, float4, op>  },
                    { device::fmx<int, int, schar, double, op>, device::fmx<int2, int2, char2, double2, op>, device::fmx<int3, int3, char3, double3, op>, device::fmx<int4, int4, char4, double4, op>  },
                },
                {
                    { device::fmx<int, int, ushort, uchar, op>, device::fmx<int2, int2, ushort2, uchar2, op>, device::fmx<int3, int3, ushort3, uchar3, op>, device::fmx<int4, int4, ushort4, uchar4, op>  },
                    { device::fmx<int, int, ushort, schar, op>, device::fmx<int2, int2, ushort2, char2, op>, device::fmx<int3, int3, ushort3, char3, op>, device::fmx<int4, int4, ushort4, char4, op>  },
                    { device::fmx<int, int, ushort, ushort, op>, device::fmx<int2, int2, ushort2, ushort2, op>, device::fmx<int3, int3, ushort3, ushort3, op>, device::fmx<int4, int4, ushort4, ushort4, op>  },
                    { device::fmx<int, int, ushort, short, op>, device::fmx<int2, int2, ushort2, short2, op>, device::fmx<int3, int3, ushort3, short3, op>, device::fmx<int4, int4, ushort4, short4, op>  },
                    { device::fmx<int, int, ushort, int, op>, device::fmx<int2, int2, ushort2, int2, op>, device::fmx<int3, int3, ushort3, int3, op>, device::fmx<int4, int4, ushort4, int4, op>  },
                    { device::fmx<int, int, ushort, float, op>, device::fmx<int2, int2, ushort2, float2, op>, device::fmx<int3, int3, ushort3, float3, op>, device::fmx<int4, int4, ushort4, float4, op>  },
                    { device::fmx<int, int, ushort, double, op>, device::fmx<int2, int2, ushort2, double2, op>, device::fmx<int3, int3, ushort3, double3, op>, device::fmx<int4, int4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<int, int, short, uchar, op>, device::fmx<int2, int2, short2, uchar2, op>, device::fmx<int3, int3, short3, uchar3, op>, device::fmx<int4, int4, short4, uchar4, op>  },
                    { device::fmx<int, int, short, schar, op>, device::fmx<int2, int2, short2, char2, op>, device::fmx<int3, int3, short3, char3, op>, device::fmx<int4, int4, short4, char4, op>  },
                    { device::fmx<int, int, short, ushort, op>, device::fmx<int2, int2, short2, ushort2, op>, device::fmx<int3, int3, short3, ushort3, op>, device::fmx<int4, int4, short4, ushort4, op>  },
                    { device::fmx<int, int, short, short, op>, device::fmx<int2, int2, short2, short2, op>, device::fmx<int3, int3, short3, short3, op>, device::fmx<int4, int4, short4, short4, op>  },
                    { device::fmx<int, int, short, int, op>, device::fmx<int2, int2, short2, int2, op>, device::fmx<int3, int3, short3, int3, op>, device::fmx<int4, int4, short4, int4, op>  },
                    { device::fmx<int, int, short, float, op>, device::fmx<int2, int2, short2, float2, op>, device::fmx<int3, int3, short3, float3, op>, device::fmx<int4, int4, short4, float4, op>  },
                    { device::fmx<int, int, short, double, op>, device::fmx<int2, int2, short2, double2, op>, device::fmx<int3, int3, short3, double3, op>, device::fmx<int4, int4, short4, double4, op>  },
                },
                {
                    { device::fmx<int, int, int, uchar, op>, device::fmx<int2, int2, int2, uchar2, op>, device::fmx<int3, int3, int3, uchar3, op>, device::fmx<int4, int4, int4, uchar4, op>  },
                    { device::fmx<int, int, int, schar, op>, device::fmx<int2, int2, int2, char2, op>, device::fmx<int3, int3, int3, char3, op>, device::fmx<int4, int4, int4, char4, op>  },
                    { device::fmx<int, int, int, ushort, op>, device::fmx<int2, int2, int2, ushort2, op>, device::fmx<int3, int3, int3, ushort3, op>, device::fmx<int4, int4, int4, ushort4, op>  },
                    { device::fmx<int, int, int, short, op>, device::fmx<int2, int2, int2, short2, op>, device::fmx<int3, int3, int3, short3, op>, device::fmx<int4, int4, int4, short4, op>  },
                    { device::fmx<int, int, int, int, op>, device::fmx<int2, int2, int2, int2, op>, device::fmx<int3, int3, int3, int3, op>, device::fmx<int4, int4, int4, int4, op>  },
                    { device::fmx<int, int, int, float, op>, device::fmx<int2, int2, int2, float2, op>, device::fmx<int3, int3, int3, float3, op>, device::fmx<int4, int4, int4, float4, op>  },
                    { device::fmx<int, int, int, double, op>, device::fmx<int2, int2, int2, double2, op>, device::fmx<int3, int3, int3, double3, op>, device::fmx<int4, int4, int4, double4, op>  },
                },
                {
                    { device::fmx<int, int, float, uchar, op>, device::fmx<int2, int2, float2, uchar2, op>, device::fmx<int3, int3, float3, uchar3, op>, device::fmx<int4, int4, float4, uchar4, op>  },
                    { device::fmx<int, int, float, schar, op>, device::fmx<int2, int2, float2, char2, op>, device::fmx<int3, int3, float3, char3, op>, device::fmx<int4, int4, float4, char4, op>  },
                    { device::fmx<int, int, float, ushort, op>, device::fmx<int2, int2, float2, ushort2, op>, device::fmx<int3, int3, float3, ushort3, op>, device::fmx<int4, int4, float4, ushort4, op>  },
                    { device::fmx<int, int, float, short, op>, device::fmx<int2, int2, float2, short2, op>, device::fmx<int3, int3, float3, short3, op>, device::fmx<int4, int4, float4, short4, op>  },
                    { device::fmx<int, int, float, int, op>, device::fmx<int2, int2, float2, int2, op>, device::fmx<int3, int3, float3, int3, op>, device::fmx<int4, int4, float4, int4, op>  },
                    { device::fmx<int, int, float, float, op>, device::fmx<int2, int2, float2, float2, op>, device::fmx<int3, int3, float3, float3, op>, device::fmx<int4, int4, float4, float4, op>  },
                    { device::fmx<int, int, float, double, op>, device::fmx<int2, int2, float2, double2, op>, device::fmx<int3, int3, float3, double3, op>, device::fmx<int4, int4, float4, double4, op>  },
                },
                {
                    { device::fmx<int, int, double, uchar, op>, device::fmx<int2, int2, double2, uchar2, op>, device::fmx<int3, int3, double3, uchar3, op>, device::fmx<int4, int4, double4, uchar4, op>  },
                    { device::fmx<int, int, double, schar, op>, device::fmx<int2, int2, double2, char2, op>, device::fmx<int3, int3, double3, char3, op>, device::fmx<int4, int4, double4, char4, op>  },
                    { device::fmx<int, int, double, ushort, op>, device::fmx<int2, int2, double2, ushort2, op>, device::fmx<int3, int3, double3, ushort3, op>, device::fmx<int4, int4, double4, ushort4, op>  },
                    { device::fmx<int, int, double, short, op>, device::fmx<int2, int2, double2, short2, op>, device::fmx<int3, int3, double3, short3, op>, device::fmx<int4, int4, double4, short4, op>  },
                    { device::fmx<int, int, double, int, op>, device::fmx<int2, int2, double2, int2, op>, device::fmx<int3, int3, double3, int3, op>, device::fmx<int4, int4, double4, int4, op>  },
                    { device::fmx<int, int, double, float, op>, device::fmx<int2, int2, double2, float2, op>, device::fmx<int3, int3, double3, float3, op>, device::fmx<int4, int4, double4, float4, op>  },
                    { device::fmx<int, int, double, double, op>, device::fmx<int2, int2, double2, double2, op>, device::fmx<int3, int3, double3, double3, op>, device::fmx<int4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<int, float, uchar, uchar, op>, device::fmx<int2, float2, uchar2, uchar2, op>, device::fmx<int3, float3, uchar3, uchar3, op>, device::fmx<int4, float4, uchar4, uchar4, op>  },
                    { device::fmx<int, float, uchar, schar, op>, device::fmx<int2, float2, uchar2, char2, op>, device::fmx<int3, float3, uchar3, char3, op>, device::fmx<int4, float4, uchar4, char4, op>  },
                    { device::fmx<int, float, uchar, ushort, op>, device::fmx<int2, float2, uchar2, ushort2, op>, device::fmx<int3, float3, uchar3, ushort3, op>, device::fmx<int4, float4, uchar4, ushort4, op>  },
                    { device::fmx<int, float, uchar, short, op>, device::fmx<int2, float2, uchar2, short2, op>, device::fmx<int3, float3, uchar3, short3, op>, device::fmx<int4, float4, uchar4, short4, op>  },
                    { device::fmx<int, float, uchar, int, op>, device::fmx<int2, float2, uchar2, int2, op>, device::fmx<int3, float3, uchar3, int3, op>, device::fmx<int4, float4, uchar4, int4, op>  },
                    { device::fmx<int, float, uchar, float, op>, device::fmx<int2, float2, uchar2, float2, op>, device::fmx<int3, float3, uchar3, float3, op>, device::fmx<int4, float4, uchar4, float4, op>  },
                    { device::fmx<int, float, uchar, double, op>, device::fmx<int2, float2, uchar2, double2, op>, device::fmx<int3, float3, uchar3, double3, op>, device::fmx<int4, float4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<int, float, schar, uchar, op>, device::fmx<int2, float2, char2, uchar2, op>, device::fmx<int3, float3, char3, uchar3, op>, device::fmx<int4, float4, char4, uchar4, op>  },
                    { device::fmx<int, float, schar, schar, op>, device::fmx<int2, float2, char2, char2, op>, device::fmx<int3, float3, char3, char3, op>, device::fmx<int4, float4, char4, char4, op>  },
                    { device::fmx<int, float, schar, ushort, op>, device::fmx<int2, float2, char2, ushort2, op>, device::fmx<int3, float3, char3, ushort3, op>, device::fmx<int4, float4, char4, ushort4, op>  },
                    { device::fmx<int, float, schar, short, op>, device::fmx<int2, float2, char2, short2, op>, device::fmx<int3, float3, char3, short3, op>, device::fmx<int4, float4, char4, short4, op>  },
                    { device::fmx<int, float, schar, int, op>, device::fmx<int2, float2, char2, int2, op>, device::fmx<int3, float3, char3, int3, op>, device::fmx<int4, float4, char4, int4, op>  },
                    { device::fmx<int, float, schar, float, op>, device::fmx<int2, float2, char2, float2, op>, device::fmx<int3, float3, char3, float3, op>, device::fmx<int4, float4, char4, float4, op>  },
                    { device::fmx<int, float, schar, double, op>, device::fmx<int2, float2, char2, double2, op>, device::fmx<int3, float3, char3, double3, op>, device::fmx<int4, float4, char4, double4, op>  },
                },
                {
                    { device::fmx<int, float, ushort, uchar, op>, device::fmx<int2, float2, ushort2, uchar2, op>, device::fmx<int3, float3, ushort3, uchar3, op>, device::fmx<int4, float4, ushort4, uchar4, op>  },
                    { device::fmx<int, float, ushort, schar, op>, device::fmx<int2, float2, ushort2, char2, op>, device::fmx<int3, float3, ushort3, char3, op>, device::fmx<int4, float4, ushort4, char4, op>  },
                    { device::fmx<int, float, ushort, ushort, op>, device::fmx<int2, float2, ushort2, ushort2, op>, device::fmx<int3, float3, ushort3, ushort3, op>, device::fmx<int4, float4, ushort4, ushort4, op>  },
                    { device::fmx<int, float, ushort, short, op>, device::fmx<int2, float2, ushort2, short2, op>, device::fmx<int3, float3, ushort3, short3, op>, device::fmx<int4, float4, ushort4, short4, op>  },
                    { device::fmx<int, float, ushort, int, op>, device::fmx<int2, float2, ushort2, int2, op>, device::fmx<int3, float3, ushort3, int3, op>, device::fmx<int4, float4, ushort4, int4, op>  },
                    { device::fmx<int, float, ushort, float, op>, device::fmx<int2, float2, ushort2, float2, op>, device::fmx<int3, float3, ushort3, float3, op>, device::fmx<int4, float4, ushort4, float4, op>  },
                    { device::fmx<int, float, ushort, double, op>, device::fmx<int2, float2, ushort2, double2, op>, device::fmx<int3, float3, ushort3, double3, op>, device::fmx<int4, float4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<int, float, short, uchar, op>, device::fmx<int2, float2, short2, uchar2, op>, device::fmx<int3, float3, short3, uchar3, op>, device::fmx<int4, float4, short4, uchar4, op>  },
                    { device::fmx<int, float, short, schar, op>, device::fmx<int2, float2, short2, char2, op>, device::fmx<int3, float3, short3, char3, op>, device::fmx<int4, float4, short4, char4, op>  },
                    { device::fmx<int, float, short, ushort, op>, device::fmx<int2, float2, short2, ushort2, op>, device::fmx<int3, float3, short3, ushort3, op>, device::fmx<int4, float4, short4, ushort4, op>  },
                    { device::fmx<int, float, short, short, op>, device::fmx<int2, float2, short2, short2, op>, device::fmx<int3, float3, short3, short3, op>, device::fmx<int4, float4, short4, short4, op>  },
                    { device::fmx<int, float, short, int, op>, device::fmx<int2, float2, short2, int2, op>, device::fmx<int3, float3, short3, int3, op>, device::fmx<int4, float4, short4, int4, op>  },
                    { device::fmx<int, float, short, float, op>, device::fmx<int2, float2, short2, float2, op>, device::fmx<int3, float3, short3, float3, op>, device::fmx<int4, float4, short4, float4, op>  },
                    { device::fmx<int, float, short, double, op>, device::fmx<int2, float2, short2, double2, op>, device::fmx<int3, float3, short3, double3, op>, device::fmx<int4, float4, short4, double4, op>  },
                },
                {
                    { device::fmx<int, float, int, uchar, op>, device::fmx<int2, float2, int2, uchar2, op>, device::fmx<int3, float3, int3, uchar3, op>, device::fmx<int4, float4, int4, uchar4, op>  },
                    { device::fmx<int, float, int, schar, op>, device::fmx<int2, float2, int2, char2, op>, device::fmx<int3, float3, int3, char3, op>, device::fmx<int4, float4, int4, char4, op>  },
                    { device::fmx<int, float, int, ushort, op>, device::fmx<int2, float2, int2, ushort2, op>, device::fmx<int3, float3, int3, ushort3, op>, device::fmx<int4, float4, int4, ushort4, op>  },
                    { device::fmx<int, float, int, short, op>, device::fmx<int2, float2, int2, short2, op>, device::fmx<int3, float3, int3, short3, op>, device::fmx<int4, float4, int4, short4, op>  },
                    { device::fmx<int, float, int, int, op>, device::fmx<int2, float2, int2, int2, op>, device::fmx<int3, float3, int3, int3, op>, device::fmx<int4, float4, int4, int4, op>  },
                    { device::fmx<int, float, int, float, op>, device::fmx<int2, float2, int2, float2, op>, device::fmx<int3, float3, int3, float3, op>, device::fmx<int4, float4, int4, float4, op>  },
                    { device::fmx<int, float, int, double, op>, device::fmx<int2, float2, int2, double2, op>, device::fmx<int3, float3, int3, double3, op>, device::fmx<int4, float4, int4, double4, op>  },
                },
                {
                    { device::fmx<int, float, float, uchar, op>, device::fmx<int2, float2, float2, uchar2, op>, device::fmx<int3, float3, float3, uchar3, op>, device::fmx<int4, float4, float4, uchar4, op>  },
                    { device::fmx<int, float, float, schar, op>, device::fmx<int2, float2, float2, char2, op>, device::fmx<int3, float3, float3, char3, op>, device::fmx<int4, float4, float4, char4, op>  },
                    { device::fmx<int, float, float, ushort, op>, device::fmx<int2, float2, float2, ushort2, op>, device::fmx<int3, float3, float3, ushort3, op>, device::fmx<int4, float4, float4, ushort4, op>  },
                    { device::fmx<int, float, float, short, op>, device::fmx<int2, float2, float2, short2, op>, device::fmx<int3, float3, float3, short3, op>, device::fmx<int4, float4, float4, short4, op>  },
                    { device::fmx<int, float, float, int, op>, device::fmx<int2, float2, float2, int2, op>, device::fmx<int3, float3, float3, int3, op>, device::fmx<int4, float4, float4, int4, op>  },
                    { device::fmx<int, float, float, float, op>, device::fmx<int2, float2, float2, float2, op>, device::fmx<int3, float3, float3, float3, op>, device::fmx<int4, float4, float4, float4, op>  },
                    { device::fmx<int, float, float, double, op>, device::fmx<int2, float2, float2, double2, op>, device::fmx<int3, float3, float3, double3, op>, device::fmx<int4, float4, float4, double4, op>  },
                },
                {
                    { device::fmx<int, float, double, uchar, op>, device::fmx<int2, float2, double2, uchar2, op>, device::fmx<int3, float3, double3, uchar3, op>, device::fmx<int4, float4, double4, uchar4, op>  },
                    { device::fmx<int, float, double, schar, op>, device::fmx<int2, float2, double2, char2, op>, device::fmx<int3, float3, double3, char3, op>, device::fmx<int4, float4, double4, char4, op>  },
                    { device::fmx<int, float, double, ushort, op>, device::fmx<int2, float2, double2, ushort2, op>, device::fmx<int3, float3, double3, ushort3, op>, device::fmx<int4, float4, double4, ushort4, op>  },
                    { device::fmx<int, float, double, short, op>, device::fmx<int2, float2, double2, short2, op>, device::fmx<int3, float3, double3, short3, op>, device::fmx<int4, float4, double4, short4, op>  },
                    { device::fmx<int, float, double, int, op>, device::fmx<int2, float2, double2, int2, op>, device::fmx<int3, float3, double3, int3, op>, device::fmx<int4, float4, double4, int4, op>  },
                    { device::fmx<int, float, double, float, op>, device::fmx<int2, float2, double2, float2, op>, device::fmx<int3, float3, double3, float3, op>, device::fmx<int4, float4, double4, float4, op>  },
                    { device::fmx<int, float, double, double, op>, device::fmx<int2, float2, double2, double2, op>, device::fmx<int3, float3, double3, double3, op>, device::fmx<int4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<int, double, uchar, uchar, op>, device::fmx<int2, double2, uchar2, uchar2, op>, device::fmx<int3, double3, uchar3, uchar3, op>, device::fmx<int4, double4, uchar4, uchar4, op>  },
                    { device::fmx<int, double, uchar, schar, op>, device::fmx<int2, double2, uchar2, char2, op>, device::fmx<int3, double3, uchar3, char3, op>, device::fmx<int4, double4, uchar4, char4, op>  },
                    { device::fmx<int, double, uchar, ushort, op>, device::fmx<int2, double2, uchar2, ushort2, op>, device::fmx<int3, double3, uchar3, ushort3, op>, device::fmx<int4, double4, uchar4, ushort4, op>  },
                    { device::fmx<int, double, uchar, short, op>, device::fmx<int2, double2, uchar2, short2, op>, device::fmx<int3, double3, uchar3, short3, op>, device::fmx<int4, double4, uchar4, short4, op>  },
                    { device::fmx<int, double, uchar, int, op>, device::fmx<int2, double2, uchar2, int2, op>, device::fmx<int3, double3, uchar3, int3, op>, device::fmx<int4, double4, uchar4, int4, op>  },
                    { device::fmx<int, double, uchar, float, op>, device::fmx<int2, double2, uchar2, float2, op>, device::fmx<int3, double3, uchar3, float3, op>, device::fmx<int4, double4, uchar4, float4, op>  },
                    { device::fmx<int, double, uchar, double, op>, device::fmx<int2, double2, uchar2, double2, op>, device::fmx<int3, double3, uchar3, double3, op>, device::fmx<int4, double4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<int, double, schar, uchar, op>, device::fmx<int2, double2, char2, uchar2, op>, device::fmx<int3, double3, char3, uchar3, op>, device::fmx<int4, double4, char4, uchar4, op>  },
                    { device::fmx<int, double, schar, schar, op>, device::fmx<int2, double2, char2, char2, op>, device::fmx<int3, double3, char3, char3, op>, device::fmx<int4, double4, char4, char4, op>  },
                    { device::fmx<int, double, schar, ushort, op>, device::fmx<int2, double2, char2, ushort2, op>, device::fmx<int3, double3, char3, ushort3, op>, device::fmx<int4, double4, char4, ushort4, op>  },
                    { device::fmx<int, double, schar, short, op>, device::fmx<int2, double2, char2, short2, op>, device::fmx<int3, double3, char3, short3, op>, device::fmx<int4, double4, char4, short4, op>  },
                    { device::fmx<int, double, schar, int, op>, device::fmx<int2, double2, char2, int2, op>, device::fmx<int3, double3, char3, int3, op>, device::fmx<int4, double4, char4, int4, op>  },
                    { device::fmx<int, double, schar, float, op>, device::fmx<int2, double2, char2, float2, op>, device::fmx<int3, double3, char3, float3, op>, device::fmx<int4, double4, char4, float4, op>  },
                    { device::fmx<int, double, schar, double, op>, device::fmx<int2, double2, char2, double2, op>, device::fmx<int3, double3, char3, double3, op>, device::fmx<int4, double4, char4, double4, op>  },
                },
                {
                    { device::fmx<int, double, ushort, uchar, op>, device::fmx<int2, double2, ushort2, uchar2, op>, device::fmx<int3, double3, ushort3, uchar3, op>, device::fmx<int4, double4, ushort4, uchar4, op>  },
                    { device::fmx<int, double, ushort, schar, op>, device::fmx<int2, double2, ushort2, char2, op>, device::fmx<int3, double3, ushort3, char3, op>, device::fmx<int4, double4, ushort4, char4, op>  },
                    { device::fmx<int, double, ushort, ushort, op>, device::fmx<int2, double2, ushort2, ushort2, op>, device::fmx<int3, double3, ushort3, ushort3, op>, device::fmx<int4, double4, ushort4, ushort4, op>  },
                    { device::fmx<int, double, ushort, short, op>, device::fmx<int2, double2, ushort2, short2, op>, device::fmx<int3, double3, ushort3, short3, op>, device::fmx<int4, double4, ushort4, short4, op>  },
                    { device::fmx<int, double, ushort, int, op>, device::fmx<int2, double2, ushort2, int2, op>, device::fmx<int3, double3, ushort3, int3, op>, device::fmx<int4, double4, ushort4, int4, op>  },
                    { device::fmx<int, double, ushort, float, op>, device::fmx<int2, double2, ushort2, float2, op>, device::fmx<int3, double3, ushort3, float3, op>, device::fmx<int4, double4, ushort4, float4, op>  },
                    { device::fmx<int, double, ushort, double, op>, device::fmx<int2, double2, ushort2, double2, op>, device::fmx<int3, double3, ushort3, double3, op>, device::fmx<int4, double4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<int, double, short, uchar, op>, device::fmx<int2, double2, short2, uchar2, op>, device::fmx<int3, double3, short3, uchar3, op>, device::fmx<int4, double4, short4, uchar4, op>  },
                    { device::fmx<int, double, short, schar, op>, device::fmx<int2, double2, short2, char2, op>, device::fmx<int3, double3, short3, char3, op>, device::fmx<int4, double4, short4, char4, op>  },
                    { device::fmx<int, double, short, ushort, op>, device::fmx<int2, double2, short2, ushort2, op>, device::fmx<int3, double3, short3, ushort3, op>, device::fmx<int4, double4, short4, ushort4, op>  },
                    { device::fmx<int, double, short, short, op>, device::fmx<int2, double2, short2, short2, op>, device::fmx<int3, double3, short3, short3, op>, device::fmx<int4, double4, short4, short4, op>  },
                    { device::fmx<int, double, short, int, op>, device::fmx<int2, double2, short2, int2, op>, device::fmx<int3, double3, short3, int3, op>, device::fmx<int4, double4, short4, int4, op>  },
                    { device::fmx<int, double, short, float, op>, device::fmx<int2, double2, short2, float2, op>, device::fmx<int3, double3, short3, float3, op>, device::fmx<int4, double4, short4, float4, op>  },
                    { device::fmx<int, double, short, double, op>, device::fmx<int2, double2, short2, double2, op>, device::fmx<int3, double3, short3, double3, op>, device::fmx<int4, double4, short4, double4, op>  },
                },
                {
                    { device::fmx<int, double, int, uchar, op>, device::fmx<int2, double2, int2, uchar2, op>, device::fmx<int3, double3, int3, uchar3, op>, device::fmx<int4, double4, int4, uchar4, op>  },
                    { device::fmx<int, double, int, schar, op>, device::fmx<int2, double2, int2, char2, op>, device::fmx<int3, double3, int3, char3, op>, device::fmx<int4, double4, int4, char4, op>  },
                    { device::fmx<int, double, int, ushort, op>, device::fmx<int2, double2, int2, ushort2, op>, device::fmx<int3, double3, int3, ushort3, op>, device::fmx<int4, double4, int4, ushort4, op>  },
                    { device::fmx<int, double, int, short, op>, device::fmx<int2, double2, int2, short2, op>, device::fmx<int3, double3, int3, short3, op>, device::fmx<int4, double4, int4, short4, op>  },
                    { device::fmx<int, double, int, int, op>, device::fmx<int2, double2, int2, int2, op>, device::fmx<int3, double3, int3, int3, op>, device::fmx<int4, double4, int4, int4, op>  },
                    { device::fmx<int, double, int, float, op>, device::fmx<int2, double2, int2, float2, op>, device::fmx<int3, double3, int3, float3, op>, device::fmx<int4, double4, int4, float4, op>  },
                    { device::fmx<int, double, int, double, op>, device::fmx<int2, double2, int2, double2, op>, device::fmx<int3, double3, int3, double3, op>, device::fmx<int4, double4, int4, double4, op>  },
                },
                {
                    { device::fmx<int, double, float, uchar, op>, device::fmx<int2, double2, float2, uchar2, op>, device::fmx<int3, double3, float3, uchar3, op>, device::fmx<int4, double4, float4, uchar4, op>  },
                    { device::fmx<int, double, float, schar, op>, device::fmx<int2, double2, float2, char2, op>, device::fmx<int3, double3, float3, char3, op>, device::fmx<int4, double4, float4, char4, op>  },
                    { device::fmx<int, double, float, ushort, op>, device::fmx<int2, double2, float2, ushort2, op>, device::fmx<int3, double3, float3, ushort3, op>, device::fmx<int4, double4, float4, ushort4, op>  },
                    { device::fmx<int, double, float, short, op>, device::fmx<int2, double2, float2, short2, op>, device::fmx<int3, double3, float3, short3, op>, device::fmx<int4, double4, float4, short4, op>  },
                    { device::fmx<int, double, float, int, op>, device::fmx<int2, double2, float2, int2, op>, device::fmx<int3, double3, float3, int3, op>, device::fmx<int4, double4, float4, int4, op>  },
                    { device::fmx<int, double, float, float, op>, device::fmx<int2, double2, float2, float2, op>, device::fmx<int3, double3, float3, float3, op>, device::fmx<int4, double4, float4, float4, op>  },
                    { device::fmx<int, double, float, double, op>, device::fmx<int2, double2, float2, double2, op>, device::fmx<int3, double3, float3, double3, op>, device::fmx<int4, double4, float4, double4, op>  },
                },
                {
                    { device::fmx<int, double, double, uchar, op>, device::fmx<int2, double2, double2, uchar2, op>, device::fmx<int3, double3, double3, uchar3, op>, device::fmx<int4, double4, double4, uchar4, op>  },
                    { device::fmx<int, double, double, schar, op>, device::fmx<int2, double2, double2, char2, op>, device::fmx<int3, double3, double3, char3, op>, device::fmx<int4, double4, double4, char4, op>  },
                    { device::fmx<int, double, double, ushort, op>, device::fmx<int2, double2, double2, ushort2, op>, device::fmx<int3, double3, double3, ushort3, op>, device::fmx<int4, double4, double4, ushort4, op>  },
                    { device::fmx<int, double, double, short, op>, device::fmx<int2, double2, double2, short2, op>, device::fmx<int3, double3, double3, short3, op>, device::fmx<int4, double4, double4, short4, op>  },
                    { device::fmx<int, double, double, int, op>, device::fmx<int2, double2, double2, int2, op>, device::fmx<int3, double3, double3, int3, op>, device::fmx<int4, double4, double4, int4, op>  },
                    { device::fmx<int, double, double, float, op>, device::fmx<int2, double2, double2, float2, op>, device::fmx<int3, double3, double3, float3, op>, device::fmx<int4, double4, double4, float4, op>  },
                    { device::fmx<int, double, double, double, op>, device::fmx<int2, double2, double2, double2, op>, device::fmx<int3, double3, double3, double3, op>, device::fmx<int4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { device::fmx<float, uchar, uchar, uchar, op>, device::fmx<float2, uchar2, uchar2, uchar2, op>, device::fmx<float3, uchar3, uchar3, uchar3, op>, device::fmx<float4, uchar4, uchar4, uchar4, op>  },
                    { device::fmx<float, uchar, uchar, schar, op>, device::fmx<float2, uchar2, uchar2, char2, op>, device::fmx<float3, uchar3, uchar3, char3, op>, device::fmx<float4, uchar4, uchar4, char4, op>  },
                    { device::fmx<float, uchar, uchar, ushort, op>, device::fmx<float2, uchar2, uchar2, ushort2, op>, device::fmx<float3, uchar3, uchar3, ushort3, op>, device::fmx<float4, uchar4, uchar4, ushort4, op>  },
                    { device::fmx<float, uchar, uchar, short, op>, device::fmx<float2, uchar2, uchar2, short2, op>, device::fmx<float3, uchar3, uchar3, short3, op>, device::fmx<float4, uchar4, uchar4, short4, op>  },
                    { device::fmx<float, uchar, uchar, int, op>, device::fmx<float2, uchar2, uchar2, int2, op>, device::fmx<float3, uchar3, uchar3, int3, op>, device::fmx<float4, uchar4, uchar4, int4, op>  },
                    { device::fmx<float, uchar, uchar, float, op>, device::fmx<float2, uchar2, uchar2, float2, op>, device::fmx<float3, uchar3, uchar3, float3, op>, device::fmx<float4, uchar4, uchar4, float4, op>  },
                    { device::fmx<float, uchar, uchar, double, op>, device::fmx<float2, uchar2, uchar2, double2, op>, device::fmx<float3, uchar3, uchar3, double3, op>, device::fmx<float4, uchar4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<float, uchar, schar, uchar, op>, device::fmx<float2, uchar2, char2, uchar2, op>, device::fmx<float3, uchar3, char3, uchar3, op>, device::fmx<float4, uchar4, char4, uchar4, op>  },
                    { device::fmx<float, uchar, schar, schar, op>, device::fmx<float2, uchar2, char2, char2, op>, device::fmx<float3, uchar3, char3, char3, op>, device::fmx<float4, uchar4, char4, char4, op>  },
                    { device::fmx<float, uchar, schar, ushort, op>, device::fmx<float2, uchar2, char2, ushort2, op>, device::fmx<float3, uchar3, char3, ushort3, op>, device::fmx<float4, uchar4, char4, ushort4, op>  },
                    { device::fmx<float, uchar, schar, short, op>, device::fmx<float2, uchar2, char2, short2, op>, device::fmx<float3, uchar3, char3, short3, op>, device::fmx<float4, uchar4, char4, short4, op>  },
                    { device::fmx<float, uchar, schar, int, op>, device::fmx<float2, uchar2, char2, int2, op>, device::fmx<float3, uchar3, char3, int3, op>, device::fmx<float4, uchar4, char4, int4, op>  },
                    { device::fmx<float, uchar, schar, float, op>, device::fmx<float2, uchar2, char2, float2, op>, device::fmx<float3, uchar3, char3, float3, op>, device::fmx<float4, uchar4, char4, float4, op>  },
                    { device::fmx<float, uchar, schar, double, op>, device::fmx<float2, uchar2, char2, double2, op>, device::fmx<float3, uchar3, char3, double3, op>, device::fmx<float4, uchar4, char4, double4, op>  },
                },
                {
                    { device::fmx<float, uchar, ushort, uchar, op>, device::fmx<float2, uchar2, ushort2, uchar2, op>, device::fmx<float3, uchar3, ushort3, uchar3, op>, device::fmx<float4, uchar4, ushort4, uchar4, op>  },
                    { device::fmx<float, uchar, ushort, schar, op>, device::fmx<float2, uchar2, ushort2, char2, op>, device::fmx<float3, uchar3, ushort3, char3, op>, device::fmx<float4, uchar4, ushort4, char4, op>  },
                    { device::fmx<float, uchar, ushort, ushort, op>, device::fmx<float2, uchar2, ushort2, ushort2, op>, device::fmx<float3, uchar3, ushort3, ushort3, op>, device::fmx<float4, uchar4, ushort4, ushort4, op>  },
                    { device::fmx<float, uchar, ushort, short, op>, device::fmx<float2, uchar2, ushort2, short2, op>, device::fmx<float3, uchar3, ushort3, short3, op>, device::fmx<float4, uchar4, ushort4, short4, op>  },
                    { device::fmx<float, uchar, ushort, int, op>, device::fmx<float2, uchar2, ushort2, int2, op>, device::fmx<float3, uchar3, ushort3, int3, op>, device::fmx<float4, uchar4, ushort4, int4, op>  },
                    { device::fmx<float, uchar, ushort, float, op>, device::fmx<float2, uchar2, ushort2, float2, op>, device::fmx<float3, uchar3, ushort3, float3, op>, device::fmx<float4, uchar4, ushort4, float4, op>  },
                    { device::fmx<float, uchar, ushort, double, op>, device::fmx<float2, uchar2, ushort2, double2, op>, device::fmx<float3, uchar3, ushort3, double3, op>, device::fmx<float4, uchar4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<float, uchar, short, uchar, op>, device::fmx<float2, uchar2, short2, uchar2, op>, device::fmx<float3, uchar3, short3, uchar3, op>, device::fmx<float4, uchar4, short4, uchar4, op>  },
                    { device::fmx<float, uchar, short, schar, op>, device::fmx<float2, uchar2, short2, char2, op>, device::fmx<float3, uchar3, short3, char3, op>, device::fmx<float4, uchar4, short4, char4, op>  },
                    { device::fmx<float, uchar, short, ushort, op>, device::fmx<float2, uchar2, short2, ushort2, op>, device::fmx<float3, uchar3, short3, ushort3, op>, device::fmx<float4, uchar4, short4, ushort4, op>  },
                    { device::fmx<float, uchar, short, short, op>, device::fmx<float2, uchar2, short2, short2, op>, device::fmx<float3, uchar3, short3, short3, op>, device::fmx<float4, uchar4, short4, short4, op>  },
                    { device::fmx<float, uchar, short, int, op>, device::fmx<float2, uchar2, short2, int2, op>, device::fmx<float3, uchar3, short3, int3, op>, device::fmx<float4, uchar4, short4, int4, op>  },
                    { device::fmx<float, uchar, short, float, op>, device::fmx<float2, uchar2, short2, float2, op>, device::fmx<float3, uchar3, short3, float3, op>, device::fmx<float4, uchar4, short4, float4, op>  },
                    { device::fmx<float, uchar, short, double, op>, device::fmx<float2, uchar2, short2, double2, op>, device::fmx<float3, uchar3, short3, double3, op>, device::fmx<float4, uchar4, short4, double4, op>  },
                },
                {
                    { device::fmx<float, uchar, int, uchar, op>, device::fmx<float2, uchar2, int2, uchar2, op>, device::fmx<float3, uchar3, int3, uchar3, op>, device::fmx<float4, uchar4, int4, uchar4, op>  },
                    { device::fmx<float, uchar, int, schar, op>, device::fmx<float2, uchar2, int2, char2, op>, device::fmx<float3, uchar3, int3, char3, op>, device::fmx<float4, uchar4, int4, char4, op>  },
                    { device::fmx<float, uchar, int, ushort, op>, device::fmx<float2, uchar2, int2, ushort2, op>, device::fmx<float3, uchar3, int3, ushort3, op>, device::fmx<float4, uchar4, int4, ushort4, op>  },
                    { device::fmx<float, uchar, int, short, op>, device::fmx<float2, uchar2, int2, short2, op>, device::fmx<float3, uchar3, int3, short3, op>, device::fmx<float4, uchar4, int4, short4, op>  },
                    { device::fmx<float, uchar, int, int, op>, device::fmx<float2, uchar2, int2, int2, op>, device::fmx<float3, uchar3, int3, int3, op>, device::fmx<float4, uchar4, int4, int4, op>  },
                    { device::fmx<float, uchar, int, float, op>, device::fmx<float2, uchar2, int2, float2, op>, device::fmx<float3, uchar3, int3, float3, op>, device::fmx<float4, uchar4, int4, float4, op>  },
                    { device::fmx<float, uchar, int, double, op>, device::fmx<float2, uchar2, int2, double2, op>, device::fmx<float3, uchar3, int3, double3, op>, device::fmx<float4, uchar4, int4, double4, op>  },
                },
                {
                    { device::fmx<float, uchar, float, uchar, op>, device::fmx<float2, uchar2, float2, uchar2, op>, device::fmx<float3, uchar3, float3, uchar3, op>, device::fmx<float4, uchar4, float4, uchar4, op>  },
                    { device::fmx<float, uchar, float, schar, op>, device::fmx<float2, uchar2, float2, char2, op>, device::fmx<float3, uchar3, float3, char3, op>, device::fmx<float4, uchar4, float4, char4, op>  },
                    { device::fmx<float, uchar, float, ushort, op>, device::fmx<float2, uchar2, float2, ushort2, op>, device::fmx<float3, uchar3, float3, ushort3, op>, device::fmx<float4, uchar4, float4, ushort4, op>  },
                    { device::fmx<float, uchar, float, short, op>, device::fmx<float2, uchar2, float2, short2, op>, device::fmx<float3, uchar3, float3, short3, op>, device::fmx<float4, uchar4, float4, short4, op>  },
                    { device::fmx<float, uchar, float, int, op>, device::fmx<float2, uchar2, float2, int2, op>, device::fmx<float3, uchar3, float3, int3, op>, device::fmx<float4, uchar4, float4, int4, op>  },
                    { device::fmx<float, uchar, float, float, op>, device::fmx<float2, uchar2, float2, float2, op>, device::fmx<float3, uchar3, float3, float3, op>, device::fmx<float4, uchar4, float4, float4, op>  },
                    { device::fmx<float, uchar, float, double, op>, device::fmx<float2, uchar2, float2, double2, op>, device::fmx<float3, uchar3, float3, double3, op>, device::fmx<float4, uchar4, float4, double4, op>  },
                },
                {
                    { device::fmx<float, uchar, double, uchar, op>, device::fmx<float2, uchar2, double2, uchar2, op>, device::fmx<float3, uchar3, double3, uchar3, op>, device::fmx<float4, uchar4, double4, uchar4, op>  },
                    { device::fmx<float, uchar, double, schar, op>, device::fmx<float2, uchar2, double2, char2, op>, device::fmx<float3, uchar3, double3, char3, op>, device::fmx<float4, uchar4, double4, char4, op>  },
                    { device::fmx<float, uchar, double, ushort, op>, device::fmx<float2, uchar2, double2, ushort2, op>, device::fmx<float3, uchar3, double3, ushort3, op>, device::fmx<float4, uchar4, double4, ushort4, op>  },
                    { device::fmx<float, uchar, double, short, op>, device::fmx<float2, uchar2, double2, short2, op>, device::fmx<float3, uchar3, double3, short3, op>, device::fmx<float4, uchar4, double4, short4, op>  },
                    { device::fmx<float, uchar, double, int, op>, device::fmx<float2, uchar2, double2, int2, op>, device::fmx<float3, uchar3, double3, int3, op>, device::fmx<float4, uchar4, double4, int4, op>  },
                    { device::fmx<float, uchar, double, float, op>, device::fmx<float2, uchar2, double2, float2, op>, device::fmx<float3, uchar3, double3, float3, op>, device::fmx<float4, uchar4, double4, float4, op>  },
                    { device::fmx<float, uchar, double, double, op>, device::fmx<float2, uchar2, double2, double2, op>, device::fmx<float3, uchar3, double3, double3, op>, device::fmx<float4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<float, schar, uchar, uchar, op>, device::fmx<float2, char2, uchar2, uchar2, op>, device::fmx<float3, char3, uchar3, uchar3, op>, device::fmx<float4, char4, uchar4, uchar4, op>  },
                    { device::fmx<float, schar, uchar, schar, op>, device::fmx<float2, char2, uchar2, char2, op>, device::fmx<float3, char3, uchar3, char3, op>, device::fmx<float4, char4, uchar4, char4, op>  },
                    { device::fmx<float, schar, uchar, ushort, op>, device::fmx<float2, char2, uchar2, ushort2, op>, device::fmx<float3, char3, uchar3, ushort3, op>, device::fmx<float4, char4, uchar4, ushort4, op>  },
                    { device::fmx<float, schar, uchar, short, op>, device::fmx<float2, char2, uchar2, short2, op>, device::fmx<float3, char3, uchar3, short3, op>, device::fmx<float4, char4, uchar4, short4, op>  },
                    { device::fmx<float, schar, uchar, int, op>, device::fmx<float2, char2, uchar2, int2, op>, device::fmx<float3, char3, uchar3, int3, op>, device::fmx<float4, char4, uchar4, int4, op>  },
                    { device::fmx<float, schar, uchar, float, op>, device::fmx<float2, char2, uchar2, float2, op>, device::fmx<float3, char3, uchar3, float3, op>, device::fmx<float4, char4, uchar4, float4, op>  },
                    { device::fmx<float, schar, uchar, double, op>, device::fmx<float2, char2, uchar2, double2, op>, device::fmx<float3, char3, uchar3, double3, op>, device::fmx<float4, char4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<float, schar, schar, uchar, op>, device::fmx<float2, char2, char2, uchar2, op>, device::fmx<float3, char3, char3, uchar3, op>, device::fmx<float4, char4, char4, uchar4, op>  },
                    { device::fmx<float, schar, schar, schar, op>, device::fmx<float2, char2, char2, char2, op>, device::fmx<float3, char3, char3, char3, op>, device::fmx<float4, char4, char4, char4, op>  },
                    { device::fmx<float, schar, schar, ushort, op>, device::fmx<float2, char2, char2, ushort2, op>, device::fmx<float3, char3, char3, ushort3, op>, device::fmx<float4, char4, char4, ushort4, op>  },
                    { device::fmx<float, schar, schar, short, op>, device::fmx<float2, char2, char2, short2, op>, device::fmx<float3, char3, char3, short3, op>, device::fmx<float4, char4, char4, short4, op>  },
                    { device::fmx<float, schar, schar, int, op>, device::fmx<float2, char2, char2, int2, op>, device::fmx<float3, char3, char3, int3, op>, device::fmx<float4, char4, char4, int4, op>  },
                    { device::fmx<float, schar, schar, float, op>, device::fmx<float2, char2, char2, float2, op>, device::fmx<float3, char3, char3, float3, op>, device::fmx<float4, char4, char4, float4, op>  },
                    { device::fmx<float, schar, schar, double, op>, device::fmx<float2, char2, char2, double2, op>, device::fmx<float3, char3, char3, double3, op>, device::fmx<float4, char4, char4, double4, op>  },
                },
                {
                    { device::fmx<float, schar, ushort, uchar, op>, device::fmx<float2, char2, ushort2, uchar2, op>, device::fmx<float3, char3, ushort3, uchar3, op>, device::fmx<float4, char4, ushort4, uchar4, op>  },
                    { device::fmx<float, schar, ushort, schar, op>, device::fmx<float2, char2, ushort2, char2, op>, device::fmx<float3, char3, ushort3, char3, op>, device::fmx<float4, char4, ushort4, char4, op>  },
                    { device::fmx<float, schar, ushort, ushort, op>, device::fmx<float2, char2, ushort2, ushort2, op>, device::fmx<float3, char3, ushort3, ushort3, op>, device::fmx<float4, char4, ushort4, ushort4, op>  },
                    { device::fmx<float, schar, ushort, short, op>, device::fmx<float2, char2, ushort2, short2, op>, device::fmx<float3, char3, ushort3, short3, op>, device::fmx<float4, char4, ushort4, short4, op>  },
                    { device::fmx<float, schar, ushort, int, op>, device::fmx<float2, char2, ushort2, int2, op>, device::fmx<float3, char3, ushort3, int3, op>, device::fmx<float4, char4, ushort4, int4, op>  },
                    { device::fmx<float, schar, ushort, float, op>, device::fmx<float2, char2, ushort2, float2, op>, device::fmx<float3, char3, ushort3, float3, op>, device::fmx<float4, char4, ushort4, float4, op>  },
                    { device::fmx<float, schar, ushort, double, op>, device::fmx<float2, char2, ushort2, double2, op>, device::fmx<float3, char3, ushort3, double3, op>, device::fmx<float4, char4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<float, schar, short, uchar, op>, device::fmx<float2, char2, short2, uchar2, op>, device::fmx<float3, char3, short3, uchar3, op>, device::fmx<float4, char4, short4, uchar4, op>  },
                    { device::fmx<float, schar, short, schar, op>, device::fmx<float2, char2, short2, char2, op>, device::fmx<float3, char3, short3, char3, op>, device::fmx<float4, char4, short4, char4, op>  },
                    { device::fmx<float, schar, short, ushort, op>, device::fmx<float2, char2, short2, ushort2, op>, device::fmx<float3, char3, short3, ushort3, op>, device::fmx<float4, char4, short4, ushort4, op>  },
                    { device::fmx<float, schar, short, short, op>, device::fmx<float2, char2, short2, short2, op>, device::fmx<float3, char3, short3, short3, op>, device::fmx<float4, char4, short4, short4, op>  },
                    { device::fmx<float, schar, short, int, op>, device::fmx<float2, char2, short2, int2, op>, device::fmx<float3, char3, short3, int3, op>, device::fmx<float4, char4, short4, int4, op>  },
                    { device::fmx<float, schar, short, float, op>, device::fmx<float2, char2, short2, float2, op>, device::fmx<float3, char3, short3, float3, op>, device::fmx<float4, char4, short4, float4, op>  },
                    { device::fmx<float, schar, short, double, op>, device::fmx<float2, char2, short2, double2, op>, device::fmx<float3, char3, short3, double3, op>, device::fmx<float4, char4, short4, double4, op>  },
                },
                {
                    { device::fmx<float, schar, int, uchar, op>, device::fmx<float2, char2, int2, uchar2, op>, device::fmx<float3, char3, int3, uchar3, op>, device::fmx<float4, char4, int4, uchar4, op>  },
                    { device::fmx<float, schar, int, schar, op>, device::fmx<float2, char2, int2, char2, op>, device::fmx<float3, char3, int3, char3, op>, device::fmx<float4, char4, int4, char4, op>  },
                    { device::fmx<float, schar, int, ushort, op>, device::fmx<float2, char2, int2, ushort2, op>, device::fmx<float3, char3, int3, ushort3, op>, device::fmx<float4, char4, int4, ushort4, op>  },
                    { device::fmx<float, schar, int, short, op>, device::fmx<float2, char2, int2, short2, op>, device::fmx<float3, char3, int3, short3, op>, device::fmx<float4, char4, int4, short4, op>  },
                    { device::fmx<float, schar, int, int, op>, device::fmx<float2, char2, int2, int2, op>, device::fmx<float3, char3, int3, int3, op>, device::fmx<float4, char4, int4, int4, op>  },
                    { device::fmx<float, schar, int, float, op>, device::fmx<float2, char2, int2, float2, op>, device::fmx<float3, char3, int3, float3, op>, device::fmx<float4, char4, int4, float4, op>  },
                    { device::fmx<float, schar, int, double, op>, device::fmx<float2, char2, int2, double2, op>, device::fmx<float3, char3, int3, double3, op>, device::fmx<float4, char4, int4, double4, op>  },
                },
                {
                    { device::fmx<float, schar, float, uchar, op>, device::fmx<float2, char2, float2, uchar2, op>, device::fmx<float3, char3, float3, uchar3, op>, device::fmx<float4, char4, float4, uchar4, op>  },
                    { device::fmx<float, schar, float, schar, op>, device::fmx<float2, char2, float2, char2, op>, device::fmx<float3, char3, float3, char3, op>, device::fmx<float4, char4, float4, char4, op>  },
                    { device::fmx<float, schar, float, ushort, op>, device::fmx<float2, char2, float2, ushort2, op>, device::fmx<float3, char3, float3, ushort3, op>, device::fmx<float4, char4, float4, ushort4, op>  },
                    { device::fmx<float, schar, float, short, op>, device::fmx<float2, char2, float2, short2, op>, device::fmx<float3, char3, float3, short3, op>, device::fmx<float4, char4, float4, short4, op>  },
                    { device::fmx<float, schar, float, int, op>, device::fmx<float2, char2, float2, int2, op>, device::fmx<float3, char3, float3, int3, op>, device::fmx<float4, char4, float4, int4, op>  },
                    { device::fmx<float, schar, float, float, op>, device::fmx<float2, char2, float2, float2, op>, device::fmx<float3, char3, float3, float3, op>, device::fmx<float4, char4, float4, float4, op>  },
                    { device::fmx<float, schar, float, double, op>, device::fmx<float2, char2, float2, double2, op>, device::fmx<float3, char3, float3, double3, op>, device::fmx<float4, char4, float4, double4, op>  },
                },
                {
                    { device::fmx<float, schar, double, uchar, op>, device::fmx<float2, char2, double2, uchar2, op>, device::fmx<float3, char3, double3, uchar3, op>, device::fmx<float4, char4, double4, uchar4, op>  },
                    { device::fmx<float, schar, double, schar, op>, device::fmx<float2, char2, double2, char2, op>, device::fmx<float3, char3, double3, char3, op>, device::fmx<float4, char4, double4, char4, op>  },
                    { device::fmx<float, schar, double, ushort, op>, device::fmx<float2, char2, double2, ushort2, op>, device::fmx<float3, char3, double3, ushort3, op>, device::fmx<float4, char4, double4, ushort4, op>  },
                    { device::fmx<float, schar, double, short, op>, device::fmx<float2, char2, double2, short2, op>, device::fmx<float3, char3, double3, short3, op>, device::fmx<float4, char4, double4, short4, op>  },
                    { device::fmx<float, schar, double, int, op>, device::fmx<float2, char2, double2, int2, op>, device::fmx<float3, char3, double3, int3, op>, device::fmx<float4, char4, double4, int4, op>  },
                    { device::fmx<float, schar, double, float, op>, device::fmx<float2, char2, double2, float2, op>, device::fmx<float3, char3, double3, float3, op>, device::fmx<float4, char4, double4, float4, op>  },
                    { device::fmx<float, schar, double, double, op>, device::fmx<float2, char2, double2, double2, op>, device::fmx<float3, char3, double3, double3, op>, device::fmx<float4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<float, ushort, uchar, uchar, op>, device::fmx<float2, ushort2, uchar2, uchar2, op>, device::fmx<float3, ushort3, uchar3, uchar3, op>, device::fmx<float4, ushort4, uchar4, uchar4, op>  },
                    { device::fmx<float, ushort, uchar, schar, op>, device::fmx<float2, ushort2, uchar2, char2, op>, device::fmx<float3, ushort3, uchar3, char3, op>, device::fmx<float4, ushort4, uchar4, char4, op>  },
                    { device::fmx<float, ushort, uchar, ushort, op>, device::fmx<float2, ushort2, uchar2, ushort2, op>, device::fmx<float3, ushort3, uchar3, ushort3, op>, device::fmx<float4, ushort4, uchar4, ushort4, op>  },
                    { device::fmx<float, ushort, uchar, short, op>, device::fmx<float2, ushort2, uchar2, short2, op>, device::fmx<float3, ushort3, uchar3, short3, op>, device::fmx<float4, ushort4, uchar4, short4, op>  },
                    { device::fmx<float, ushort, uchar, int, op>, device::fmx<float2, ushort2, uchar2, int2, op>, device::fmx<float3, ushort3, uchar3, int3, op>, device::fmx<float4, ushort4, uchar4, int4, op>  },
                    { device::fmx<float, ushort, uchar, float, op>, device::fmx<float2, ushort2, uchar2, float2, op>, device::fmx<float3, ushort3, uchar3, float3, op>, device::fmx<float4, ushort4, uchar4, float4, op>  },
                    { device::fmx<float, ushort, uchar, double, op>, device::fmx<float2, ushort2, uchar2, double2, op>, device::fmx<float3, ushort3, uchar3, double3, op>, device::fmx<float4, ushort4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<float, ushort, schar, uchar, op>, device::fmx<float2, ushort2, char2, uchar2, op>, device::fmx<float3, ushort3, char3, uchar3, op>, device::fmx<float4, ushort4, char4, uchar4, op>  },
                    { device::fmx<float, ushort, schar, schar, op>, device::fmx<float2, ushort2, char2, char2, op>, device::fmx<float3, ushort3, char3, char3, op>, device::fmx<float4, ushort4, char4, char4, op>  },
                    { device::fmx<float, ushort, schar, ushort, op>, device::fmx<float2, ushort2, char2, ushort2, op>, device::fmx<float3, ushort3, char3, ushort3, op>, device::fmx<float4, ushort4, char4, ushort4, op>  },
                    { device::fmx<float, ushort, schar, short, op>, device::fmx<float2, ushort2, char2, short2, op>, device::fmx<float3, ushort3, char3, short3, op>, device::fmx<float4, ushort4, char4, short4, op>  },
                    { device::fmx<float, ushort, schar, int, op>, device::fmx<float2, ushort2, char2, int2, op>, device::fmx<float3, ushort3, char3, int3, op>, device::fmx<float4, ushort4, char4, int4, op>  },
                    { device::fmx<float, ushort, schar, float, op>, device::fmx<float2, ushort2, char2, float2, op>, device::fmx<float3, ushort3, char3, float3, op>, device::fmx<float4, ushort4, char4, float4, op>  },
                    { device::fmx<float, ushort, schar, double, op>, device::fmx<float2, ushort2, char2, double2, op>, device::fmx<float3, ushort3, char3, double3, op>, device::fmx<float4, ushort4, char4, double4, op>  },
                },
                {
                    { device::fmx<float, ushort, ushort, uchar, op>, device::fmx<float2, ushort2, ushort2, uchar2, op>, device::fmx<float3, ushort3, ushort3, uchar3, op>, device::fmx<float4, ushort4, ushort4, uchar4, op>  },
                    { device::fmx<float, ushort, ushort, schar, op>, device::fmx<float2, ushort2, ushort2, char2, op>, device::fmx<float3, ushort3, ushort3, char3, op>, device::fmx<float4, ushort4, ushort4, char4, op>  },
                    { device::fmx<float, ushort, ushort, ushort, op>, device::fmx<float2, ushort2, ushort2, ushort2, op>, device::fmx<float3, ushort3, ushort3, ushort3, op>, device::fmx<float4, ushort4, ushort4, ushort4, op>  },
                    { device::fmx<float, ushort, ushort, short, op>, device::fmx<float2, ushort2, ushort2, short2, op>, device::fmx<float3, ushort3, ushort3, short3, op>, device::fmx<float4, ushort4, ushort4, short4, op>  },
                    { device::fmx<float, ushort, ushort, int, op>, device::fmx<float2, ushort2, ushort2, int2, op>, device::fmx<float3, ushort3, ushort3, int3, op>, device::fmx<float4, ushort4, ushort4, int4, op>  },
                    { device::fmx<float, ushort, ushort, float, op>, device::fmx<float2, ushort2, ushort2, float2, op>, device::fmx<float3, ushort3, ushort3, float3, op>, device::fmx<float4, ushort4, ushort4, float4, op>  },
                    { device::fmx<float, ushort, ushort, double, op>, device::fmx<float2, ushort2, ushort2, double2, op>, device::fmx<float3, ushort3, ushort3, double3, op>, device::fmx<float4, ushort4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<float, ushort, short, uchar, op>, device::fmx<float2, ushort2, short2, uchar2, op>, device::fmx<float3, ushort3, short3, uchar3, op>, device::fmx<float4, ushort4, short4, uchar4, op>  },
                    { device::fmx<float, ushort, short, schar, op>, device::fmx<float2, ushort2, short2, char2, op>, device::fmx<float3, ushort3, short3, char3, op>, device::fmx<float4, ushort4, short4, char4, op>  },
                    { device::fmx<float, ushort, short, ushort, op>, device::fmx<float2, ushort2, short2, ushort2, op>, device::fmx<float3, ushort3, short3, ushort3, op>, device::fmx<float4, ushort4, short4, ushort4, op>  },
                    { device::fmx<float, ushort, short, short, op>, device::fmx<float2, ushort2, short2, short2, op>, device::fmx<float3, ushort3, short3, short3, op>, device::fmx<float4, ushort4, short4, short4, op>  },
                    { device::fmx<float, ushort, short, int, op>, device::fmx<float2, ushort2, short2, int2, op>, device::fmx<float3, ushort3, short3, int3, op>, device::fmx<float4, ushort4, short4, int4, op>  },
                    { device::fmx<float, ushort, short, float, op>, device::fmx<float2, ushort2, short2, float2, op>, device::fmx<float3, ushort3, short3, float3, op>, device::fmx<float4, ushort4, short4, float4, op>  },
                    { device::fmx<float, ushort, short, double, op>, device::fmx<float2, ushort2, short2, double2, op>, device::fmx<float3, ushort3, short3, double3, op>, device::fmx<float4, ushort4, short4, double4, op>  },
                },
                {
                    { device::fmx<float, ushort, int, uchar, op>, device::fmx<float2, ushort2, int2, uchar2, op>, device::fmx<float3, ushort3, int3, uchar3, op>, device::fmx<float4, ushort4, int4, uchar4, op>  },
                    { device::fmx<float, ushort, int, schar, op>, device::fmx<float2, ushort2, int2, char2, op>, device::fmx<float3, ushort3, int3, char3, op>, device::fmx<float4, ushort4, int4, char4, op>  },
                    { device::fmx<float, ushort, int, ushort, op>, device::fmx<float2, ushort2, int2, ushort2, op>, device::fmx<float3, ushort3, int3, ushort3, op>, device::fmx<float4, ushort4, int4, ushort4, op>  },
                    { device::fmx<float, ushort, int, short, op>, device::fmx<float2, ushort2, int2, short2, op>, device::fmx<float3, ushort3, int3, short3, op>, device::fmx<float4, ushort4, int4, short4, op>  },
                    { device::fmx<float, ushort, int, int, op>, device::fmx<float2, ushort2, int2, int2, op>, device::fmx<float3, ushort3, int3, int3, op>, device::fmx<float4, ushort4, int4, int4, op>  },
                    { device::fmx<float, ushort, int, float, op>, device::fmx<float2, ushort2, int2, float2, op>, device::fmx<float3, ushort3, int3, float3, op>, device::fmx<float4, ushort4, int4, float4, op>  },
                    { device::fmx<float, ushort, int, double, op>, device::fmx<float2, ushort2, int2, double2, op>, device::fmx<float3, ushort3, int3, double3, op>, device::fmx<float4, ushort4, int4, double4, op>  },
                },
                {
                    { device::fmx<float, ushort, float, uchar, op>, device::fmx<float2, ushort2, float2, uchar2, op>, device::fmx<float3, ushort3, float3, uchar3, op>, device::fmx<float4, ushort4, float4, uchar4, op>  },
                    { device::fmx<float, ushort, float, schar, op>, device::fmx<float2, ushort2, float2, char2, op>, device::fmx<float3, ushort3, float3, char3, op>, device::fmx<float4, ushort4, float4, char4, op>  },
                    { device::fmx<float, ushort, float, ushort, op>, device::fmx<float2, ushort2, float2, ushort2, op>, device::fmx<float3, ushort3, float3, ushort3, op>, device::fmx<float4, ushort4, float4, ushort4, op>  },
                    { device::fmx<float, ushort, float, short, op>, device::fmx<float2, ushort2, float2, short2, op>, device::fmx<float3, ushort3, float3, short3, op>, device::fmx<float4, ushort4, float4, short4, op>  },
                    { device::fmx<float, ushort, float, int, op>, device::fmx<float2, ushort2, float2, int2, op>, device::fmx<float3, ushort3, float3, int3, op>, device::fmx<float4, ushort4, float4, int4, op>  },
                    { device::fmx<float, ushort, float, float, op>, device::fmx<float2, ushort2, float2, float2, op>, device::fmx<float3, ushort3, float3, float3, op>, device::fmx<float4, ushort4, float4, float4, op>  },
                    { device::fmx<float, ushort, float, double, op>, device::fmx<float2, ushort2, float2, double2, op>, device::fmx<float3, ushort3, float3, double3, op>, device::fmx<float4, ushort4, float4, double4, op>  },
                },
                {
                    { device::fmx<float, ushort, double, uchar, op>, device::fmx<float2, ushort2, double2, uchar2, op>, device::fmx<float3, ushort3, double3, uchar3, op>, device::fmx<float4, ushort4, double4, uchar4, op>  },
                    { device::fmx<float, ushort, double, schar, op>, device::fmx<float2, ushort2, double2, char2, op>, device::fmx<float3, ushort3, double3, char3, op>, device::fmx<float4, ushort4, double4, char4, op>  },
                    { device::fmx<float, ushort, double, ushort, op>, device::fmx<float2, ushort2, double2, ushort2, op>, device::fmx<float3, ushort3, double3, ushort3, op>, device::fmx<float4, ushort4, double4, ushort4, op>  },
                    { device::fmx<float, ushort, double, short, op>, device::fmx<float2, ushort2, double2, short2, op>, device::fmx<float3, ushort3, double3, short3, op>, device::fmx<float4, ushort4, double4, short4, op>  },
                    { device::fmx<float, ushort, double, int, op>, device::fmx<float2, ushort2, double2, int2, op>, device::fmx<float3, ushort3, double3, int3, op>, device::fmx<float4, ushort4, double4, int4, op>  },
                    { device::fmx<float, ushort, double, float, op>, device::fmx<float2, ushort2, double2, float2, op>, device::fmx<float3, ushort3, double3, float3, op>, device::fmx<float4, ushort4, double4, float4, op>  },
                    { device::fmx<float, ushort, double, double, op>, device::fmx<float2, ushort2, double2, double2, op>, device::fmx<float3, ushort3, double3, double3, op>, device::fmx<float4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<float, short, uchar, uchar, op>, device::fmx<float2, short2, uchar2, uchar2, op>, device::fmx<float3, short3, uchar3, uchar3, op>, device::fmx<float4, short4, uchar4, uchar4, op>  },
                    { device::fmx<float, short, uchar, schar, op>, device::fmx<float2, short2, uchar2, char2, op>, device::fmx<float3, short3, uchar3, char3, op>, device::fmx<float4, short4, uchar4, char4, op>  },
                    { device::fmx<float, short, uchar, ushort, op>, device::fmx<float2, short2, uchar2, ushort2, op>, device::fmx<float3, short3, uchar3, ushort3, op>, device::fmx<float4, short4, uchar4, ushort4, op>  },
                    { device::fmx<float, short, uchar, short, op>, device::fmx<float2, short2, uchar2, short2, op>, device::fmx<float3, short3, uchar3, short3, op>, device::fmx<float4, short4, uchar4, short4, op>  },
                    { device::fmx<float, short, uchar, int, op>, device::fmx<float2, short2, uchar2, int2, op>, device::fmx<float3, short3, uchar3, int3, op>, device::fmx<float4, short4, uchar4, int4, op>  },
                    { device::fmx<float, short, uchar, float, op>, device::fmx<float2, short2, uchar2, float2, op>, device::fmx<float3, short3, uchar3, float3, op>, device::fmx<float4, short4, uchar4, float4, op>  },
                    { device::fmx<float, short, uchar, double, op>, device::fmx<float2, short2, uchar2, double2, op>, device::fmx<float3, short3, uchar3, double3, op>, device::fmx<float4, short4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<float, short, schar, uchar, op>, device::fmx<float2, short2, char2, uchar2, op>, device::fmx<float3, short3, char3, uchar3, op>, device::fmx<float4, short4, char4, uchar4, op>  },
                    { device::fmx<float, short, schar, schar, op>, device::fmx<float2, short2, char2, char2, op>, device::fmx<float3, short3, char3, char3, op>, device::fmx<float4, short4, char4, char4, op>  },
                    { device::fmx<float, short, schar, ushort, op>, device::fmx<float2, short2, char2, ushort2, op>, device::fmx<float3, short3, char3, ushort3, op>, device::fmx<float4, short4, char4, ushort4, op>  },
                    { device::fmx<float, short, schar, short, op>, device::fmx<float2, short2, char2, short2, op>, device::fmx<float3, short3, char3, short3, op>, device::fmx<float4, short4, char4, short4, op>  },
                    { device::fmx<float, short, schar, int, op>, device::fmx<float2, short2, char2, int2, op>, device::fmx<float3, short3, char3, int3, op>, device::fmx<float4, short4, char4, int4, op>  },
                    { device::fmx<float, short, schar, float, op>, device::fmx<float2, short2, char2, float2, op>, device::fmx<float3, short3, char3, float3, op>, device::fmx<float4, short4, char4, float4, op>  },
                    { device::fmx<float, short, schar, double, op>, device::fmx<float2, short2, char2, double2, op>, device::fmx<float3, short3, char3, double3, op>, device::fmx<float4, short4, char4, double4, op>  },
                },
                {
                    { device::fmx<float, short, ushort, uchar, op>, device::fmx<float2, short2, ushort2, uchar2, op>, device::fmx<float3, short3, ushort3, uchar3, op>, device::fmx<float4, short4, ushort4, uchar4, op>  },
                    { device::fmx<float, short, ushort, schar, op>, device::fmx<float2, short2, ushort2, char2, op>, device::fmx<float3, short3, ushort3, char3, op>, device::fmx<float4, short4, ushort4, char4, op>  },
                    { device::fmx<float, short, ushort, ushort, op>, device::fmx<float2, short2, ushort2, ushort2, op>, device::fmx<float3, short3, ushort3, ushort3, op>, device::fmx<float4, short4, ushort4, ushort4, op>  },
                    { device::fmx<float, short, ushort, short, op>, device::fmx<float2, short2, ushort2, short2, op>, device::fmx<float3, short3, ushort3, short3, op>, device::fmx<float4, short4, ushort4, short4, op>  },
                    { device::fmx<float, short, ushort, int, op>, device::fmx<float2, short2, ushort2, int2, op>, device::fmx<float3, short3, ushort3, int3, op>, device::fmx<float4, short4, ushort4, int4, op>  },
                    { device::fmx<float, short, ushort, float, op>, device::fmx<float2, short2, ushort2, float2, op>, device::fmx<float3, short3, ushort3, float3, op>, device::fmx<float4, short4, ushort4, float4, op>  },
                    { device::fmx<float, short, ushort, double, op>, device::fmx<float2, short2, ushort2, double2, op>, device::fmx<float3, short3, ushort3, double3, op>, device::fmx<float4, short4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<float, short, short, uchar, op>, device::fmx<float2, short2, short2, uchar2, op>, device::fmx<float3, short3, short3, uchar3, op>, device::fmx<float4, short4, short4, uchar4, op>  },
                    { device::fmx<float, short, short, schar, op>, device::fmx<float2, short2, short2, char2, op>, device::fmx<float3, short3, short3, char3, op>, device::fmx<float4, short4, short4, char4, op>  },
                    { device::fmx<float, short, short, ushort, op>, device::fmx<float2, short2, short2, ushort2, op>, device::fmx<float3, short3, short3, ushort3, op>, device::fmx<float4, short4, short4, ushort4, op>  },
                    { device::fmx<float, short, short, short, op>, device::fmx<float2, short2, short2, short2, op>, device::fmx<float3, short3, short3, short3, op>, device::fmx<float4, short4, short4, short4, op>  },
                    { device::fmx<float, short, short, int, op>, device::fmx<float2, short2, short2, int2, op>, device::fmx<float3, short3, short3, int3, op>, device::fmx<float4, short4, short4, int4, op>  },
                    { device::fmx<float, short, short, float, op>, device::fmx<float2, short2, short2, float2, op>, device::fmx<float3, short3, short3, float3, op>, device::fmx<float4, short4, short4, float4, op>  },
                    { device::fmx<float, short, short, double, op>, device::fmx<float2, short2, short2, double2, op>, device::fmx<float3, short3, short3, double3, op>, device::fmx<float4, short4, short4, double4, op>  },
                },
                {
                    { device::fmx<float, short, int, uchar, op>, device::fmx<float2, short2, int2, uchar2, op>, device::fmx<float3, short3, int3, uchar3, op>, device::fmx<float4, short4, int4, uchar4, op>  },
                    { device::fmx<float, short, int, schar, op>, device::fmx<float2, short2, int2, char2, op>, device::fmx<float3, short3, int3, char3, op>, device::fmx<float4, short4, int4, char4, op>  },
                    { device::fmx<float, short, int, ushort, op>, device::fmx<float2, short2, int2, ushort2, op>, device::fmx<float3, short3, int3, ushort3, op>, device::fmx<float4, short4, int4, ushort4, op>  },
                    { device::fmx<float, short, int, short, op>, device::fmx<float2, short2, int2, short2, op>, device::fmx<float3, short3, int3, short3, op>, device::fmx<float4, short4, int4, short4, op>  },
                    { device::fmx<float, short, int, int, op>, device::fmx<float2, short2, int2, int2, op>, device::fmx<float3, short3, int3, int3, op>, device::fmx<float4, short4, int4, int4, op>  },
                    { device::fmx<float, short, int, float, op>, device::fmx<float2, short2, int2, float2, op>, device::fmx<float3, short3, int3, float3, op>, device::fmx<float4, short4, int4, float4, op>  },
                    { device::fmx<float, short, int, double, op>, device::fmx<float2, short2, int2, double2, op>, device::fmx<float3, short3, int3, double3, op>, device::fmx<float4, short4, int4, double4, op>  },
                },
                {
                    { device::fmx<float, short, float, uchar, op>, device::fmx<float2, short2, float2, uchar2, op>, device::fmx<float3, short3, float3, uchar3, op>, device::fmx<float4, short4, float4, uchar4, op>  },
                    { device::fmx<float, short, float, schar, op>, device::fmx<float2, short2, float2, char2, op>, device::fmx<float3, short3, float3, char3, op>, device::fmx<float4, short4, float4, char4, op>  },
                    { device::fmx<float, short, float, ushort, op>, device::fmx<float2, short2, float2, ushort2, op>, device::fmx<float3, short3, float3, ushort3, op>, device::fmx<float4, short4, float4, ushort4, op>  },
                    { device::fmx<float, short, float, short, op>, device::fmx<float2, short2, float2, short2, op>, device::fmx<float3, short3, float3, short3, op>, device::fmx<float4, short4, float4, short4, op>  },
                    { device::fmx<float, short, float, int, op>, device::fmx<float2, short2, float2, int2, op>, device::fmx<float3, short3, float3, int3, op>, device::fmx<float4, short4, float4, int4, op>  },
                    { device::fmx<float, short, float, float, op>, device::fmx<float2, short2, float2, float2, op>, device::fmx<float3, short3, float3, float3, op>, device::fmx<float4, short4, float4, float4, op>  },
                    { device::fmx<float, short, float, double, op>, device::fmx<float2, short2, float2, double2, op>, device::fmx<float3, short3, float3, double3, op>, device::fmx<float4, short4, float4, double4, op>  },
                },
                {
                    { device::fmx<float, short, double, uchar, op>, device::fmx<float2, short2, double2, uchar2, op>, device::fmx<float3, short3, double3, uchar3, op>, device::fmx<float4, short4, double4, uchar4, op>  },
                    { device::fmx<float, short, double, schar, op>, device::fmx<float2, short2, double2, char2, op>, device::fmx<float3, short3, double3, char3, op>, device::fmx<float4, short4, double4, char4, op>  },
                    { device::fmx<float, short, double, ushort, op>, device::fmx<float2, short2, double2, ushort2, op>, device::fmx<float3, short3, double3, ushort3, op>, device::fmx<float4, short4, double4, ushort4, op>  },
                    { device::fmx<float, short, double, short, op>, device::fmx<float2, short2, double2, short2, op>, device::fmx<float3, short3, double3, short3, op>, device::fmx<float4, short4, double4, short4, op>  },
                    { device::fmx<float, short, double, int, op>, device::fmx<float2, short2, double2, int2, op>, device::fmx<float3, short3, double3, int3, op>, device::fmx<float4, short4, double4, int4, op>  },
                    { device::fmx<float, short, double, float, op>, device::fmx<float2, short2, double2, float2, op>, device::fmx<float3, short3, double3, float3, op>, device::fmx<float4, short4, double4, float4, op>  },
                    { device::fmx<float, short, double, double, op>, device::fmx<float2, short2, double2, double2, op>, device::fmx<float3, short3, double3, double3, op>, device::fmx<float4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<float, int, uchar, uchar, op>, device::fmx<float2, int2, uchar2, uchar2, op>, device::fmx<float3, int3, uchar3, uchar3, op>, device::fmx<float4, int4, uchar4, uchar4, op>  },
                    { device::fmx<float, int, uchar, schar, op>, device::fmx<float2, int2, uchar2, char2, op>, device::fmx<float3, int3, uchar3, char3, op>, device::fmx<float4, int4, uchar4, char4, op>  },
                    { device::fmx<float, int, uchar, ushort, op>, device::fmx<float2, int2, uchar2, ushort2, op>, device::fmx<float3, int3, uchar3, ushort3, op>, device::fmx<float4, int4, uchar4, ushort4, op>  },
                    { device::fmx<float, int, uchar, short, op>, device::fmx<float2, int2, uchar2, short2, op>, device::fmx<float3, int3, uchar3, short3, op>, device::fmx<float4, int4, uchar4, short4, op>  },
                    { device::fmx<float, int, uchar, int, op>, device::fmx<float2, int2, uchar2, int2, op>, device::fmx<float3, int3, uchar3, int3, op>, device::fmx<float4, int4, uchar4, int4, op>  },
                    { device::fmx<float, int, uchar, float, op>, device::fmx<float2, int2, uchar2, float2, op>, device::fmx<float3, int3, uchar3, float3, op>, device::fmx<float4, int4, uchar4, float4, op>  },
                    { device::fmx<float, int, uchar, double, op>, device::fmx<float2, int2, uchar2, double2, op>, device::fmx<float3, int3, uchar3, double3, op>, device::fmx<float4, int4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<float, int, schar, uchar, op>, device::fmx<float2, int2, char2, uchar2, op>, device::fmx<float3, int3, char3, uchar3, op>, device::fmx<float4, int4, char4, uchar4, op>  },
                    { device::fmx<float, int, schar, schar, op>, device::fmx<float2, int2, char2, char2, op>, device::fmx<float3, int3, char3, char3, op>, device::fmx<float4, int4, char4, char4, op>  },
                    { device::fmx<float, int, schar, ushort, op>, device::fmx<float2, int2, char2, ushort2, op>, device::fmx<float3, int3, char3, ushort3, op>, device::fmx<float4, int4, char4, ushort4, op>  },
                    { device::fmx<float, int, schar, short, op>, device::fmx<float2, int2, char2, short2, op>, device::fmx<float3, int3, char3, short3, op>, device::fmx<float4, int4, char4, short4, op>  },
                    { device::fmx<float, int, schar, int, op>, device::fmx<float2, int2, char2, int2, op>, device::fmx<float3, int3, char3, int3, op>, device::fmx<float4, int4, char4, int4, op>  },
                    { device::fmx<float, int, schar, float, op>, device::fmx<float2, int2, char2, float2, op>, device::fmx<float3, int3, char3, float3, op>, device::fmx<float4, int4, char4, float4, op>  },
                    { device::fmx<float, int, schar, double, op>, device::fmx<float2, int2, char2, double2, op>, device::fmx<float3, int3, char3, double3, op>, device::fmx<float4, int4, char4, double4, op>  },
                },
                {
                    { device::fmx<float, int, ushort, uchar, op>, device::fmx<float2, int2, ushort2, uchar2, op>, device::fmx<float3, int3, ushort3, uchar3, op>, device::fmx<float4, int4, ushort4, uchar4, op>  },
                    { device::fmx<float, int, ushort, schar, op>, device::fmx<float2, int2, ushort2, char2, op>, device::fmx<float3, int3, ushort3, char3, op>, device::fmx<float4, int4, ushort4, char4, op>  },
                    { device::fmx<float, int, ushort, ushort, op>, device::fmx<float2, int2, ushort2, ushort2, op>, device::fmx<float3, int3, ushort3, ushort3, op>, device::fmx<float4, int4, ushort4, ushort4, op>  },
                    { device::fmx<float, int, ushort, short, op>, device::fmx<float2, int2, ushort2, short2, op>, device::fmx<float3, int3, ushort3, short3, op>, device::fmx<float4, int4, ushort4, short4, op>  },
                    { device::fmx<float, int, ushort, int, op>, device::fmx<float2, int2, ushort2, int2, op>, device::fmx<float3, int3, ushort3, int3, op>, device::fmx<float4, int4, ushort4, int4, op>  },
                    { device::fmx<float, int, ushort, float, op>, device::fmx<float2, int2, ushort2, float2, op>, device::fmx<float3, int3, ushort3, float3, op>, device::fmx<float4, int4, ushort4, float4, op>  },
                    { device::fmx<float, int, ushort, double, op>, device::fmx<float2, int2, ushort2, double2, op>, device::fmx<float3, int3, ushort3, double3, op>, device::fmx<float4, int4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<float, int, short, uchar, op>, device::fmx<float2, int2, short2, uchar2, op>, device::fmx<float3, int3, short3, uchar3, op>, device::fmx<float4, int4, short4, uchar4, op>  },
                    { device::fmx<float, int, short, schar, op>, device::fmx<float2, int2, short2, char2, op>, device::fmx<float3, int3, short3, char3, op>, device::fmx<float4, int4, short4, char4, op>  },
                    { device::fmx<float, int, short, ushort, op>, device::fmx<float2, int2, short2, ushort2, op>, device::fmx<float3, int3, short3, ushort3, op>, device::fmx<float4, int4, short4, ushort4, op>  },
                    { device::fmx<float, int, short, short, op>, device::fmx<float2, int2, short2, short2, op>, device::fmx<float3, int3, short3, short3, op>, device::fmx<float4, int4, short4, short4, op>  },
                    { device::fmx<float, int, short, int, op>, device::fmx<float2, int2, short2, int2, op>, device::fmx<float3, int3, short3, int3, op>, device::fmx<float4, int4, short4, int4, op>  },
                    { device::fmx<float, int, short, float, op>, device::fmx<float2, int2, short2, float2, op>, device::fmx<float3, int3, short3, float3, op>, device::fmx<float4, int4, short4, float4, op>  },
                    { device::fmx<float, int, short, double, op>, device::fmx<float2, int2, short2, double2, op>, device::fmx<float3, int3, short3, double3, op>, device::fmx<float4, int4, short4, double4, op>  },
                },
                {
                    { device::fmx<float, int, int, uchar, op>, device::fmx<float2, int2, int2, uchar2, op>, device::fmx<float3, int3, int3, uchar3, op>, device::fmx<float4, int4, int4, uchar4, op>  },
                    { device::fmx<float, int, int, schar, op>, device::fmx<float2, int2, int2, char2, op>, device::fmx<float3, int3, int3, char3, op>, device::fmx<float4, int4, int4, char4, op>  },
                    { device::fmx<float, int, int, ushort, op>, device::fmx<float2, int2, int2, ushort2, op>, device::fmx<float3, int3, int3, ushort3, op>, device::fmx<float4, int4, int4, ushort4, op>  },
                    { device::fmx<float, int, int, short, op>, device::fmx<float2, int2, int2, short2, op>, device::fmx<float3, int3, int3, short3, op>, device::fmx<float4, int4, int4, short4, op>  },
                    { device::fmx<float, int, int, int, op>, device::fmx<float2, int2, int2, int2, op>, device::fmx<float3, int3, int3, int3, op>, device::fmx<float4, int4, int4, int4, op>  },
                    { device::fmx<float, int, int, float, op>, device::fmx<float2, int2, int2, float2, op>, device::fmx<float3, int3, int3, float3, op>, device::fmx<float4, int4, int4, float4, op>  },
                    { device::fmx<float, int, int, double, op>, device::fmx<float2, int2, int2, double2, op>, device::fmx<float3, int3, int3, double3, op>, device::fmx<float4, int4, int4, double4, op>  },
                },
                {
                    { device::fmx<float, int, float, uchar, op>, device::fmx<float2, int2, float2, uchar2, op>, device::fmx<float3, int3, float3, uchar3, op>, device::fmx<float4, int4, float4, uchar4, op>  },
                    { device::fmx<float, int, float, schar, op>, device::fmx<float2, int2, float2, char2, op>, device::fmx<float3, int3, float3, char3, op>, device::fmx<float4, int4, float4, char4, op>  },
                    { device::fmx<float, int, float, ushort, op>, device::fmx<float2, int2, float2, ushort2, op>, device::fmx<float3, int3, float3, ushort3, op>, device::fmx<float4, int4, float4, ushort4, op>  },
                    { device::fmx<float, int, float, short, op>, device::fmx<float2, int2, float2, short2, op>, device::fmx<float3, int3, float3, short3, op>, device::fmx<float4, int4, float4, short4, op>  },
                    { device::fmx<float, int, float, int, op>, device::fmx<float2, int2, float2, int2, op>, device::fmx<float3, int3, float3, int3, op>, device::fmx<float4, int4, float4, int4, op>  },
                    { device::fmx<float, int, float, float, op>, device::fmx<float2, int2, float2, float2, op>, device::fmx<float3, int3, float3, float3, op>, device::fmx<float4, int4, float4, float4, op>  },
                    { device::fmx<float, int, float, double, op>, device::fmx<float2, int2, float2, double2, op>, device::fmx<float3, int3, float3, double3, op>, device::fmx<float4, int4, float4, double4, op>  },
                },
                {
                    { device::fmx<float, int, double, uchar, op>, device::fmx<float2, int2, double2, uchar2, op>, device::fmx<float3, int3, double3, uchar3, op>, device::fmx<float4, int4, double4, uchar4, op>  },
                    { device::fmx<float, int, double, schar, op>, device::fmx<float2, int2, double2, char2, op>, device::fmx<float3, int3, double3, char3, op>, device::fmx<float4, int4, double4, char4, op>  },
                    { device::fmx<float, int, double, ushort, op>, device::fmx<float2, int2, double2, ushort2, op>, device::fmx<float3, int3, double3, ushort3, op>, device::fmx<float4, int4, double4, ushort4, op>  },
                    { device::fmx<float, int, double, short, op>, device::fmx<float2, int2, double2, short2, op>, device::fmx<float3, int3, double3, short3, op>, device::fmx<float4, int4, double4, short4, op>  },
                    { device::fmx<float, int, double, int, op>, device::fmx<float2, int2, double2, int2, op>, device::fmx<float3, int3, double3, int3, op>, device::fmx<float4, int4, double4, int4, op>  },
                    { device::fmx<float, int, double, float, op>, device::fmx<float2, int2, double2, float2, op>, device::fmx<float3, int3, double3, float3, op>, device::fmx<float4, int4, double4, float4, op>  },
                    { device::fmx<float, int, double, double, op>, device::fmx<float2, int2, double2, double2, op>, device::fmx<float3, int3, double3, double3, op>, device::fmx<float4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<float, float, uchar, uchar, op>, device::fmx<float2, float2, uchar2, uchar2, op>, device::fmx<float3, float3, uchar3, uchar3, op>, device::fmx<float4, float4, uchar4, uchar4, op>  },
                    { device::fmx<float, float, uchar, schar, op>, device::fmx<float2, float2, uchar2, char2, op>, device::fmx<float3, float3, uchar3, char3, op>, device::fmx<float4, float4, uchar4, char4, op>  },
                    { device::fmx<float, float, uchar, ushort, op>, device::fmx<float2, float2, uchar2, ushort2, op>, device::fmx<float3, float3, uchar3, ushort3, op>, device::fmx<float4, float4, uchar4, ushort4, op>  },
                    { device::fmx<float, float, uchar, short, op>, device::fmx<float2, float2, uchar2, short2, op>, device::fmx<float3, float3, uchar3, short3, op>, device::fmx<float4, float4, uchar4, short4, op>  },
                    { device::fmx<float, float, uchar, int, op>, device::fmx<float2, float2, uchar2, int2, op>, device::fmx<float3, float3, uchar3, int3, op>, device::fmx<float4, float4, uchar4, int4, op>  },
                    { device::fmx<float, float, uchar, float, op>, device::fmx<float2, float2, uchar2, float2, op>, device::fmx<float3, float3, uchar3, float3, op>, device::fmx<float4, float4, uchar4, float4, op>  },
                    { device::fmx<float, float, uchar, double, op>, device::fmx<float2, float2, uchar2, double2, op>, device::fmx<float3, float3, uchar3, double3, op>, device::fmx<float4, float4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<float, float, schar, uchar, op>, device::fmx<float2, float2, char2, uchar2, op>, device::fmx<float3, float3, char3, uchar3, op>, device::fmx<float4, float4, char4, uchar4, op>  },
                    { device::fmx<float, float, schar, schar, op>, device::fmx<float2, float2, char2, char2, op>, device::fmx<float3, float3, char3, char3, op>, device::fmx<float4, float4, char4, char4, op>  },
                    { device::fmx<float, float, schar, ushort, op>, device::fmx<float2, float2, char2, ushort2, op>, device::fmx<float3, float3, char3, ushort3, op>, device::fmx<float4, float4, char4, ushort4, op>  },
                    { device::fmx<float, float, schar, short, op>, device::fmx<float2, float2, char2, short2, op>, device::fmx<float3, float3, char3, short3, op>, device::fmx<float4, float4, char4, short4, op>  },
                    { device::fmx<float, float, schar, int, op>, device::fmx<float2, float2, char2, int2, op>, device::fmx<float3, float3, char3, int3, op>, device::fmx<float4, float4, char4, int4, op>  },
                    { device::fmx<float, float, schar, float, op>, device::fmx<float2, float2, char2, float2, op>, device::fmx<float3, float3, char3, float3, op>, device::fmx<float4, float4, char4, float4, op>  },
                    { device::fmx<float, float, schar, double, op>, device::fmx<float2, float2, char2, double2, op>, device::fmx<float3, float3, char3, double3, op>, device::fmx<float4, float4, char4, double4, op>  },
                },
                {
                    { device::fmx<float, float, ushort, uchar, op>, device::fmx<float2, float2, ushort2, uchar2, op>, device::fmx<float3, float3, ushort3, uchar3, op>, device::fmx<float4, float4, ushort4, uchar4, op>  },
                    { device::fmx<float, float, ushort, schar, op>, device::fmx<float2, float2, ushort2, char2, op>, device::fmx<float3, float3, ushort3, char3, op>, device::fmx<float4, float4, ushort4, char4, op>  },
                    { device::fmx<float, float, ushort, ushort, op>, device::fmx<float2, float2, ushort2, ushort2, op>, device::fmx<float3, float3, ushort3, ushort3, op>, device::fmx<float4, float4, ushort4, ushort4, op>  },
                    { device::fmx<float, float, ushort, short, op>, device::fmx<float2, float2, ushort2, short2, op>, device::fmx<float3, float3, ushort3, short3, op>, device::fmx<float4, float4, ushort4, short4, op>  },
                    { device::fmx<float, float, ushort, int, op>, device::fmx<float2, float2, ushort2, int2, op>, device::fmx<float3, float3, ushort3, int3, op>, device::fmx<float4, float4, ushort4, int4, op>  },
                    { device::fmx<float, float, ushort, float, op>, device::fmx<float2, float2, ushort2, float2, op>, device::fmx<float3, float3, ushort3, float3, op>, device::fmx<float4, float4, ushort4, float4, op>  },
                    { device::fmx<float, float, ushort, double, op>, device::fmx<float2, float2, ushort2, double2, op>, device::fmx<float3, float3, ushort3, double3, op>, device::fmx<float4, float4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<float, float, short, uchar, op>, device::fmx<float2, float2, short2, uchar2, op>, device::fmx<float3, float3, short3, uchar3, op>, device::fmx<float4, float4, short4, uchar4, op>  },
                    { device::fmx<float, float, short, schar, op>, device::fmx<float2, float2, short2, char2, op>, device::fmx<float3, float3, short3, char3, op>, device::fmx<float4, float4, short4, char4, op>  },
                    { device::fmx<float, float, short, ushort, op>, device::fmx<float2, float2, short2, ushort2, op>, device::fmx<float3, float3, short3, ushort3, op>, device::fmx<float4, float4, short4, ushort4, op>  },
                    { device::fmx<float, float, short, short, op>, device::fmx<float2, float2, short2, short2, op>, device::fmx<float3, float3, short3, short3, op>, device::fmx<float4, float4, short4, short4, op>  },
                    { device::fmx<float, float, short, int, op>, device::fmx<float2, float2, short2, int2, op>, device::fmx<float3, float3, short3, int3, op>, device::fmx<float4, float4, short4, int4, op>  },
                    { device::fmx<float, float, short, float, op>, device::fmx<float2, float2, short2, float2, op>, device::fmx<float3, float3, short3, float3, op>, device::fmx<float4, float4, short4, float4, op>  },
                    { device::fmx<float, float, short, double, op>, device::fmx<float2, float2, short2, double2, op>, device::fmx<float3, float3, short3, double3, op>, device::fmx<float4, float4, short4, double4, op>  },
                },
                {
                    { device::fmx<float, float, int, uchar, op>, device::fmx<float2, float2, int2, uchar2, op>, device::fmx<float3, float3, int3, uchar3, op>, device::fmx<float4, float4, int4, uchar4, op>  },
                    { device::fmx<float, float, int, schar, op>, device::fmx<float2, float2, int2, char2, op>, device::fmx<float3, float3, int3, char3, op>, device::fmx<float4, float4, int4, char4, op>  },
                    { device::fmx<float, float, int, ushort, op>, device::fmx<float2, float2, int2, ushort2, op>, device::fmx<float3, float3, int3, ushort3, op>, device::fmx<float4, float4, int4, ushort4, op>  },
                    { device::fmx<float, float, int, short, op>, device::fmx<float2, float2, int2, short2, op>, device::fmx<float3, float3, int3, short3, op>, device::fmx<float4, float4, int4, short4, op>  },
                    { device::fmx<float, float, int, int, op>, device::fmx<float2, float2, int2, int2, op>, device::fmx<float3, float3, int3, int3, op>, device::fmx<float4, float4, int4, int4, op>  },
                    { device::fmx<float, float, int, float, op>, device::fmx<float2, float2, int2, float2, op>, device::fmx<float3, float3, int3, float3, op>, device::fmx<float4, float4, int4, float4, op>  },
                    { device::fmx<float, float, int, double, op>, device::fmx<float2, float2, int2, double2, op>, device::fmx<float3, float3, int3, double3, op>, device::fmx<float4, float4, int4, double4, op>  },
                },
                {
                    { device::fmx<float, float, float, uchar, op>, device::fmx<float2, float2, float2, uchar2, op>, device::fmx<float3, float3, float3, uchar3, op>, device::fmx<float4, float4, float4, uchar4, op>  },
                    { device::fmx<float, float, float, schar, op>, device::fmx<float2, float2, float2, char2, op>, device::fmx<float3, float3, float3, char3, op>, device::fmx<float4, float4, float4, char4, op>  },
                    { device::fmx<float, float, float, ushort, op>, device::fmx<float2, float2, float2, ushort2, op>, device::fmx<float3, float3, float3, ushort3, op>, device::fmx<float4, float4, float4, ushort4, op>  },
                    { device::fmx<float, float, float, short, op>, device::fmx<float2, float2, float2, short2, op>, device::fmx<float3, float3, float3, short3, op>, device::fmx<float4, float4, float4, short4, op>  },
                    { device::fmx<float, float, float, int, op>, device::fmx<float2, float2, float2, int2, op>, device::fmx<float3, float3, float3, int3, op>, device::fmx<float4, float4, float4, int4, op>  },
                    { device::fmx<float, float, float, float, op>, device::fmx<float2, float2, float2, float2, op>, device::fmx<float3, float3, float3, float3, op>, device::fmx<float4, float4, float4, float4, op>  },
                    { device::fmx<float, float, float, double, op>, device::fmx<float2, float2, float2, double2, op>, device::fmx<float3, float3, float3, double3, op>, device::fmx<float4, float4, float4, double4, op>  },
                },
                {
                    { device::fmx<float, float, double, uchar, op>, device::fmx<float2, float2, double2, uchar2, op>, device::fmx<float3, float3, double3, uchar3, op>, device::fmx<float4, float4, double4, uchar4, op>  },
                    { device::fmx<float, float, double, schar, op>, device::fmx<float2, float2, double2, char2, op>, device::fmx<float3, float3, double3, char3, op>, device::fmx<float4, float4, double4, char4, op>  },
                    { device::fmx<float, float, double, ushort, op>, device::fmx<float2, float2, double2, ushort2, op>, device::fmx<float3, float3, double3, ushort3, op>, device::fmx<float4, float4, double4, ushort4, op>  },
                    { device::fmx<float, float, double, short, op>, device::fmx<float2, float2, double2, short2, op>, device::fmx<float3, float3, double3, short3, op>, device::fmx<float4, float4, double4, short4, op>  },
                    { device::fmx<float, float, double, int, op>, device::fmx<float2, float2, double2, int2, op>, device::fmx<float3, float3, double3, int3, op>, device::fmx<float4, float4, double4, int4, op>  },
                    { device::fmx<float, float, double, float, op>, device::fmx<float2, float2, double2, float2, op>, device::fmx<float3, float3, double3, float3, op>, device::fmx<float4, float4, double4, float4, op>  },
                    { device::fmx<float, float, double, double, op>, device::fmx<float2, float2, double2, double2, op>, device::fmx<float3, float3, double3, double3, op>, device::fmx<float4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<float, double, uchar, uchar, op>, device::fmx<float2, double2, uchar2, uchar2, op>, device::fmx<float3, double3, uchar3, uchar3, op>, device::fmx<float4, double4, uchar4, uchar4, op>  },
                    { device::fmx<float, double, uchar, schar, op>, device::fmx<float2, double2, uchar2, char2, op>, device::fmx<float3, double3, uchar3, char3, op>, device::fmx<float4, double4, uchar4, char4, op>  },
                    { device::fmx<float, double, uchar, ushort, op>, device::fmx<float2, double2, uchar2, ushort2, op>, device::fmx<float3, double3, uchar3, ushort3, op>, device::fmx<float4, double4, uchar4, ushort4, op>  },
                    { device::fmx<float, double, uchar, short, op>, device::fmx<float2, double2, uchar2, short2, op>, device::fmx<float3, double3, uchar3, short3, op>, device::fmx<float4, double4, uchar4, short4, op>  },
                    { device::fmx<float, double, uchar, int, op>, device::fmx<float2, double2, uchar2, int2, op>, device::fmx<float3, double3, uchar3, int3, op>, device::fmx<float4, double4, uchar4, int4, op>  },
                    { device::fmx<float, double, uchar, float, op>, device::fmx<float2, double2, uchar2, float2, op>, device::fmx<float3, double3, uchar3, float3, op>, device::fmx<float4, double4, uchar4, float4, op>  },
                    { device::fmx<float, double, uchar, double, op>, device::fmx<float2, double2, uchar2, double2, op>, device::fmx<float3, double3, uchar3, double3, op>, device::fmx<float4, double4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<float, double, schar, uchar, op>, device::fmx<float2, double2, char2, uchar2, op>, device::fmx<float3, double3, char3, uchar3, op>, device::fmx<float4, double4, char4, uchar4, op>  },
                    { device::fmx<float, double, schar, schar, op>, device::fmx<float2, double2, char2, char2, op>, device::fmx<float3, double3, char3, char3, op>, device::fmx<float4, double4, char4, char4, op>  },
                    { device::fmx<float, double, schar, ushort, op>, device::fmx<float2, double2, char2, ushort2, op>, device::fmx<float3, double3, char3, ushort3, op>, device::fmx<float4, double4, char4, ushort4, op>  },
                    { device::fmx<float, double, schar, short, op>, device::fmx<float2, double2, char2, short2, op>, device::fmx<float3, double3, char3, short3, op>, device::fmx<float4, double4, char4, short4, op>  },
                    { device::fmx<float, double, schar, int, op>, device::fmx<float2, double2, char2, int2, op>, device::fmx<float3, double3, char3, int3, op>, device::fmx<float4, double4, char4, int4, op>  },
                    { device::fmx<float, double, schar, float, op>, device::fmx<float2, double2, char2, float2, op>, device::fmx<float3, double3, char3, float3, op>, device::fmx<float4, double4, char4, float4, op>  },
                    { device::fmx<float, double, schar, double, op>, device::fmx<float2, double2, char2, double2, op>, device::fmx<float3, double3, char3, double3, op>, device::fmx<float4, double4, char4, double4, op>  },
                },
                {
                    { device::fmx<float, double, ushort, uchar, op>, device::fmx<float2, double2, ushort2, uchar2, op>, device::fmx<float3, double3, ushort3, uchar3, op>, device::fmx<float4, double4, ushort4, uchar4, op>  },
                    { device::fmx<float, double, ushort, schar, op>, device::fmx<float2, double2, ushort2, char2, op>, device::fmx<float3, double3, ushort3, char3, op>, device::fmx<float4, double4, ushort4, char4, op>  },
                    { device::fmx<float, double, ushort, ushort, op>, device::fmx<float2, double2, ushort2, ushort2, op>, device::fmx<float3, double3, ushort3, ushort3, op>, device::fmx<float4, double4, ushort4, ushort4, op>  },
                    { device::fmx<float, double, ushort, short, op>, device::fmx<float2, double2, ushort2, short2, op>, device::fmx<float3, double3, ushort3, short3, op>, device::fmx<float4, double4, ushort4, short4, op>  },
                    { device::fmx<float, double, ushort, int, op>, device::fmx<float2, double2, ushort2, int2, op>, device::fmx<float3, double3, ushort3, int3, op>, device::fmx<float4, double4, ushort4, int4, op>  },
                    { device::fmx<float, double, ushort, float, op>, device::fmx<float2, double2, ushort2, float2, op>, device::fmx<float3, double3, ushort3, float3, op>, device::fmx<float4, double4, ushort4, float4, op>  },
                    { device::fmx<float, double, ushort, double, op>, device::fmx<float2, double2, ushort2, double2, op>, device::fmx<float3, double3, ushort3, double3, op>, device::fmx<float4, double4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<float, double, short, uchar, op>, device::fmx<float2, double2, short2, uchar2, op>, device::fmx<float3, double3, short3, uchar3, op>, device::fmx<float4, double4, short4, uchar4, op>  },
                    { device::fmx<float, double, short, schar, op>, device::fmx<float2, double2, short2, char2, op>, device::fmx<float3, double3, short3, char3, op>, device::fmx<float4, double4, short4, char4, op>  },
                    { device::fmx<float, double, short, ushort, op>, device::fmx<float2, double2, short2, ushort2, op>, device::fmx<float3, double3, short3, ushort3, op>, device::fmx<float4, double4, short4, ushort4, op>  },
                    { device::fmx<float, double, short, short, op>, device::fmx<float2, double2, short2, short2, op>, device::fmx<float3, double3, short3, short3, op>, device::fmx<float4, double4, short4, short4, op>  },
                    { device::fmx<float, double, short, int, op>, device::fmx<float2, double2, short2, int2, op>, device::fmx<float3, double3, short3, int3, op>, device::fmx<float4, double4, short4, int4, op>  },
                    { device::fmx<float, double, short, float, op>, device::fmx<float2, double2, short2, float2, op>, device::fmx<float3, double3, short3, float3, op>, device::fmx<float4, double4, short4, float4, op>  },
                    { device::fmx<float, double, short, double, op>, device::fmx<float2, double2, short2, double2, op>, device::fmx<float3, double3, short3, double3, op>, device::fmx<float4, double4, short4, double4, op>  },
                },
                {
                    { device::fmx<float, double, int, uchar, op>, device::fmx<float2, double2, int2, uchar2, op>, device::fmx<float3, double3, int3, uchar3, op>, device::fmx<float4, double4, int4, uchar4, op>  },
                    { device::fmx<float, double, int, schar, op>, device::fmx<float2, double2, int2, char2, op>, device::fmx<float3, double3, int3, char3, op>, device::fmx<float4, double4, int4, char4, op>  },
                    { device::fmx<float, double, int, ushort, op>, device::fmx<float2, double2, int2, ushort2, op>, device::fmx<float3, double3, int3, ushort3, op>, device::fmx<float4, double4, int4, ushort4, op>  },
                    { device::fmx<float, double, int, short, op>, device::fmx<float2, double2, int2, short2, op>, device::fmx<float3, double3, int3, short3, op>, device::fmx<float4, double4, int4, short4, op>  },
                    { device::fmx<float, double, int, int, op>, device::fmx<float2, double2, int2, int2, op>, device::fmx<float3, double3, int3, int3, op>, device::fmx<float4, double4, int4, int4, op>  },
                    { device::fmx<float, double, int, float, op>, device::fmx<float2, double2, int2, float2, op>, device::fmx<float3, double3, int3, float3, op>, device::fmx<float4, double4, int4, float4, op>  },
                    { device::fmx<float, double, int, double, op>, device::fmx<float2, double2, int2, double2, op>, device::fmx<float3, double3, int3, double3, op>, device::fmx<float4, double4, int4, double4, op>  },
                },
                {
                    { device::fmx<float, double, float, uchar, op>, device::fmx<float2, double2, float2, uchar2, op>, device::fmx<float3, double3, float3, uchar3, op>, device::fmx<float4, double4, float4, uchar4, op>  },
                    { device::fmx<float, double, float, schar, op>, device::fmx<float2, double2, float2, char2, op>, device::fmx<float3, double3, float3, char3, op>, device::fmx<float4, double4, float4, char4, op>  },
                    { device::fmx<float, double, float, ushort, op>, device::fmx<float2, double2, float2, ushort2, op>, device::fmx<float3, double3, float3, ushort3, op>, device::fmx<float4, double4, float4, ushort4, op>  },
                    { device::fmx<float, double, float, short, op>, device::fmx<float2, double2, float2, short2, op>, device::fmx<float3, double3, float3, short3, op>, device::fmx<float4, double4, float4, short4, op>  },
                    { device::fmx<float, double, float, int, op>, device::fmx<float2, double2, float2, int2, op>, device::fmx<float3, double3, float3, int3, op>, device::fmx<float4, double4, float4, int4, op>  },
                    { device::fmx<float, double, float, float, op>, device::fmx<float2, double2, float2, float2, op>, device::fmx<float3, double3, float3, float3, op>, device::fmx<float4, double4, float4, float4, op>  },
                    { device::fmx<float, double, float, double, op>, device::fmx<float2, double2, float2, double2, op>, device::fmx<float3, double3, float3, double3, op>, device::fmx<float4, double4, float4, double4, op>  },
                },
                {
                    { device::fmx<float, double, double, uchar, op>, device::fmx<float2, double2, double2, uchar2, op>, device::fmx<float3, double3, double3, uchar3, op>, device::fmx<float4, double4, double4, uchar4, op>  },
                    { device::fmx<float, double, double, schar, op>, device::fmx<float2, double2, double2, char2, op>, device::fmx<float3, double3, double3, char3, op>, device::fmx<float4, double4, double4, char4, op>  },
                    { device::fmx<float, double, double, ushort, op>, device::fmx<float2, double2, double2, ushort2, op>, device::fmx<float3, double3, double3, ushort3, op>, device::fmx<float4, double4, double4, ushort4, op>  },
                    { device::fmx<float, double, double, short, op>, device::fmx<float2, double2, double2, short2, op>, device::fmx<float3, double3, double3, short3, op>, device::fmx<float4, double4, double4, short4, op>  },
                    { device::fmx<float, double, double, int, op>, device::fmx<float2, double2, double2, int2, op>, device::fmx<float3, double3, double3, int3, op>, device::fmx<float4, double4, double4, int4, op>  },
                    { device::fmx<float, double, double, float, op>, device::fmx<float2, double2, double2, float2, op>, device::fmx<float3, double3, double3, float3, op>, device::fmx<float4, double4, double4, float4, op>  },
                    { device::fmx<float, double, double, double, op>, device::fmx<float2, double2, double2, double2, op>, device::fmx<float3, double3, double3, double3, op>, device::fmx<float4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { device::fmx<double, uchar, uchar, uchar, op>, device::fmx<double2, uchar2, uchar2, uchar2, op>, device::fmx<double3, uchar3, uchar3, uchar3, op>, device::fmx<double4, uchar4, uchar4, uchar4, op>  },
                    { device::fmx<double, uchar, uchar, schar, op>, device::fmx<double2, uchar2, uchar2, char2, op>, device::fmx<double3, uchar3, uchar3, char3, op>, device::fmx<double4, uchar4, uchar4, char4, op>  },
                    { device::fmx<double, uchar, uchar, ushort, op>, device::fmx<double2, uchar2, uchar2, ushort2, op>, device::fmx<double3, uchar3, uchar3, ushort3, op>, device::fmx<double4, uchar4, uchar4, ushort4, op>  },
                    { device::fmx<double, uchar, uchar, short, op>, device::fmx<double2, uchar2, uchar2, short2, op>, device::fmx<double3, uchar3, uchar3, short3, op>, device::fmx<double4, uchar4, uchar4, short4, op>  },
                    { device::fmx<double, uchar, uchar, int, op>, device::fmx<double2, uchar2, uchar2, int2, op>, device::fmx<double3, uchar3, uchar3, int3, op>, device::fmx<double4, uchar4, uchar4, int4, op>  },
                    { device::fmx<double, uchar, uchar, float, op>, device::fmx<double2, uchar2, uchar2, float2, op>, device::fmx<double3, uchar3, uchar3, float3, op>, device::fmx<double4, uchar4, uchar4, float4, op>  },
                    { device::fmx<double, uchar, uchar, double, op>, device::fmx<double2, uchar2, uchar2, double2, op>, device::fmx<double3, uchar3, uchar3, double3, op>, device::fmx<double4, uchar4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<double, uchar, schar, uchar, op>, device::fmx<double2, uchar2, char2, uchar2, op>, device::fmx<double3, uchar3, char3, uchar3, op>, device::fmx<double4, uchar4, char4, uchar4, op>  },
                    { device::fmx<double, uchar, schar, schar, op>, device::fmx<double2, uchar2, char2, char2, op>, device::fmx<double3, uchar3, char3, char3, op>, device::fmx<double4, uchar4, char4, char4, op>  },
                    { device::fmx<double, uchar, schar, ushort, op>, device::fmx<double2, uchar2, char2, ushort2, op>, device::fmx<double3, uchar3, char3, ushort3, op>, device::fmx<double4, uchar4, char4, ushort4, op>  },
                    { device::fmx<double, uchar, schar, short, op>, device::fmx<double2, uchar2, char2, short2, op>, device::fmx<double3, uchar3, char3, short3, op>, device::fmx<double4, uchar4, char4, short4, op>  },
                    { device::fmx<double, uchar, schar, int, op>, device::fmx<double2, uchar2, char2, int2, op>, device::fmx<double3, uchar3, char3, int3, op>, device::fmx<double4, uchar4, char4, int4, op>  },
                    { device::fmx<double, uchar, schar, float, op>, device::fmx<double2, uchar2, char2, float2, op>, device::fmx<double3, uchar3, char3, float3, op>, device::fmx<double4, uchar4, char4, float4, op>  },
                    { device::fmx<double, uchar, schar, double, op>, device::fmx<double2, uchar2, char2, double2, op>, device::fmx<double3, uchar3, char3, double3, op>, device::fmx<double4, uchar4, char4, double4, op>  },
                },
                {
                    { device::fmx<double, uchar, ushort, uchar, op>, device::fmx<double2, uchar2, ushort2, uchar2, op>, device::fmx<double3, uchar3, ushort3, uchar3, op>, device::fmx<double4, uchar4, ushort4, uchar4, op>  },
                    { device::fmx<double, uchar, ushort, schar, op>, device::fmx<double2, uchar2, ushort2, char2, op>, device::fmx<double3, uchar3, ushort3, char3, op>, device::fmx<double4, uchar4, ushort4, char4, op>  },
                    { device::fmx<double, uchar, ushort, ushort, op>, device::fmx<double2, uchar2, ushort2, ushort2, op>, device::fmx<double3, uchar3, ushort3, ushort3, op>, device::fmx<double4, uchar4, ushort4, ushort4, op>  },
                    { device::fmx<double, uchar, ushort, short, op>, device::fmx<double2, uchar2, ushort2, short2, op>, device::fmx<double3, uchar3, ushort3, short3, op>, device::fmx<double4, uchar4, ushort4, short4, op>  },
                    { device::fmx<double, uchar, ushort, int, op>, device::fmx<double2, uchar2, ushort2, int2, op>, device::fmx<double3, uchar3, ushort3, int3, op>, device::fmx<double4, uchar4, ushort4, int4, op>  },
                    { device::fmx<double, uchar, ushort, float, op>, device::fmx<double2, uchar2, ushort2, float2, op>, device::fmx<double3, uchar3, ushort3, float3, op>, device::fmx<double4, uchar4, ushort4, float4, op>  },
                    { device::fmx<double, uchar, ushort, double, op>, device::fmx<double2, uchar2, ushort2, double2, op>, device::fmx<double3, uchar3, ushort3, double3, op>, device::fmx<double4, uchar4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<double, uchar, short, uchar, op>, device::fmx<double2, uchar2, short2, uchar2, op>, device::fmx<double3, uchar3, short3, uchar3, op>, device::fmx<double4, uchar4, short4, uchar4, op>  },
                    { device::fmx<double, uchar, short, schar, op>, device::fmx<double2, uchar2, short2, char2, op>, device::fmx<double3, uchar3, short3, char3, op>, device::fmx<double4, uchar4, short4, char4, op>  },
                    { device::fmx<double, uchar, short, ushort, op>, device::fmx<double2, uchar2, short2, ushort2, op>, device::fmx<double3, uchar3, short3, ushort3, op>, device::fmx<double4, uchar4, short4, ushort4, op>  },
                    { device::fmx<double, uchar, short, short, op>, device::fmx<double2, uchar2, short2, short2, op>, device::fmx<double3, uchar3, short3, short3, op>, device::fmx<double4, uchar4, short4, short4, op>  },
                    { device::fmx<double, uchar, short, int, op>, device::fmx<double2, uchar2, short2, int2, op>, device::fmx<double3, uchar3, short3, int3, op>, device::fmx<double4, uchar4, short4, int4, op>  },
                    { device::fmx<double, uchar, short, float, op>, device::fmx<double2, uchar2, short2, float2, op>, device::fmx<double3, uchar3, short3, float3, op>, device::fmx<double4, uchar4, short4, float4, op>  },
                    { device::fmx<double, uchar, short, double, op>, device::fmx<double2, uchar2, short2, double2, op>, device::fmx<double3, uchar3, short3, double3, op>, device::fmx<double4, uchar4, short4, double4, op>  },
                },
                {
                    { device::fmx<double, uchar, int, uchar, op>, device::fmx<double2, uchar2, int2, uchar2, op>, device::fmx<double3, uchar3, int3, uchar3, op>, device::fmx<double4, uchar4, int4, uchar4, op>  },
                    { device::fmx<double, uchar, int, schar, op>, device::fmx<double2, uchar2, int2, char2, op>, device::fmx<double3, uchar3, int3, char3, op>, device::fmx<double4, uchar4, int4, char4, op>  },
                    { device::fmx<double, uchar, int, ushort, op>, device::fmx<double2, uchar2, int2, ushort2, op>, device::fmx<double3, uchar3, int3, ushort3, op>, device::fmx<double4, uchar4, int4, ushort4, op>  },
                    { device::fmx<double, uchar, int, short, op>, device::fmx<double2, uchar2, int2, short2, op>, device::fmx<double3, uchar3, int3, short3, op>, device::fmx<double4, uchar4, int4, short4, op>  },
                    { device::fmx<double, uchar, int, int, op>, device::fmx<double2, uchar2, int2, int2, op>, device::fmx<double3, uchar3, int3, int3, op>, device::fmx<double4, uchar4, int4, int4, op>  },
                    { device::fmx<double, uchar, int, float, op>, device::fmx<double2, uchar2, int2, float2, op>, device::fmx<double3, uchar3, int3, float3, op>, device::fmx<double4, uchar4, int4, float4, op>  },
                    { device::fmx<double, uchar, int, double, op>, device::fmx<double2, uchar2, int2, double2, op>, device::fmx<double3, uchar3, int3, double3, op>, device::fmx<double4, uchar4, int4, double4, op>  },
                },
                {
                    { device::fmx<double, uchar, float, uchar, op>, device::fmx<double2, uchar2, float2, uchar2, op>, device::fmx<double3, uchar3, float3, uchar3, op>, device::fmx<double4, uchar4, float4, uchar4, op>  },
                    { device::fmx<double, uchar, float, schar, op>, device::fmx<double2, uchar2, float2, char2, op>, device::fmx<double3, uchar3, float3, char3, op>, device::fmx<double4, uchar4, float4, char4, op>  },
                    { device::fmx<double, uchar, float, ushort, op>, device::fmx<double2, uchar2, float2, ushort2, op>, device::fmx<double3, uchar3, float3, ushort3, op>, device::fmx<double4, uchar4, float4, ushort4, op>  },
                    { device::fmx<double, uchar, float, short, op>, device::fmx<double2, uchar2, float2, short2, op>, device::fmx<double3, uchar3, float3, short3, op>, device::fmx<double4, uchar4, float4, short4, op>  },
                    { device::fmx<double, uchar, float, int, op>, device::fmx<double2, uchar2, float2, int2, op>, device::fmx<double3, uchar3, float3, int3, op>, device::fmx<double4, uchar4, float4, int4, op>  },
                    { device::fmx<double, uchar, float, float, op>, device::fmx<double2, uchar2, float2, float2, op>, device::fmx<double3, uchar3, float3, float3, op>, device::fmx<double4, uchar4, float4, float4, op>  },
                    { device::fmx<double, uchar, float, double, op>, device::fmx<double2, uchar2, float2, double2, op>, device::fmx<double3, uchar3, float3, double3, op>, device::fmx<double4, uchar4, float4, double4, op>  },
                },
                {
                    { device::fmx<double, uchar, double, uchar, op>, device::fmx<double2, uchar2, double2, uchar2, op>, device::fmx<double3, uchar3, double3, uchar3, op>, device::fmx<double4, uchar4, double4, uchar4, op>  },
                    { device::fmx<double, uchar, double, schar, op>, device::fmx<double2, uchar2, double2, char2, op>, device::fmx<double3, uchar3, double3, char3, op>, device::fmx<double4, uchar4, double4, char4, op>  },
                    { device::fmx<double, uchar, double, ushort, op>, device::fmx<double2, uchar2, double2, ushort2, op>, device::fmx<double3, uchar3, double3, ushort3, op>, device::fmx<double4, uchar4, double4, ushort4, op>  },
                    { device::fmx<double, uchar, double, short, op>, device::fmx<double2, uchar2, double2, short2, op>, device::fmx<double3, uchar3, double3, short3, op>, device::fmx<double4, uchar4, double4, short4, op>  },
                    { device::fmx<double, uchar, double, int, op>, device::fmx<double2, uchar2, double2, int2, op>, device::fmx<double3, uchar3, double3, int3, op>, device::fmx<double4, uchar4, double4, int4, op>  },
                    { device::fmx<double, uchar, double, float, op>, device::fmx<double2, uchar2, double2, float2, op>, device::fmx<double3, uchar3, double3, float3, op>, device::fmx<double4, uchar4, double4, float4, op>  },
                    { device::fmx<double, uchar, double, double, op>, device::fmx<double2, uchar2, double2, double2, op>, device::fmx<double3, uchar3, double3, double3, op>, device::fmx<double4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<double, schar, uchar, uchar, op>, device::fmx<double2, char2, uchar2, uchar2, op>, device::fmx<double3, char3, uchar3, uchar3, op>, device::fmx<double4, char4, uchar4, uchar4, op>  },
                    { device::fmx<double, schar, uchar, schar, op>, device::fmx<double2, char2, uchar2, char2, op>, device::fmx<double3, char3, uchar3, char3, op>, device::fmx<double4, char4, uchar4, char4, op>  },
                    { device::fmx<double, schar, uchar, ushort, op>, device::fmx<double2, char2, uchar2, ushort2, op>, device::fmx<double3, char3, uchar3, ushort3, op>, device::fmx<double4, char4, uchar4, ushort4, op>  },
                    { device::fmx<double, schar, uchar, short, op>, device::fmx<double2, char2, uchar2, short2, op>, device::fmx<double3, char3, uchar3, short3, op>, device::fmx<double4, char4, uchar4, short4, op>  },
                    { device::fmx<double, schar, uchar, int, op>, device::fmx<double2, char2, uchar2, int2, op>, device::fmx<double3, char3, uchar3, int3, op>, device::fmx<double4, char4, uchar4, int4, op>  },
                    { device::fmx<double, schar, uchar, float, op>, device::fmx<double2, char2, uchar2, float2, op>, device::fmx<double3, char3, uchar3, float3, op>, device::fmx<double4, char4, uchar4, float4, op>  },
                    { device::fmx<double, schar, uchar, double, op>, device::fmx<double2, char2, uchar2, double2, op>, device::fmx<double3, char3, uchar3, double3, op>, device::fmx<double4, char4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<double, schar, schar, uchar, op>, device::fmx<double2, char2, char2, uchar2, op>, device::fmx<double3, char3, char3, uchar3, op>, device::fmx<double4, char4, char4, uchar4, op>  },
                    { device::fmx<double, schar, schar, schar, op>, device::fmx<double2, char2, char2, char2, op>, device::fmx<double3, char3, char3, char3, op>, device::fmx<double4, char4, char4, char4, op>  },
                    { device::fmx<double, schar, schar, ushort, op>, device::fmx<double2, char2, char2, ushort2, op>, device::fmx<double3, char3, char3, ushort3, op>, device::fmx<double4, char4, char4, ushort4, op>  },
                    { device::fmx<double, schar, schar, short, op>, device::fmx<double2, char2, char2, short2, op>, device::fmx<double3, char3, char3, short3, op>, device::fmx<double4, char4, char4, short4, op>  },
                    { device::fmx<double, schar, schar, int, op>, device::fmx<double2, char2, char2, int2, op>, device::fmx<double3, char3, char3, int3, op>, device::fmx<double4, char4, char4, int4, op>  },
                    { device::fmx<double, schar, schar, float, op>, device::fmx<double2, char2, char2, float2, op>, device::fmx<double3, char3, char3, float3, op>, device::fmx<double4, char4, char4, float4, op>  },
                    { device::fmx<double, schar, schar, double, op>, device::fmx<double2, char2, char2, double2, op>, device::fmx<double3, char3, char3, double3, op>, device::fmx<double4, char4, char4, double4, op>  },
                },
                {
                    { device::fmx<double, schar, ushort, uchar, op>, device::fmx<double2, char2, ushort2, uchar2, op>, device::fmx<double3, char3, ushort3, uchar3, op>, device::fmx<double4, char4, ushort4, uchar4, op>  },
                    { device::fmx<double, schar, ushort, schar, op>, device::fmx<double2, char2, ushort2, char2, op>, device::fmx<double3, char3, ushort3, char3, op>, device::fmx<double4, char4, ushort4, char4, op>  },
                    { device::fmx<double, schar, ushort, ushort, op>, device::fmx<double2, char2, ushort2, ushort2, op>, device::fmx<double3, char3, ushort3, ushort3, op>, device::fmx<double4, char4, ushort4, ushort4, op>  },
                    { device::fmx<double, schar, ushort, short, op>, device::fmx<double2, char2, ushort2, short2, op>, device::fmx<double3, char3, ushort3, short3, op>, device::fmx<double4, char4, ushort4, short4, op>  },
                    { device::fmx<double, schar, ushort, int, op>, device::fmx<double2, char2, ushort2, int2, op>, device::fmx<double3, char3, ushort3, int3, op>, device::fmx<double4, char4, ushort4, int4, op>  },
                    { device::fmx<double, schar, ushort, float, op>, device::fmx<double2, char2, ushort2, float2, op>, device::fmx<double3, char3, ushort3, float3, op>, device::fmx<double4, char4, ushort4, float4, op>  },
                    { device::fmx<double, schar, ushort, double, op>, device::fmx<double2, char2, ushort2, double2, op>, device::fmx<double3, char3, ushort3, double3, op>, device::fmx<double4, char4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<double, schar, short, uchar, op>, device::fmx<double2, char2, short2, uchar2, op>, device::fmx<double3, char3, short3, uchar3, op>, device::fmx<double4, char4, short4, uchar4, op>  },
                    { device::fmx<double, schar, short, schar, op>, device::fmx<double2, char2, short2, char2, op>, device::fmx<double3, char3, short3, char3, op>, device::fmx<double4, char4, short4, char4, op>  },
                    { device::fmx<double, schar, short, ushort, op>, device::fmx<double2, char2, short2, ushort2, op>, device::fmx<double3, char3, short3, ushort3, op>, device::fmx<double4, char4, short4, ushort4, op>  },
                    { device::fmx<double, schar, short, short, op>, device::fmx<double2, char2, short2, short2, op>, device::fmx<double3, char3, short3, short3, op>, device::fmx<double4, char4, short4, short4, op>  },
                    { device::fmx<double, schar, short, int, op>, device::fmx<double2, char2, short2, int2, op>, device::fmx<double3, char3, short3, int3, op>, device::fmx<double4, char4, short4, int4, op>  },
                    { device::fmx<double, schar, short, float, op>, device::fmx<double2, char2, short2, float2, op>, device::fmx<double3, char3, short3, float3, op>, device::fmx<double4, char4, short4, float4, op>  },
                    { device::fmx<double, schar, short, double, op>, device::fmx<double2, char2, short2, double2, op>, device::fmx<double3, char3, short3, double3, op>, device::fmx<double4, char4, short4, double4, op>  },
                },
                {
                    { device::fmx<double, schar, int, uchar, op>, device::fmx<double2, char2, int2, uchar2, op>, device::fmx<double3, char3, int3, uchar3, op>, device::fmx<double4, char4, int4, uchar4, op>  },
                    { device::fmx<double, schar, int, schar, op>, device::fmx<double2, char2, int2, char2, op>, device::fmx<double3, char3, int3, char3, op>, device::fmx<double4, char4, int4, char4, op>  },
                    { device::fmx<double, schar, int, ushort, op>, device::fmx<double2, char2, int2, ushort2, op>, device::fmx<double3, char3, int3, ushort3, op>, device::fmx<double4, char4, int4, ushort4, op>  },
                    { device::fmx<double, schar, int, short, op>, device::fmx<double2, char2, int2, short2, op>, device::fmx<double3, char3, int3, short3, op>, device::fmx<double4, char4, int4, short4, op>  },
                    { device::fmx<double, schar, int, int, op>, device::fmx<double2, char2, int2, int2, op>, device::fmx<double3, char3, int3, int3, op>, device::fmx<double4, char4, int4, int4, op>  },
                    { device::fmx<double, schar, int, float, op>, device::fmx<double2, char2, int2, float2, op>, device::fmx<double3, char3, int3, float3, op>, device::fmx<double4, char4, int4, float4, op>  },
                    { device::fmx<double, schar, int, double, op>, device::fmx<double2, char2, int2, double2, op>, device::fmx<double3, char3, int3, double3, op>, device::fmx<double4, char4, int4, double4, op>  },
                },
                {
                    { device::fmx<double, schar, float, uchar, op>, device::fmx<double2, char2, float2, uchar2, op>, device::fmx<double3, char3, float3, uchar3, op>, device::fmx<double4, char4, float4, uchar4, op>  },
                    { device::fmx<double, schar, float, schar, op>, device::fmx<double2, char2, float2, char2, op>, device::fmx<double3, char3, float3, char3, op>, device::fmx<double4, char4, float4, char4, op>  },
                    { device::fmx<double, schar, float, ushort, op>, device::fmx<double2, char2, float2, ushort2, op>, device::fmx<double3, char3, float3, ushort3, op>, device::fmx<double4, char4, float4, ushort4, op>  },
                    { device::fmx<double, schar, float, short, op>, device::fmx<double2, char2, float2, short2, op>, device::fmx<double3, char3, float3, short3, op>, device::fmx<double4, char4, float4, short4, op>  },
                    { device::fmx<double, schar, float, int, op>, device::fmx<double2, char2, float2, int2, op>, device::fmx<double3, char3, float3, int3, op>, device::fmx<double4, char4, float4, int4, op>  },
                    { device::fmx<double, schar, float, float, op>, device::fmx<double2, char2, float2, float2, op>, device::fmx<double3, char3, float3, float3, op>, device::fmx<double4, char4, float4, float4, op>  },
                    { device::fmx<double, schar, float, double, op>, device::fmx<double2, char2, float2, double2, op>, device::fmx<double3, char3, float3, double3, op>, device::fmx<double4, char4, float4, double4, op>  },
                },
                {
                    { device::fmx<double, schar, double, uchar, op>, device::fmx<double2, char2, double2, uchar2, op>, device::fmx<double3, char3, double3, uchar3, op>, device::fmx<double4, char4, double4, uchar4, op>  },
                    { device::fmx<double, schar, double, schar, op>, device::fmx<double2, char2, double2, char2, op>, device::fmx<double3, char3, double3, char3, op>, device::fmx<double4, char4, double4, char4, op>  },
                    { device::fmx<double, schar, double, ushort, op>, device::fmx<double2, char2, double2, ushort2, op>, device::fmx<double3, char3, double3, ushort3, op>, device::fmx<double4, char4, double4, ushort4, op>  },
                    { device::fmx<double, schar, double, short, op>, device::fmx<double2, char2, double2, short2, op>, device::fmx<double3, char3, double3, short3, op>, device::fmx<double4, char4, double4, short4, op>  },
                    { device::fmx<double, schar, double, int, op>, device::fmx<double2, char2, double2, int2, op>, device::fmx<double3, char3, double3, int3, op>, device::fmx<double4, char4, double4, int4, op>  },
                    { device::fmx<double, schar, double, float, op>, device::fmx<double2, char2, double2, float2, op>, device::fmx<double3, char3, double3, float3, op>, device::fmx<double4, char4, double4, float4, op>  },
                    { device::fmx<double, schar, double, double, op>, device::fmx<double2, char2, double2, double2, op>, device::fmx<double3, char3, double3, double3, op>, device::fmx<double4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<double, ushort, uchar, uchar, op>, device::fmx<double2, ushort2, uchar2, uchar2, op>, device::fmx<double3, ushort3, uchar3, uchar3, op>, device::fmx<double4, ushort4, uchar4, uchar4, op>  },
                    { device::fmx<double, ushort, uchar, schar, op>, device::fmx<double2, ushort2, uchar2, char2, op>, device::fmx<double3, ushort3, uchar3, char3, op>, device::fmx<double4, ushort4, uchar4, char4, op>  },
                    { device::fmx<double, ushort, uchar, ushort, op>, device::fmx<double2, ushort2, uchar2, ushort2, op>, device::fmx<double3, ushort3, uchar3, ushort3, op>, device::fmx<double4, ushort4, uchar4, ushort4, op>  },
                    { device::fmx<double, ushort, uchar, short, op>, device::fmx<double2, ushort2, uchar2, short2, op>, device::fmx<double3, ushort3, uchar3, short3, op>, device::fmx<double4, ushort4, uchar4, short4, op>  },
                    { device::fmx<double, ushort, uchar, int, op>, device::fmx<double2, ushort2, uchar2, int2, op>, device::fmx<double3, ushort3, uchar3, int3, op>, device::fmx<double4, ushort4, uchar4, int4, op>  },
                    { device::fmx<double, ushort, uchar, float, op>, device::fmx<double2, ushort2, uchar2, float2, op>, device::fmx<double3, ushort3, uchar3, float3, op>, device::fmx<double4, ushort4, uchar4, float4, op>  },
                    { device::fmx<double, ushort, uchar, double, op>, device::fmx<double2, ushort2, uchar2, double2, op>, device::fmx<double3, ushort3, uchar3, double3, op>, device::fmx<double4, ushort4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<double, ushort, schar, uchar, op>, device::fmx<double2, ushort2, char2, uchar2, op>, device::fmx<double3, ushort3, char3, uchar3, op>, device::fmx<double4, ushort4, char4, uchar4, op>  },
                    { device::fmx<double, ushort, schar, schar, op>, device::fmx<double2, ushort2, char2, char2, op>, device::fmx<double3, ushort3, char3, char3, op>, device::fmx<double4, ushort4, char4, char4, op>  },
                    { device::fmx<double, ushort, schar, ushort, op>, device::fmx<double2, ushort2, char2, ushort2, op>, device::fmx<double3, ushort3, char3, ushort3, op>, device::fmx<double4, ushort4, char4, ushort4, op>  },
                    { device::fmx<double, ushort, schar, short, op>, device::fmx<double2, ushort2, char2, short2, op>, device::fmx<double3, ushort3, char3, short3, op>, device::fmx<double4, ushort4, char4, short4, op>  },
                    { device::fmx<double, ushort, schar, int, op>, device::fmx<double2, ushort2, char2, int2, op>, device::fmx<double3, ushort3, char3, int3, op>, device::fmx<double4, ushort4, char4, int4, op>  },
                    { device::fmx<double, ushort, schar, float, op>, device::fmx<double2, ushort2, char2, float2, op>, device::fmx<double3, ushort3, char3, float3, op>, device::fmx<double4, ushort4, char4, float4, op>  },
                    { device::fmx<double, ushort, schar, double, op>, device::fmx<double2, ushort2, char2, double2, op>, device::fmx<double3, ushort3, char3, double3, op>, device::fmx<double4, ushort4, char4, double4, op>  },
                },
                {
                    { device::fmx<double, ushort, ushort, uchar, op>, device::fmx<double2, ushort2, ushort2, uchar2, op>, device::fmx<double3, ushort3, ushort3, uchar3, op>, device::fmx<double4, ushort4, ushort4, uchar4, op>  },
                    { device::fmx<double, ushort, ushort, schar, op>, device::fmx<double2, ushort2, ushort2, char2, op>, device::fmx<double3, ushort3, ushort3, char3, op>, device::fmx<double4, ushort4, ushort4, char4, op>  },
                    { device::fmx<double, ushort, ushort, ushort, op>, device::fmx<double2, ushort2, ushort2, ushort2, op>, device::fmx<double3, ushort3, ushort3, ushort3, op>, device::fmx<double4, ushort4, ushort4, ushort4, op>  },
                    { device::fmx<double, ushort, ushort, short, op>, device::fmx<double2, ushort2, ushort2, short2, op>, device::fmx<double3, ushort3, ushort3, short3, op>, device::fmx<double4, ushort4, ushort4, short4, op>  },
                    { device::fmx<double, ushort, ushort, int, op>, device::fmx<double2, ushort2, ushort2, int2, op>, device::fmx<double3, ushort3, ushort3, int3, op>, device::fmx<double4, ushort4, ushort4, int4, op>  },
                    { device::fmx<double, ushort, ushort, float, op>, device::fmx<double2, ushort2, ushort2, float2, op>, device::fmx<double3, ushort3, ushort3, float3, op>, device::fmx<double4, ushort4, ushort4, float4, op>  },
                    { device::fmx<double, ushort, ushort, double, op>, device::fmx<double2, ushort2, ushort2, double2, op>, device::fmx<double3, ushort3, ushort3, double3, op>, device::fmx<double4, ushort4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<double, ushort, short, uchar, op>, device::fmx<double2, ushort2, short2, uchar2, op>, device::fmx<double3, ushort3, short3, uchar3, op>, device::fmx<double4, ushort4, short4, uchar4, op>  },
                    { device::fmx<double, ushort, short, schar, op>, device::fmx<double2, ushort2, short2, char2, op>, device::fmx<double3, ushort3, short3, char3, op>, device::fmx<double4, ushort4, short4, char4, op>  },
                    { device::fmx<double, ushort, short, ushort, op>, device::fmx<double2, ushort2, short2, ushort2, op>, device::fmx<double3, ushort3, short3, ushort3, op>, device::fmx<double4, ushort4, short4, ushort4, op>  },
                    { device::fmx<double, ushort, short, short, op>, device::fmx<double2, ushort2, short2, short2, op>, device::fmx<double3, ushort3, short3, short3, op>, device::fmx<double4, ushort4, short4, short4, op>  },
                    { device::fmx<double, ushort, short, int, op>, device::fmx<double2, ushort2, short2, int2, op>, device::fmx<double3, ushort3, short3, int3, op>, device::fmx<double4, ushort4, short4, int4, op>  },
                    { device::fmx<double, ushort, short, float, op>, device::fmx<double2, ushort2, short2, float2, op>, device::fmx<double3, ushort3, short3, float3, op>, device::fmx<double4, ushort4, short4, float4, op>  },
                    { device::fmx<double, ushort, short, double, op>, device::fmx<double2, ushort2, short2, double2, op>, device::fmx<double3, ushort3, short3, double3, op>, device::fmx<double4, ushort4, short4, double4, op>  },
                },
                {
                    { device::fmx<double, ushort, int, uchar, op>, device::fmx<double2, ushort2, int2, uchar2, op>, device::fmx<double3, ushort3, int3, uchar3, op>, device::fmx<double4, ushort4, int4, uchar4, op>  },
                    { device::fmx<double, ushort, int, schar, op>, device::fmx<double2, ushort2, int2, char2, op>, device::fmx<double3, ushort3, int3, char3, op>, device::fmx<double4, ushort4, int4, char4, op>  },
                    { device::fmx<double, ushort, int, ushort, op>, device::fmx<double2, ushort2, int2, ushort2, op>, device::fmx<double3, ushort3, int3, ushort3, op>, device::fmx<double4, ushort4, int4, ushort4, op>  },
                    { device::fmx<double, ushort, int, short, op>, device::fmx<double2, ushort2, int2, short2, op>, device::fmx<double3, ushort3, int3, short3, op>, device::fmx<double4, ushort4, int4, short4, op>  },
                    { device::fmx<double, ushort, int, int, op>, device::fmx<double2, ushort2, int2, int2, op>, device::fmx<double3, ushort3, int3, int3, op>, device::fmx<double4, ushort4, int4, int4, op>  },
                    { device::fmx<double, ushort, int, float, op>, device::fmx<double2, ushort2, int2, float2, op>, device::fmx<double3, ushort3, int3, float3, op>, device::fmx<double4, ushort4, int4, float4, op>  },
                    { device::fmx<double, ushort, int, double, op>, device::fmx<double2, ushort2, int2, double2, op>, device::fmx<double3, ushort3, int3, double3, op>, device::fmx<double4, ushort4, int4, double4, op>  },
                },
                {
                    { device::fmx<double, ushort, float, uchar, op>, device::fmx<double2, ushort2, float2, uchar2, op>, device::fmx<double3, ushort3, float3, uchar3, op>, device::fmx<double4, ushort4, float4, uchar4, op>  },
                    { device::fmx<double, ushort, float, schar, op>, device::fmx<double2, ushort2, float2, char2, op>, device::fmx<double3, ushort3, float3, char3, op>, device::fmx<double4, ushort4, float4, char4, op>  },
                    { device::fmx<double, ushort, float, ushort, op>, device::fmx<double2, ushort2, float2, ushort2, op>, device::fmx<double3, ushort3, float3, ushort3, op>, device::fmx<double4, ushort4, float4, ushort4, op>  },
                    { device::fmx<double, ushort, float, short, op>, device::fmx<double2, ushort2, float2, short2, op>, device::fmx<double3, ushort3, float3, short3, op>, device::fmx<double4, ushort4, float4, short4, op>  },
                    { device::fmx<double, ushort, float, int, op>, device::fmx<double2, ushort2, float2, int2, op>, device::fmx<double3, ushort3, float3, int3, op>, device::fmx<double4, ushort4, float4, int4, op>  },
                    { device::fmx<double, ushort, float, float, op>, device::fmx<double2, ushort2, float2, float2, op>, device::fmx<double3, ushort3, float3, float3, op>, device::fmx<double4, ushort4, float4, float4, op>  },
                    { device::fmx<double, ushort, float, double, op>, device::fmx<double2, ushort2, float2, double2, op>, device::fmx<double3, ushort3, float3, double3, op>, device::fmx<double4, ushort4, float4, double4, op>  },
                },
                {
                    { device::fmx<double, ushort, double, uchar, op>, device::fmx<double2, ushort2, double2, uchar2, op>, device::fmx<double3, ushort3, double3, uchar3, op>, device::fmx<double4, ushort4, double4, uchar4, op>  },
                    { device::fmx<double, ushort, double, schar, op>, device::fmx<double2, ushort2, double2, char2, op>, device::fmx<double3, ushort3, double3, char3, op>, device::fmx<double4, ushort4, double4, char4, op>  },
                    { device::fmx<double, ushort, double, ushort, op>, device::fmx<double2, ushort2, double2, ushort2, op>, device::fmx<double3, ushort3, double3, ushort3, op>, device::fmx<double4, ushort4, double4, ushort4, op>  },
                    { device::fmx<double, ushort, double, short, op>, device::fmx<double2, ushort2, double2, short2, op>, device::fmx<double3, ushort3, double3, short3, op>, device::fmx<double4, ushort4, double4, short4, op>  },
                    { device::fmx<double, ushort, double, int, op>, device::fmx<double2, ushort2, double2, int2, op>, device::fmx<double3, ushort3, double3, int3, op>, device::fmx<double4, ushort4, double4, int4, op>  },
                    { device::fmx<double, ushort, double, float, op>, device::fmx<double2, ushort2, double2, float2, op>, device::fmx<double3, ushort3, double3, float3, op>, device::fmx<double4, ushort4, double4, float4, op>  },
                    { device::fmx<double, ushort, double, double, op>, device::fmx<double2, ushort2, double2, double2, op>, device::fmx<double3, ushort3, double3, double3, op>, device::fmx<double4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<double, short, uchar, uchar, op>, device::fmx<double2, short2, uchar2, uchar2, op>, device::fmx<double3, short3, uchar3, uchar3, op>, device::fmx<double4, short4, uchar4, uchar4, op>  },
                    { device::fmx<double, short, uchar, schar, op>, device::fmx<double2, short2, uchar2, char2, op>, device::fmx<double3, short3, uchar3, char3, op>, device::fmx<double4, short4, uchar4, char4, op>  },
                    { device::fmx<double, short, uchar, ushort, op>, device::fmx<double2, short2, uchar2, ushort2, op>, device::fmx<double3, short3, uchar3, ushort3, op>, device::fmx<double4, short4, uchar4, ushort4, op>  },
                    { device::fmx<double, short, uchar, short, op>, device::fmx<double2, short2, uchar2, short2, op>, device::fmx<double3, short3, uchar3, short3, op>, device::fmx<double4, short4, uchar4, short4, op>  },
                    { device::fmx<double, short, uchar, int, op>, device::fmx<double2, short2, uchar2, int2, op>, device::fmx<double3, short3, uchar3, int3, op>, device::fmx<double4, short4, uchar4, int4, op>  },
                    { device::fmx<double, short, uchar, float, op>, device::fmx<double2, short2, uchar2, float2, op>, device::fmx<double3, short3, uchar3, float3, op>, device::fmx<double4, short4, uchar4, float4, op>  },
                    { device::fmx<double, short, uchar, double, op>, device::fmx<double2, short2, uchar2, double2, op>, device::fmx<double3, short3, uchar3, double3, op>, device::fmx<double4, short4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<double, short, schar, uchar, op>, device::fmx<double2, short2, char2, uchar2, op>, device::fmx<double3, short3, char3, uchar3, op>, device::fmx<double4, short4, char4, uchar4, op>  },
                    { device::fmx<double, short, schar, schar, op>, device::fmx<double2, short2, char2, char2, op>, device::fmx<double3, short3, char3, char3, op>, device::fmx<double4, short4, char4, char4, op>  },
                    { device::fmx<double, short, schar, ushort, op>, device::fmx<double2, short2, char2, ushort2, op>, device::fmx<double3, short3, char3, ushort3, op>, device::fmx<double4, short4, char4, ushort4, op>  },
                    { device::fmx<double, short, schar, short, op>, device::fmx<double2, short2, char2, short2, op>, device::fmx<double3, short3, char3, short3, op>, device::fmx<double4, short4, char4, short4, op>  },
                    { device::fmx<double, short, schar, int, op>, device::fmx<double2, short2, char2, int2, op>, device::fmx<double3, short3, char3, int3, op>, device::fmx<double4, short4, char4, int4, op>  },
                    { device::fmx<double, short, schar, float, op>, device::fmx<double2, short2, char2, float2, op>, device::fmx<double3, short3, char3, float3, op>, device::fmx<double4, short4, char4, float4, op>  },
                    { device::fmx<double, short, schar, double, op>, device::fmx<double2, short2, char2, double2, op>, device::fmx<double3, short3, char3, double3, op>, device::fmx<double4, short4, char4, double4, op>  },
                },
                {
                    { device::fmx<double, short, ushort, uchar, op>, device::fmx<double2, short2, ushort2, uchar2, op>, device::fmx<double3, short3, ushort3, uchar3, op>, device::fmx<double4, short4, ushort4, uchar4, op>  },
                    { device::fmx<double, short, ushort, schar, op>, device::fmx<double2, short2, ushort2, char2, op>, device::fmx<double3, short3, ushort3, char3, op>, device::fmx<double4, short4, ushort4, char4, op>  },
                    { device::fmx<double, short, ushort, ushort, op>, device::fmx<double2, short2, ushort2, ushort2, op>, device::fmx<double3, short3, ushort3, ushort3, op>, device::fmx<double4, short4, ushort4, ushort4, op>  },
                    { device::fmx<double, short, ushort, short, op>, device::fmx<double2, short2, ushort2, short2, op>, device::fmx<double3, short3, ushort3, short3, op>, device::fmx<double4, short4, ushort4, short4, op>  },
                    { device::fmx<double, short, ushort, int, op>, device::fmx<double2, short2, ushort2, int2, op>, device::fmx<double3, short3, ushort3, int3, op>, device::fmx<double4, short4, ushort4, int4, op>  },
                    { device::fmx<double, short, ushort, float, op>, device::fmx<double2, short2, ushort2, float2, op>, device::fmx<double3, short3, ushort3, float3, op>, device::fmx<double4, short4, ushort4, float4, op>  },
                    { device::fmx<double, short, ushort, double, op>, device::fmx<double2, short2, ushort2, double2, op>, device::fmx<double3, short3, ushort3, double3, op>, device::fmx<double4, short4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<double, short, short, uchar, op>, device::fmx<double2, short2, short2, uchar2, op>, device::fmx<double3, short3, short3, uchar3, op>, device::fmx<double4, short4, short4, uchar4, op>  },
                    { device::fmx<double, short, short, schar, op>, device::fmx<double2, short2, short2, char2, op>, device::fmx<double3, short3, short3, char3, op>, device::fmx<double4, short4, short4, char4, op>  },
                    { device::fmx<double, short, short, ushort, op>, device::fmx<double2, short2, short2, ushort2, op>, device::fmx<double3, short3, short3, ushort3, op>, device::fmx<double4, short4, short4, ushort4, op>  },
                    { device::fmx<double, short, short, short, op>, device::fmx<double2, short2, short2, short2, op>, device::fmx<double3, short3, short3, short3, op>, device::fmx<double4, short4, short4, short4, op>  },
                    { device::fmx<double, short, short, int, op>, device::fmx<double2, short2, short2, int2, op>, device::fmx<double3, short3, short3, int3, op>, device::fmx<double4, short4, short4, int4, op>  },
                    { device::fmx<double, short, short, float, op>, device::fmx<double2, short2, short2, float2, op>, device::fmx<double3, short3, short3, float3, op>, device::fmx<double4, short4, short4, float4, op>  },
                    { device::fmx<double, short, short, double, op>, device::fmx<double2, short2, short2, double2, op>, device::fmx<double3, short3, short3, double3, op>, device::fmx<double4, short4, short4, double4, op>  },
                },
                {
                    { device::fmx<double, short, int, uchar, op>, device::fmx<double2, short2, int2, uchar2, op>, device::fmx<double3, short3, int3, uchar3, op>, device::fmx<double4, short4, int4, uchar4, op>  },
                    { device::fmx<double, short, int, schar, op>, device::fmx<double2, short2, int2, char2, op>, device::fmx<double3, short3, int3, char3, op>, device::fmx<double4, short4, int4, char4, op>  },
                    { device::fmx<double, short, int, ushort, op>, device::fmx<double2, short2, int2, ushort2, op>, device::fmx<double3, short3, int3, ushort3, op>, device::fmx<double4, short4, int4, ushort4, op>  },
                    { device::fmx<double, short, int, short, op>, device::fmx<double2, short2, int2, short2, op>, device::fmx<double3, short3, int3, short3, op>, device::fmx<double4, short4, int4, short4, op>  },
                    { device::fmx<double, short, int, int, op>, device::fmx<double2, short2, int2, int2, op>, device::fmx<double3, short3, int3, int3, op>, device::fmx<double4, short4, int4, int4, op>  },
                    { device::fmx<double, short, int, float, op>, device::fmx<double2, short2, int2, float2, op>, device::fmx<double3, short3, int3, float3, op>, device::fmx<double4, short4, int4, float4, op>  },
                    { device::fmx<double, short, int, double, op>, device::fmx<double2, short2, int2, double2, op>, device::fmx<double3, short3, int3, double3, op>, device::fmx<double4, short4, int4, double4, op>  },
                },
                {
                    { device::fmx<double, short, float, uchar, op>, device::fmx<double2, short2, float2, uchar2, op>, device::fmx<double3, short3, float3, uchar3, op>, device::fmx<double4, short4, float4, uchar4, op>  },
                    { device::fmx<double, short, float, schar, op>, device::fmx<double2, short2, float2, char2, op>, device::fmx<double3, short3, float3, char3, op>, device::fmx<double4, short4, float4, char4, op>  },
                    { device::fmx<double, short, float, ushort, op>, device::fmx<double2, short2, float2, ushort2, op>, device::fmx<double3, short3, float3, ushort3, op>, device::fmx<double4, short4, float4, ushort4, op>  },
                    { device::fmx<double, short, float, short, op>, device::fmx<double2, short2, float2, short2, op>, device::fmx<double3, short3, float3, short3, op>, device::fmx<double4, short4, float4, short4, op>  },
                    { device::fmx<double, short, float, int, op>, device::fmx<double2, short2, float2, int2, op>, device::fmx<double3, short3, float3, int3, op>, device::fmx<double4, short4, float4, int4, op>  },
                    { device::fmx<double, short, float, float, op>, device::fmx<double2, short2, float2, float2, op>, device::fmx<double3, short3, float3, float3, op>, device::fmx<double4, short4, float4, float4, op>  },
                    { device::fmx<double, short, float, double, op>, device::fmx<double2, short2, float2, double2, op>, device::fmx<double3, short3, float3, double3, op>, device::fmx<double4, short4, float4, double4, op>  },
                },
                {
                    { device::fmx<double, short, double, uchar, op>, device::fmx<double2, short2, double2, uchar2, op>, device::fmx<double3, short3, double3, uchar3, op>, device::fmx<double4, short4, double4, uchar4, op>  },
                    { device::fmx<double, short, double, schar, op>, device::fmx<double2, short2, double2, char2, op>, device::fmx<double3, short3, double3, char3, op>, device::fmx<double4, short4, double4, char4, op>  },
                    { device::fmx<double, short, double, ushort, op>, device::fmx<double2, short2, double2, ushort2, op>, device::fmx<double3, short3, double3, ushort3, op>, device::fmx<double4, short4, double4, ushort4, op>  },
                    { device::fmx<double, short, double, short, op>, device::fmx<double2, short2, double2, short2, op>, device::fmx<double3, short3, double3, short3, op>, device::fmx<double4, short4, double4, short4, op>  },
                    { device::fmx<double, short, double, int, op>, device::fmx<double2, short2, double2, int2, op>, device::fmx<double3, short3, double3, int3, op>, device::fmx<double4, short4, double4, int4, op>  },
                    { device::fmx<double, short, double, float, op>, device::fmx<double2, short2, double2, float2, op>, device::fmx<double3, short3, double3, float3, op>, device::fmx<double4, short4, double4, float4, op>  },
                    { device::fmx<double, short, double, double, op>, device::fmx<double2, short2, double2, double2, op>, device::fmx<double3, short3, double3, double3, op>, device::fmx<double4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<double, int, uchar, uchar, op>, device::fmx<double2, int2, uchar2, uchar2, op>, device::fmx<double3, int3, uchar3, uchar3, op>, device::fmx<double4, int4, uchar4, uchar4, op>  },
                    { device::fmx<double, int, uchar, schar, op>, device::fmx<double2, int2, uchar2, char2, op>, device::fmx<double3, int3, uchar3, char3, op>, device::fmx<double4, int4, uchar4, char4, op>  },
                    { device::fmx<double, int, uchar, ushort, op>, device::fmx<double2, int2, uchar2, ushort2, op>, device::fmx<double3, int3, uchar3, ushort3, op>, device::fmx<double4, int4, uchar4, ushort4, op>  },
                    { device::fmx<double, int, uchar, short, op>, device::fmx<double2, int2, uchar2, short2, op>, device::fmx<double3, int3, uchar3, short3, op>, device::fmx<double4, int4, uchar4, short4, op>  },
                    { device::fmx<double, int, uchar, int, op>, device::fmx<double2, int2, uchar2, int2, op>, device::fmx<double3, int3, uchar3, int3, op>, device::fmx<double4, int4, uchar4, int4, op>  },
                    { device::fmx<double, int, uchar, float, op>, device::fmx<double2, int2, uchar2, float2, op>, device::fmx<double3, int3, uchar3, float3, op>, device::fmx<double4, int4, uchar4, float4, op>  },
                    { device::fmx<double, int, uchar, double, op>, device::fmx<double2, int2, uchar2, double2, op>, device::fmx<double3, int3, uchar3, double3, op>, device::fmx<double4, int4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<double, int, schar, uchar, op>, device::fmx<double2, int2, char2, uchar2, op>, device::fmx<double3, int3, char3, uchar3, op>, device::fmx<double4, int4, char4, uchar4, op>  },
                    { device::fmx<double, int, schar, schar, op>, device::fmx<double2, int2, char2, char2, op>, device::fmx<double3, int3, char3, char3, op>, device::fmx<double4, int4, char4, char4, op>  },
                    { device::fmx<double, int, schar, ushort, op>, device::fmx<double2, int2, char2, ushort2, op>, device::fmx<double3, int3, char3, ushort3, op>, device::fmx<double4, int4, char4, ushort4, op>  },
                    { device::fmx<double, int, schar, short, op>, device::fmx<double2, int2, char2, short2, op>, device::fmx<double3, int3, char3, short3, op>, device::fmx<double4, int4, char4, short4, op>  },
                    { device::fmx<double, int, schar, int, op>, device::fmx<double2, int2, char2, int2, op>, device::fmx<double3, int3, char3, int3, op>, device::fmx<double4, int4, char4, int4, op>  },
                    { device::fmx<double, int, schar, float, op>, device::fmx<double2, int2, char2, float2, op>, device::fmx<double3, int3, char3, float3, op>, device::fmx<double4, int4, char4, float4, op>  },
                    { device::fmx<double, int, schar, double, op>, device::fmx<double2, int2, char2, double2, op>, device::fmx<double3, int3, char3, double3, op>, device::fmx<double4, int4, char4, double4, op>  },
                },
                {
                    { device::fmx<double, int, ushort, uchar, op>, device::fmx<double2, int2, ushort2, uchar2, op>, device::fmx<double3, int3, ushort3, uchar3, op>, device::fmx<double4, int4, ushort4, uchar4, op>  },
                    { device::fmx<double, int, ushort, schar, op>, device::fmx<double2, int2, ushort2, char2, op>, device::fmx<double3, int3, ushort3, char3, op>, device::fmx<double4, int4, ushort4, char4, op>  },
                    { device::fmx<double, int, ushort, ushort, op>, device::fmx<double2, int2, ushort2, ushort2, op>, device::fmx<double3, int3, ushort3, ushort3, op>, device::fmx<double4, int4, ushort4, ushort4, op>  },
                    { device::fmx<double, int, ushort, short, op>, device::fmx<double2, int2, ushort2, short2, op>, device::fmx<double3, int3, ushort3, short3, op>, device::fmx<double4, int4, ushort4, short4, op>  },
                    { device::fmx<double, int, ushort, int, op>, device::fmx<double2, int2, ushort2, int2, op>, device::fmx<double3, int3, ushort3, int3, op>, device::fmx<double4, int4, ushort4, int4, op>  },
                    { device::fmx<double, int, ushort, float, op>, device::fmx<double2, int2, ushort2, float2, op>, device::fmx<double3, int3, ushort3, float3, op>, device::fmx<double4, int4, ushort4, float4, op>  },
                    { device::fmx<double, int, ushort, double, op>, device::fmx<double2, int2, ushort2, double2, op>, device::fmx<double3, int3, ushort3, double3, op>, device::fmx<double4, int4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<double, int, short, uchar, op>, device::fmx<double2, int2, short2, uchar2, op>, device::fmx<double3, int3, short3, uchar3, op>, device::fmx<double4, int4, short4, uchar4, op>  },
                    { device::fmx<double, int, short, schar, op>, device::fmx<double2, int2, short2, char2, op>, device::fmx<double3, int3, short3, char3, op>, device::fmx<double4, int4, short4, char4, op>  },
                    { device::fmx<double, int, short, ushort, op>, device::fmx<double2, int2, short2, ushort2, op>, device::fmx<double3, int3, short3, ushort3, op>, device::fmx<double4, int4, short4, ushort4, op>  },
                    { device::fmx<double, int, short, short, op>, device::fmx<double2, int2, short2, short2, op>, device::fmx<double3, int3, short3, short3, op>, device::fmx<double4, int4, short4, short4, op>  },
                    { device::fmx<double, int, short, int, op>, device::fmx<double2, int2, short2, int2, op>, device::fmx<double3, int3, short3, int3, op>, device::fmx<double4, int4, short4, int4, op>  },
                    { device::fmx<double, int, short, float, op>, device::fmx<double2, int2, short2, float2, op>, device::fmx<double3, int3, short3, float3, op>, device::fmx<double4, int4, short4, float4, op>  },
                    { device::fmx<double, int, short, double, op>, device::fmx<double2, int2, short2, double2, op>, device::fmx<double3, int3, short3, double3, op>, device::fmx<double4, int4, short4, double4, op>  },
                },
                {
                    { device::fmx<double, int, int, uchar, op>, device::fmx<double2, int2, int2, uchar2, op>, device::fmx<double3, int3, int3, uchar3, op>, device::fmx<double4, int4, int4, uchar4, op>  },
                    { device::fmx<double, int, int, schar, op>, device::fmx<double2, int2, int2, char2, op>, device::fmx<double3, int3, int3, char3, op>, device::fmx<double4, int4, int4, char4, op>  },
                    { device::fmx<double, int, int, ushort, op>, device::fmx<double2, int2, int2, ushort2, op>, device::fmx<double3, int3, int3, ushort3, op>, device::fmx<double4, int4, int4, ushort4, op>  },
                    { device::fmx<double, int, int, short, op>, device::fmx<double2, int2, int2, short2, op>, device::fmx<double3, int3, int3, short3, op>, device::fmx<double4, int4, int4, short4, op>  },
                    { device::fmx<double, int, int, int, op>, device::fmx<double2, int2, int2, int2, op>, device::fmx<double3, int3, int3, int3, op>, device::fmx<double4, int4, int4, int4, op>  },
                    { device::fmx<double, int, int, float, op>, device::fmx<double2, int2, int2, float2, op>, device::fmx<double3, int3, int3, float3, op>, device::fmx<double4, int4, int4, float4, op>  },
                    { device::fmx<double, int, int, double, op>, device::fmx<double2, int2, int2, double2, op>, device::fmx<double3, int3, int3, double3, op>, device::fmx<double4, int4, int4, double4, op>  },
                },
                {
                    { device::fmx<double, int, float, uchar, op>, device::fmx<double2, int2, float2, uchar2, op>, device::fmx<double3, int3, float3, uchar3, op>, device::fmx<double4, int4, float4, uchar4, op>  },
                    { device::fmx<double, int, float, schar, op>, device::fmx<double2, int2, float2, char2, op>, device::fmx<double3, int3, float3, char3, op>, device::fmx<double4, int4, float4, char4, op>  },
                    { device::fmx<double, int, float, ushort, op>, device::fmx<double2, int2, float2, ushort2, op>, device::fmx<double3, int3, float3, ushort3, op>, device::fmx<double4, int4, float4, ushort4, op>  },
                    { device::fmx<double, int, float, short, op>, device::fmx<double2, int2, float2, short2, op>, device::fmx<double3, int3, float3, short3, op>, device::fmx<double4, int4, float4, short4, op>  },
                    { device::fmx<double, int, float, int, op>, device::fmx<double2, int2, float2, int2, op>, device::fmx<double3, int3, float3, int3, op>, device::fmx<double4, int4, float4, int4, op>  },
                    { device::fmx<double, int, float, float, op>, device::fmx<double2, int2, float2, float2, op>, device::fmx<double3, int3, float3, float3, op>, device::fmx<double4, int4, float4, float4, op>  },
                    { device::fmx<double, int, float, double, op>, device::fmx<double2, int2, float2, double2, op>, device::fmx<double3, int3, float3, double3, op>, device::fmx<double4, int4, float4, double4, op>  },
                },
                {
                    { device::fmx<double, int, double, uchar, op>, device::fmx<double2, int2, double2, uchar2, op>, device::fmx<double3, int3, double3, uchar3, op>, device::fmx<double4, int4, double4, uchar4, op>  },
                    { device::fmx<double, int, double, schar, op>, device::fmx<double2, int2, double2, char2, op>, device::fmx<double3, int3, double3, char3, op>, device::fmx<double4, int4, double4, char4, op>  },
                    { device::fmx<double, int, double, ushort, op>, device::fmx<double2, int2, double2, ushort2, op>, device::fmx<double3, int3, double3, ushort3, op>, device::fmx<double4, int4, double4, ushort4, op>  },
                    { device::fmx<double, int, double, short, op>, device::fmx<double2, int2, double2, short2, op>, device::fmx<double3, int3, double3, short3, op>, device::fmx<double4, int4, double4, short4, op>  },
                    { device::fmx<double, int, double, int, op>, device::fmx<double2, int2, double2, int2, op>, device::fmx<double3, int3, double3, int3, op>, device::fmx<double4, int4, double4, int4, op>  },
                    { device::fmx<double, int, double, float, op>, device::fmx<double2, int2, double2, float2, op>, device::fmx<double3, int3, double3, float3, op>, device::fmx<double4, int4, double4, float4, op>  },
                    { device::fmx<double, int, double, double, op>, device::fmx<double2, int2, double2, double2, op>, device::fmx<double3, int3, double3, double3, op>, device::fmx<double4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<double, float, uchar, uchar, op>, device::fmx<double2, float2, uchar2, uchar2, op>, device::fmx<double3, float3, uchar3, uchar3, op>, device::fmx<double4, float4, uchar4, uchar4, op>  },
                    { device::fmx<double, float, uchar, schar, op>, device::fmx<double2, float2, uchar2, char2, op>, device::fmx<double3, float3, uchar3, char3, op>, device::fmx<double4, float4, uchar4, char4, op>  },
                    { device::fmx<double, float, uchar, ushort, op>, device::fmx<double2, float2, uchar2, ushort2, op>, device::fmx<double3, float3, uchar3, ushort3, op>, device::fmx<double4, float4, uchar4, ushort4, op>  },
                    { device::fmx<double, float, uchar, short, op>, device::fmx<double2, float2, uchar2, short2, op>, device::fmx<double3, float3, uchar3, short3, op>, device::fmx<double4, float4, uchar4, short4, op>  },
                    { device::fmx<double, float, uchar, int, op>, device::fmx<double2, float2, uchar2, int2, op>, device::fmx<double3, float3, uchar3, int3, op>, device::fmx<double4, float4, uchar4, int4, op>  },
                    { device::fmx<double, float, uchar, float, op>, device::fmx<double2, float2, uchar2, float2, op>, device::fmx<double3, float3, uchar3, float3, op>, device::fmx<double4, float4, uchar4, float4, op>  },
                    { device::fmx<double, float, uchar, double, op>, device::fmx<double2, float2, uchar2, double2, op>, device::fmx<double3, float3, uchar3, double3, op>, device::fmx<double4, float4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<double, float, schar, uchar, op>, device::fmx<double2, float2, char2, uchar2, op>, device::fmx<double3, float3, char3, uchar3, op>, device::fmx<double4, float4, char4, uchar4, op>  },
                    { device::fmx<double, float, schar, schar, op>, device::fmx<double2, float2, char2, char2, op>, device::fmx<double3, float3, char3, char3, op>, device::fmx<double4, float4, char4, char4, op>  },
                    { device::fmx<double, float, schar, ushort, op>, device::fmx<double2, float2, char2, ushort2, op>, device::fmx<double3, float3, char3, ushort3, op>, device::fmx<double4, float4, char4, ushort4, op>  },
                    { device::fmx<double, float, schar, short, op>, device::fmx<double2, float2, char2, short2, op>, device::fmx<double3, float3, char3, short3, op>, device::fmx<double4, float4, char4, short4, op>  },
                    { device::fmx<double, float, schar, int, op>, device::fmx<double2, float2, char2, int2, op>, device::fmx<double3, float3, char3, int3, op>, device::fmx<double4, float4, char4, int4, op>  },
                    { device::fmx<double, float, schar, float, op>, device::fmx<double2, float2, char2, float2, op>, device::fmx<double3, float3, char3, float3, op>, device::fmx<double4, float4, char4, float4, op>  },
                    { device::fmx<double, float, schar, double, op>, device::fmx<double2, float2, char2, double2, op>, device::fmx<double3, float3, char3, double3, op>, device::fmx<double4, float4, char4, double4, op>  },
                },
                {
                    { device::fmx<double, float, ushort, uchar, op>, device::fmx<double2, float2, ushort2, uchar2, op>, device::fmx<double3, float3, ushort3, uchar3, op>, device::fmx<double4, float4, ushort4, uchar4, op>  },
                    { device::fmx<double, float, ushort, schar, op>, device::fmx<double2, float2, ushort2, char2, op>, device::fmx<double3, float3, ushort3, char3, op>, device::fmx<double4, float4, ushort4, char4, op>  },
                    { device::fmx<double, float, ushort, ushort, op>, device::fmx<double2, float2, ushort2, ushort2, op>, device::fmx<double3, float3, ushort3, ushort3, op>, device::fmx<double4, float4, ushort4, ushort4, op>  },
                    { device::fmx<double, float, ushort, short, op>, device::fmx<double2, float2, ushort2, short2, op>, device::fmx<double3, float3, ushort3, short3, op>, device::fmx<double4, float4, ushort4, short4, op>  },
                    { device::fmx<double, float, ushort, int, op>, device::fmx<double2, float2, ushort2, int2, op>, device::fmx<double3, float3, ushort3, int3, op>, device::fmx<double4, float4, ushort4, int4, op>  },
                    { device::fmx<double, float, ushort, float, op>, device::fmx<double2, float2, ushort2, float2, op>, device::fmx<double3, float3, ushort3, float3, op>, device::fmx<double4, float4, ushort4, float4, op>  },
                    { device::fmx<double, float, ushort, double, op>, device::fmx<double2, float2, ushort2, double2, op>, device::fmx<double3, float3, ushort3, double3, op>, device::fmx<double4, float4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<double, float, short, uchar, op>, device::fmx<double2, float2, short2, uchar2, op>, device::fmx<double3, float3, short3, uchar3, op>, device::fmx<double4, float4, short4, uchar4, op>  },
                    { device::fmx<double, float, short, schar, op>, device::fmx<double2, float2, short2, char2, op>, device::fmx<double3, float3, short3, char3, op>, device::fmx<double4, float4, short4, char4, op>  },
                    { device::fmx<double, float, short, ushort, op>, device::fmx<double2, float2, short2, ushort2, op>, device::fmx<double3, float3, short3, ushort3, op>, device::fmx<double4, float4, short4, ushort4, op>  },
                    { device::fmx<double, float, short, short, op>, device::fmx<double2, float2, short2, short2, op>, device::fmx<double3, float3, short3, short3, op>, device::fmx<double4, float4, short4, short4, op>  },
                    { device::fmx<double, float, short, int, op>, device::fmx<double2, float2, short2, int2, op>, device::fmx<double3, float3, short3, int3, op>, device::fmx<double4, float4, short4, int4, op>  },
                    { device::fmx<double, float, short, float, op>, device::fmx<double2, float2, short2, float2, op>, device::fmx<double3, float3, short3, float3, op>, device::fmx<double4, float4, short4, float4, op>  },
                    { device::fmx<double, float, short, double, op>, device::fmx<double2, float2, short2, double2, op>, device::fmx<double3, float3, short3, double3, op>, device::fmx<double4, float4, short4, double4, op>  },
                },
                {
                    { device::fmx<double, float, int, uchar, op>, device::fmx<double2, float2, int2, uchar2, op>, device::fmx<double3, float3, int3, uchar3, op>, device::fmx<double4, float4, int4, uchar4, op>  },
                    { device::fmx<double, float, int, schar, op>, device::fmx<double2, float2, int2, char2, op>, device::fmx<double3, float3, int3, char3, op>, device::fmx<double4, float4, int4, char4, op>  },
                    { device::fmx<double, float, int, ushort, op>, device::fmx<double2, float2, int2, ushort2, op>, device::fmx<double3, float3, int3, ushort3, op>, device::fmx<double4, float4, int4, ushort4, op>  },
                    { device::fmx<double, float, int, short, op>, device::fmx<double2, float2, int2, short2, op>, device::fmx<double3, float3, int3, short3, op>, device::fmx<double4, float4, int4, short4, op>  },
                    { device::fmx<double, float, int, int, op>, device::fmx<double2, float2, int2, int2, op>, device::fmx<double3, float3, int3, int3, op>, device::fmx<double4, float4, int4, int4, op>  },
                    { device::fmx<double, float, int, float, op>, device::fmx<double2, float2, int2, float2, op>, device::fmx<double3, float3, int3, float3, op>, device::fmx<double4, float4, int4, float4, op>  },
                    { device::fmx<double, float, int, double, op>, device::fmx<double2, float2, int2, double2, op>, device::fmx<double3, float3, int3, double3, op>, device::fmx<double4, float4, int4, double4, op>  },
                },
                {
                    { device::fmx<double, float, float, uchar, op>, device::fmx<double2, float2, float2, uchar2, op>, device::fmx<double3, float3, float3, uchar3, op>, device::fmx<double4, float4, float4, uchar4, op>  },
                    { device::fmx<double, float, float, schar, op>, device::fmx<double2, float2, float2, char2, op>, device::fmx<double3, float3, float3, char3, op>, device::fmx<double4, float4, float4, char4, op>  },
                    { device::fmx<double, float, float, ushort, op>, device::fmx<double2, float2, float2, ushort2, op>, device::fmx<double3, float3, float3, ushort3, op>, device::fmx<double4, float4, float4, ushort4, op>  },
                    { device::fmx<double, float, float, short, op>, device::fmx<double2, float2, float2, short2, op>, device::fmx<double3, float3, float3, short3, op>, device::fmx<double4, float4, float4, short4, op>  },
                    { device::fmx<double, float, float, int, op>, device::fmx<double2, float2, float2, int2, op>, device::fmx<double3, float3, float3, int3, op>, device::fmx<double4, float4, float4, int4, op>  },
                    { device::fmx<double, float, float, float, op>, device::fmx<double2, float2, float2, float2, op>, device::fmx<double3, float3, float3, float3, op>, device::fmx<double4, float4, float4, float4, op>  },
                    { device::fmx<double, float, float, double, op>, device::fmx<double2, float2, float2, double2, op>, device::fmx<double3, float3, float3, double3, op>, device::fmx<double4, float4, float4, double4, op>  },
                },
                {
                    { device::fmx<double, float, double, uchar, op>, device::fmx<double2, float2, double2, uchar2, op>, device::fmx<double3, float3, double3, uchar3, op>, device::fmx<double4, float4, double4, uchar4, op>  },
                    { device::fmx<double, float, double, schar, op>, device::fmx<double2, float2, double2, char2, op>, device::fmx<double3, float3, double3, char3, op>, device::fmx<double4, float4, double4, char4, op>  },
                    { device::fmx<double, float, double, ushort, op>, device::fmx<double2, float2, double2, ushort2, op>, device::fmx<double3, float3, double3, ushort3, op>, device::fmx<double4, float4, double4, ushort4, op>  },
                    { device::fmx<double, float, double, short, op>, device::fmx<double2, float2, double2, short2, op>, device::fmx<double3, float3, double3, short3, op>, device::fmx<double4, float4, double4, short4, op>  },
                    { device::fmx<double, float, double, int, op>, device::fmx<double2, float2, double2, int2, op>, device::fmx<double3, float3, double3, int3, op>, device::fmx<double4, float4, double4, int4, op>  },
                    { device::fmx<double, float, double, float, op>, device::fmx<double2, float2, double2, float2, op>, device::fmx<double3, float3, double3, float3, op>, device::fmx<double4, float4, double4, float4, op>  },
                    { device::fmx<double, float, double, double, op>, device::fmx<double2, float2, double2, double2, op>, device::fmx<double3, float3, double3, double3, op>, device::fmx<double4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { device::fmx<double, double, uchar, uchar, op>, device::fmx<double2, double2, uchar2, uchar2, op>, device::fmx<double3, double3, uchar3, uchar3, op>, device::fmx<double4, double4, uchar4, uchar4, op>  },
                    { device::fmx<double, double, uchar, schar, op>, device::fmx<double2, double2, uchar2, char2, op>, device::fmx<double3, double3, uchar3, char3, op>, device::fmx<double4, double4, uchar4, char4, op>  },
                    { device::fmx<double, double, uchar, ushort, op>, device::fmx<double2, double2, uchar2, ushort2, op>, device::fmx<double3, double3, uchar3, ushort3, op>, device::fmx<double4, double4, uchar4, ushort4, op>  },
                    { device::fmx<double, double, uchar, short, op>, device::fmx<double2, double2, uchar2, short2, op>, device::fmx<double3, double3, uchar3, short3, op>, device::fmx<double4, double4, uchar4, short4, op>  },
                    { device::fmx<double, double, uchar, int, op>, device::fmx<double2, double2, uchar2, int2, op>, device::fmx<double3, double3, uchar3, int3, op>, device::fmx<double4, double4, uchar4, int4, op>  },
                    { device::fmx<double, double, uchar, float, op>, device::fmx<double2, double2, uchar2, float2, op>, device::fmx<double3, double3, uchar3, float3, op>, device::fmx<double4, double4, uchar4, float4, op>  },
                    { device::fmx<double, double, uchar, double, op>, device::fmx<double2, double2, uchar2, double2, op>, device::fmx<double3, double3, uchar3, double3, op>, device::fmx<double4, double4, uchar4, double4, op>  },
                },
                {
                    { device::fmx<double, double, schar, uchar, op>, device::fmx<double2, double2, char2, uchar2, op>, device::fmx<double3, double3, char3, uchar3, op>, device::fmx<double4, double4, char4, uchar4, op>  },
                    { device::fmx<double, double, schar, schar, op>, device::fmx<double2, double2, char2, char2, op>, device::fmx<double3, double3, char3, char3, op>, device::fmx<double4, double4, char4, char4, op>  },
                    { device::fmx<double, double, schar, ushort, op>, device::fmx<double2, double2, char2, ushort2, op>, device::fmx<double3, double3, char3, ushort3, op>, device::fmx<double4, double4, char4, ushort4, op>  },
                    { device::fmx<double, double, schar, short, op>, device::fmx<double2, double2, char2, short2, op>, device::fmx<double3, double3, char3, short3, op>, device::fmx<double4, double4, char4, short4, op>  },
                    { device::fmx<double, double, schar, int, op>, device::fmx<double2, double2, char2, int2, op>, device::fmx<double3, double3, char3, int3, op>, device::fmx<double4, double4, char4, int4, op>  },
                    { device::fmx<double, double, schar, float, op>, device::fmx<double2, double2, char2, float2, op>, device::fmx<double3, double3, char3, float3, op>, device::fmx<double4, double4, char4, float4, op>  },
                    { device::fmx<double, double, schar, double, op>, device::fmx<double2, double2, char2, double2, op>, device::fmx<double3, double3, char3, double3, op>, device::fmx<double4, double4, char4, double4, op>  },
                },
                {
                    { device::fmx<double, double, ushort, uchar, op>, device::fmx<double2, double2, ushort2, uchar2, op>, device::fmx<double3, double3, ushort3, uchar3, op>, device::fmx<double4, double4, ushort4, uchar4, op>  },
                    { device::fmx<double, double, ushort, schar, op>, device::fmx<double2, double2, ushort2, char2, op>, device::fmx<double3, double3, ushort3, char3, op>, device::fmx<double4, double4, ushort4, char4, op>  },
                    { device::fmx<double, double, ushort, ushort, op>, device::fmx<double2, double2, ushort2, ushort2, op>, device::fmx<double3, double3, ushort3, ushort3, op>, device::fmx<double4, double4, ushort4, ushort4, op>  },
                    { device::fmx<double, double, ushort, short, op>, device::fmx<double2, double2, ushort2, short2, op>, device::fmx<double3, double3, ushort3, short3, op>, device::fmx<double4, double4, ushort4, short4, op>  },
                    { device::fmx<double, double, ushort, int, op>, device::fmx<double2, double2, ushort2, int2, op>, device::fmx<double3, double3, ushort3, int3, op>, device::fmx<double4, double4, ushort4, int4, op>  },
                    { device::fmx<double, double, ushort, float, op>, device::fmx<double2, double2, ushort2, float2, op>, device::fmx<double3, double3, ushort3, float3, op>, device::fmx<double4, double4, ushort4, float4, op>  },
                    { device::fmx<double, double, ushort, double, op>, device::fmx<double2, double2, ushort2, double2, op>, device::fmx<double3, double3, ushort3, double3, op>, device::fmx<double4, double4, ushort4, double4, op>  },
                },
                {
                    { device::fmx<double, double, short, uchar, op>, device::fmx<double2, double2, short2, uchar2, op>, device::fmx<double3, double3, short3, uchar3, op>, device::fmx<double4, double4, short4, uchar4, op>  },
                    { device::fmx<double, double, short, schar, op>, device::fmx<double2, double2, short2, char2, op>, device::fmx<double3, double3, short3, char3, op>, device::fmx<double4, double4, short4, char4, op>  },
                    { device::fmx<double, double, short, ushort, op>, device::fmx<double2, double2, short2, ushort2, op>, device::fmx<double3, double3, short3, ushort3, op>, device::fmx<double4, double4, short4, ushort4, op>  },
                    { device::fmx<double, double, short, short, op>, device::fmx<double2, double2, short2, short2, op>, device::fmx<double3, double3, short3, short3, op>, device::fmx<double4, double4, short4, short4, op>  },
                    { device::fmx<double, double, short, int, op>, device::fmx<double2, double2, short2, int2, op>, device::fmx<double3, double3, short3, int3, op>, device::fmx<double4, double4, short4, int4, op>  },
                    { device::fmx<double, double, short, float, op>, device::fmx<double2, double2, short2, float2, op>, device::fmx<double3, double3, short3, float3, op>, device::fmx<double4, double4, short4, float4, op>  },
                    { device::fmx<double, double, short, double, op>, device::fmx<double2, double2, short2, double2, op>, device::fmx<double3, double3, short3, double3, op>, device::fmx<double4, double4, short4, double4, op>  },
                },
                {
                    { device::fmx<double, double, int, uchar, op>, device::fmx<double2, double2, int2, uchar2, op>, device::fmx<double3, double3, int3, uchar3, op>, device::fmx<double4, double4, int4, uchar4, op>  },
                    { device::fmx<double, double, int, schar, op>, device::fmx<double2, double2, int2, char2, op>, device::fmx<double3, double3, int3, char3, op>, device::fmx<double4, double4, int4, char4, op>  },
                    { device::fmx<double, double, int, ushort, op>, device::fmx<double2, double2, int2, ushort2, op>, device::fmx<double3, double3, int3, ushort3, op>, device::fmx<double4, double4, int4, ushort4, op>  },
                    { device::fmx<double, double, int, short, op>, device::fmx<double2, double2, int2, short2, op>, device::fmx<double3, double3, int3, short3, op>, device::fmx<double4, double4, int4, short4, op>  },
                    { device::fmx<double, double, int, int, op>, device::fmx<double2, double2, int2, int2, op>, device::fmx<double3, double3, int3, int3, op>, device::fmx<double4, double4, int4, int4, op>  },
                    { device::fmx<double, double, int, float, op>, device::fmx<double2, double2, int2, float2, op>, device::fmx<double3, double3, int3, float3, op>, device::fmx<double4, double4, int4, float4, op>  },
                    { device::fmx<double, double, int, double, op>, device::fmx<double2, double2, int2, double2, op>, device::fmx<double3, double3, int3, double3, op>, device::fmx<double4, double4, int4, double4, op>  },
                },
                {
                    { device::fmx<double, double, float, uchar, op>, device::fmx<double2, double2, float2, uchar2, op>, device::fmx<double3, double3, float3, uchar3, op>, device::fmx<double4, double4, float4, uchar4, op>  },
                    { device::fmx<double, double, float, schar, op>, device::fmx<double2, double2, float2, char2, op>, device::fmx<double3, double3, float3, char3, op>, device::fmx<double4, double4, float4, char4, op>  },
                    { device::fmx<double, double, float, ushort, op>, device::fmx<double2, double2, float2, ushort2, op>, device::fmx<double3, double3, float3, ushort3, op>, device::fmx<double4, double4, float4, ushort4, op>  },
                    { device::fmx<double, double, float, short, op>, device::fmx<double2, double2, float2, short2, op>, device::fmx<double3, double3, float3, short3, op>, device::fmx<double4, double4, float4, short4, op>  },
                    { device::fmx<double, double, float, int, op>, device::fmx<double2, double2, float2, int2, op>, device::fmx<double3, double3, float3, int3, op>, device::fmx<double4, double4, float4, int4, op>  },
                    { device::fmx<double, double, float, float, op>, device::fmx<double2, double2, float2, float2, op>, device::fmx<double3, double3, float3, float3, op>, device::fmx<double4, double4, float4, float4, op>  },
                    { device::fmx<double, double, float, double, op>, device::fmx<double2, double2, float2, double2, op>, device::fmx<double3, double3, float3, double3, op>, device::fmx<double4, double4, float4, double4, op>  },
                },
                {
                    { device::fmx<double, double, double, uchar, op>, device::fmx<double2, double2, double2, uchar2, op>, device::fmx<double3, double3, double3, uchar3, op>, device::fmx<double4, double4, double4, uchar4, op>  },
                    { device::fmx<double, double, double, schar, op>, device::fmx<double2, double2, double2, char2, op>, device::fmx<double3, double3, double3, char3, op>, device::fmx<double4, double4, double4, char4, op>  },
                    { device::fmx<double, double, double, ushort, op>, device::fmx<double2, double2, double2, ushort2, op>, device::fmx<double3, double3, double3, ushort3, op>, device::fmx<double4, double4, double4, ushort4, op>  },
                    { device::fmx<double, double, double, short, op>, device::fmx<double2, double2, double2, short2, op>, device::fmx<double3, double3, double3, short3, op>, device::fmx<double4, double4, double4, short4, op>  },
                    { device::fmx<double, double, double, int, op>, device::fmx<double2, double2, double2, int2, op>, device::fmx<double3, double3, double3, int3, op>, device::fmx<double4, double4, double4, int4, op>  },
                    { device::fmx<double, double, double, float, op>, device::fmx<double2, double2, double2, float2, op>, device::fmx<double3, double3, double3, float3, op>, device::fmx<double4, double4, double4, float4, op>  },
                    { device::fmx<double, double, double, double, op>, device::fmx<double2, double2, double2, double2, op>, device::fmx<double3, double3, double3, double3, op>, device::fmx<double4, double4, double4, double4, op>  },
                },
            },
        }
    };


    GpuMat src1(_src1), src2(_src2), src3(_src3);
    GpuMat& dst = _dst;



    int stype = CV_MAKETYPE(std::min(std::min(_src1.depth(), _src2.depth()), _src3.depth()), _src1.channels());
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int wdepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruction_needed(false);

    if(cn>4)
    {
        reconstruction_needed = true;

        GpuMat tmp;

        if(!src1.isContinuous())
        {
            src1.copyTo(tmp, _stream);
            src1.release();
            src1 = tmp;
        }

        tmp = src1.reshape(1);
        src1 = tmp;

        if(!src2.isContinuous())
        {
            src2.copyTo(tmp, _stream);
            src2.release();
            src2 = tmp;
        }

        tmp = src2.reshape(1);
        src2 = tmp;

        if(!src3.isContinuous())
        {
            src3.copyTo(tmp, _stream);
            src3.release();
            src3 = tmp;
        }

        tmp = src3.reshape(1);
        src3 = tmp;
    }

    dst.create(src1.size(), !reconstruction_needed ? wtype : wdepth);

    function_type fun = functions[src1.depth()][src2.depth()][src3.depth()][dst.depth()][dst.channels()-1];

    fun(src1, src2, src3, dst, _mask, _stream);

    if(reconstruction_needed)
    {
        GpuMat tmp;

        tmp = dst.reshape(cn);
        dst = tmp;
    }
}

template<int op>
void fmx_aosoa(const GpuMat& _src1, const Scalar& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
    CV_Assert((_src1.size() == _src3.size()) && (_mask.empty() || (_mask.size() == _src1.size())) );

    typedef void (*function_type)(const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

    static const function_type functions[7][7][7][4] =
    {
        {
            {
                    { device::fmx<uchar, uchar, uchar, op>, device::fmx<uchar2, uchar2, uchar2, op>, device::fmx<uchar3, uchar3, uchar3, op>, device::fmx<uchar4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, uchar, schar, op>, device::fmx<uchar2, uchar2, char2, op>, device::fmx<uchar3, uchar3, char3, op>, device::fmx<uchar4, uchar4, char4, op>  },
                    { device::fmx<uchar, uchar, ushort, op>, device::fmx<uchar2, uchar2, ushort2, op>, device::fmx<uchar3, uchar3, ushort3, op>, device::fmx<uchar4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, uchar, short, op>, device::fmx<uchar2, uchar2, short2, op>, device::fmx<uchar3, uchar3, short3, op>, device::fmx<uchar4, uchar4, short4, op>  },
                    { device::fmx<uchar, uchar, int, op>, device::fmx<uchar2, uchar2, int2, op>, device::fmx<uchar3, uchar3, int3, op>, device::fmx<uchar4, uchar4, int4, op>  },
                    { device::fmx<uchar, uchar, float, op>, device::fmx<uchar2, uchar2, float2, op>, device::fmx<uchar3, uchar3, float3, op>, device::fmx<uchar4, uchar4, float4, op>  },
                    { device::fmx<uchar, uchar, double, op>, device::fmx<uchar2, uchar2, double2, op>, device::fmx<uchar3, uchar3, double3, op>, device::fmx<uchar4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<uchar, schar, uchar, op>, device::fmx<uchar2, char2, uchar2, op>, device::fmx<uchar3, char3, uchar3, op>, device::fmx<uchar4, char4, uchar4, op>  },
                    { device::fmx<uchar, schar, schar, op>, device::fmx<uchar2, char2, char2, op>, device::fmx<uchar3, char3, char3, op>, device::fmx<uchar4, char4, char4, op>  },
                    { device::fmx<uchar, schar, ushort, op>, device::fmx<uchar2, char2, ushort2, op>, device::fmx<uchar3, char3, ushort3, op>, device::fmx<uchar4, char4, ushort4, op>  },
                    { device::fmx<uchar, schar, short, op>, device::fmx<uchar2, char2, short2, op>, device::fmx<uchar3, char3, short3, op>, device::fmx<uchar4, char4, short4, op>  },
                    { device::fmx<uchar, schar, int, op>, device::fmx<uchar2, char2, int2, op>, device::fmx<uchar3, char3, int3, op>, device::fmx<uchar4, char4, int4, op>  },
                    { device::fmx<uchar, schar, float, op>, device::fmx<uchar2, char2, float2, op>, device::fmx<uchar3, char3, float3, op>, device::fmx<uchar4, char4, float4, op>  },
                    { device::fmx<uchar, schar, double, op>, device::fmx<uchar2, char2, double2, op>, device::fmx<uchar3, char3, double3, op>, device::fmx<uchar4, char4, double4, op>  },
            },
            {
                    { device::fmx<uchar, ushort, uchar, op>, device::fmx<uchar2, ushort2, uchar2, op>, device::fmx<uchar3, ushort3, uchar3, op>, device::fmx<uchar4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, ushort, schar, op>, device::fmx<uchar2, ushort2, char2, op>, device::fmx<uchar3, ushort3, char3, op>, device::fmx<uchar4, ushort4, char4, op>  },
                    { device::fmx<uchar, ushort, ushort, op>, device::fmx<uchar2, ushort2, ushort2, op>, device::fmx<uchar3, ushort3, ushort3, op>, device::fmx<uchar4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, ushort, short, op>, device::fmx<uchar2, ushort2, short2, op>, device::fmx<uchar3, ushort3, short3, op>, device::fmx<uchar4, ushort4, short4, op>  },
                    { device::fmx<uchar, ushort, int, op>, device::fmx<uchar2, ushort2, int2, op>, device::fmx<uchar3, ushort3, int3, op>, device::fmx<uchar4, ushort4, int4, op>  },
                    { device::fmx<uchar, ushort, float, op>, device::fmx<uchar2, ushort2, float2, op>, device::fmx<uchar3, ushort3, float3, op>, device::fmx<uchar4, ushort4, float4, op>  },
                    { device::fmx<uchar, ushort, double, op>, device::fmx<uchar2, ushort2, double2, op>, device::fmx<uchar3, ushort3, double3, op>, device::fmx<uchar4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<uchar, short, uchar, op>, device::fmx<uchar2, short2, uchar2, op>, device::fmx<uchar3, short3, uchar3, op>, device::fmx<uchar4, short4, uchar4, op>  },
                    { device::fmx<uchar, short, schar, op>, device::fmx<uchar2, short2, char2, op>, device::fmx<uchar3, short3, char3, op>, device::fmx<uchar4, short4, char4, op>  },
                    { device::fmx<uchar, short, ushort, op>, device::fmx<uchar2, short2, ushort2, op>, device::fmx<uchar3, short3, ushort3, op>, device::fmx<uchar4, short4, ushort4, op>  },
                    { device::fmx<uchar, short, short, op>, device::fmx<uchar2, short2, short2, op>, device::fmx<uchar3, short3, short3, op>, device::fmx<uchar4, short4, short4, op>  },
                    { device::fmx<uchar, short, int, op>, device::fmx<uchar2, short2, int2, op>, device::fmx<uchar3, short3, int3, op>, device::fmx<uchar4, short4, int4, op>  },
                    { device::fmx<uchar, short, float, op>, device::fmx<uchar2, short2, float2, op>, device::fmx<uchar3, short3, float3, op>, device::fmx<uchar4, short4, float4, op>  },
                    { device::fmx<uchar, short, double, op>, device::fmx<uchar2, short2, double2, op>, device::fmx<uchar3, short3, double3, op>, device::fmx<uchar4, short4, double4, op>  },
            },
            {
                    { device::fmx<uchar, int, uchar, op>, device::fmx<uchar2, int2, uchar2, op>, device::fmx<uchar3, int3, uchar3, op>, device::fmx<uchar4, int4, uchar4, op>  },
                    { device::fmx<uchar, int, schar, op>, device::fmx<uchar2, int2, char2, op>, device::fmx<uchar3, int3, char3, op>, device::fmx<uchar4, int4, char4, op>  },
                    { device::fmx<uchar, int, ushort, op>, device::fmx<uchar2, int2, ushort2, op>, device::fmx<uchar3, int3, ushort3, op>, device::fmx<uchar4, int4, ushort4, op>  },
                    { device::fmx<uchar, int, short, op>, device::fmx<uchar2, int2, short2, op>, device::fmx<uchar3, int3, short3, op>, device::fmx<uchar4, int4, short4, op>  },
                    { device::fmx<uchar, int, int, op>, device::fmx<uchar2, int2, int2, op>, device::fmx<uchar3, int3, int3, op>, device::fmx<uchar4, int4, int4, op>  },
                    { device::fmx<uchar, int, float, op>, device::fmx<uchar2, int2, float2, op>, device::fmx<uchar3, int3, float3, op>, device::fmx<uchar4, int4, float4, op>  },
                    { device::fmx<uchar, int, double, op>, device::fmx<uchar2, int2, double2, op>, device::fmx<uchar3, int3, double3, op>, device::fmx<uchar4, int4, double4, op>  },
            },
            {
                    { device::fmx<uchar, float, uchar, op>, device::fmx<uchar2, float2, uchar2, op>, device::fmx<uchar3, float3, uchar3, op>, device::fmx<uchar4, float4, uchar4, op>  },
                    { device::fmx<uchar, float, schar, op>, device::fmx<uchar2, float2, char2, op>, device::fmx<uchar3, float3, char3, op>, device::fmx<uchar4, float4, char4, op>  },
                    { device::fmx<uchar, float, ushort, op>, device::fmx<uchar2, float2, ushort2, op>, device::fmx<uchar3, float3, ushort3, op>, device::fmx<uchar4, float4, ushort4, op>  },
                    { device::fmx<uchar, float, short, op>, device::fmx<uchar2, float2, short2, op>, device::fmx<uchar3, float3, short3, op>, device::fmx<uchar4, float4, short4, op>  },
                    { device::fmx<uchar, float, int, op>, device::fmx<uchar2, float2, int2, op>, device::fmx<uchar3, float3, int3, op>, device::fmx<uchar4, float4, int4, op>  },
                    { device::fmx<uchar, float, float, op>, device::fmx<uchar2, float2, float2, op>, device::fmx<uchar3, float3, float3, op>, device::fmx<uchar4, float4, float4, op>  },
                    { device::fmx<uchar, float, double, op>, device::fmx<uchar2, float2, double2, op>, device::fmx<uchar3, float3, double3, op>, device::fmx<uchar4, float4, double4, op>  },
            },
            {
                    { device::fmx<uchar, double, uchar, op>, device::fmx<uchar2, double2, uchar2, op>, device::fmx<uchar3, double3, uchar3, op>, device::fmx<uchar4, double4, uchar4, op>  },
                    { device::fmx<uchar, double, schar, op>, device::fmx<uchar2, double2, char2, op>, device::fmx<uchar3, double3, char3, op>, device::fmx<uchar4, double4, char4, op>  },
                    { device::fmx<uchar, double, ushort, op>, device::fmx<uchar2, double2, ushort2, op>, device::fmx<uchar3, double3, ushort3, op>, device::fmx<uchar4, double4, ushort4, op>  },
                    { device::fmx<uchar, double, short, op>, device::fmx<uchar2, double2, short2, op>, device::fmx<uchar3, double3, short3, op>, device::fmx<uchar4, double4, short4, op>  },
                    { device::fmx<uchar, double, int, op>, device::fmx<uchar2, double2, int2, op>, device::fmx<uchar3, double3, int3, op>, device::fmx<uchar4, double4, int4, op>  },
                    { device::fmx<uchar, double, float, op>, device::fmx<uchar2, double2, float2, op>, device::fmx<uchar3, double3, float3, op>, device::fmx<uchar4, double4, float4, op>  },
                    { device::fmx<uchar, double, double, op>, device::fmx<uchar2, double2, double2, op>, device::fmx<uchar3, double3, double3, op>, device::fmx<uchar4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<schar, uchar, uchar, op>, device::fmx<char2, uchar2, uchar2, op>, device::fmx<char3, uchar3, uchar3, op>, device::fmx<char4, uchar4, uchar4, op>  },
                    { device::fmx<schar, uchar, schar, op>, device::fmx<char2, uchar2, char2, op>, device::fmx<char3, uchar3, char3, op>, device::fmx<char4, uchar4, char4, op>  },
                    { device::fmx<schar, uchar, ushort, op>, device::fmx<char2, uchar2, ushort2, op>, device::fmx<char3, uchar3, ushort3, op>, device::fmx<char4, uchar4, ushort4, op>  },
                    { device::fmx<schar, uchar, short, op>, device::fmx<char2, uchar2, short2, op>, device::fmx<char3, uchar3, short3, op>, device::fmx<char4, uchar4, short4, op>  },
                    { device::fmx<schar, uchar, int, op>, device::fmx<char2, uchar2, int2, op>, device::fmx<char3, uchar3, int3, op>, device::fmx<char4, uchar4, int4, op>  },
                    { device::fmx<schar, uchar, float, op>, device::fmx<char2, uchar2, float2, op>, device::fmx<char3, uchar3, float3, op>, device::fmx<char4, uchar4, float4, op>  },
                    { device::fmx<schar, uchar, double, op>, device::fmx<char2, uchar2, double2, op>, device::fmx<char3, uchar3, double3, op>, device::fmx<char4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<schar, schar, uchar, op>, device::fmx<char2, char2, uchar2, op>, device::fmx<char3, char3, uchar3, op>, device::fmx<char4, char4, uchar4, op>  },
                    { device::fmx<schar, schar, schar, op>, device::fmx<char2, char2, char2, op>, device::fmx<char3, char3, char3, op>, device::fmx<char4, char4, char4, op>  },
                    { device::fmx<schar, schar, ushort, op>, device::fmx<char2, char2, ushort2, op>, device::fmx<char3, char3, ushort3, op>, device::fmx<char4, char4, ushort4, op>  },
                    { device::fmx<schar, schar, short, op>, device::fmx<char2, char2, short2, op>, device::fmx<char3, char3, short3, op>, device::fmx<char4, char4, short4, op>  },
                    { device::fmx<schar, schar, int, op>, device::fmx<char2, char2, int2, op>, device::fmx<char3, char3, int3, op>, device::fmx<char4, char4, int4, op>  },
                    { device::fmx<schar, schar, float, op>, device::fmx<char2, char2, float2, op>, device::fmx<char3, char3, float3, op>, device::fmx<char4, char4, float4, op>  },
                    { device::fmx<schar, schar, double, op>, device::fmx<char2, char2, double2, op>, device::fmx<char3, char3, double3, op>, device::fmx<char4, char4, double4, op>  },
            },
            {
                    { device::fmx<schar, ushort, uchar, op>, device::fmx<char2, ushort2, uchar2, op>, device::fmx<char3, ushort3, uchar3, op>, device::fmx<char4, ushort4, uchar4, op>  },
                    { device::fmx<schar, ushort, schar, op>, device::fmx<char2, ushort2, char2, op>, device::fmx<char3, ushort3, char3, op>, device::fmx<char4, ushort4, char4, op>  },
                    { device::fmx<schar, ushort, ushort, op>, device::fmx<char2, ushort2, ushort2, op>, device::fmx<char3, ushort3, ushort3, op>, device::fmx<char4, ushort4, ushort4, op>  },
                    { device::fmx<schar, ushort, short, op>, device::fmx<char2, ushort2, short2, op>, device::fmx<char3, ushort3, short3, op>, device::fmx<char4, ushort4, short4, op>  },
                    { device::fmx<schar, ushort, int, op>, device::fmx<char2, ushort2, int2, op>, device::fmx<char3, ushort3, int3, op>, device::fmx<char4, ushort4, int4, op>  },
                    { device::fmx<schar, ushort, float, op>, device::fmx<char2, ushort2, float2, op>, device::fmx<char3, ushort3, float3, op>, device::fmx<char4, ushort4, float4, op>  },
                    { device::fmx<schar, ushort, double, op>, device::fmx<char2, ushort2, double2, op>, device::fmx<char3, ushort3, double3, op>, device::fmx<char4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<schar, short, uchar, op>, device::fmx<char2, short2, uchar2, op>, device::fmx<char3, short3, uchar3, op>, device::fmx<char4, short4, uchar4, op>  },
                    { device::fmx<schar, short, schar, op>, device::fmx<char2, short2, char2, op>, device::fmx<char3, short3, char3, op>, device::fmx<char4, short4, char4, op>  },
                    { device::fmx<schar, short, ushort, op>, device::fmx<char2, short2, ushort2, op>, device::fmx<char3, short3, ushort3, op>, device::fmx<char4, short4, ushort4, op>  },
                    { device::fmx<schar, short, short, op>, device::fmx<char2, short2, short2, op>, device::fmx<char3, short3, short3, op>, device::fmx<char4, short4, short4, op>  },
                    { device::fmx<schar, short, int, op>, device::fmx<char2, short2, int2, op>, device::fmx<char3, short3, int3, op>, device::fmx<char4, short4, int4, op>  },
                    { device::fmx<schar, short, float, op>, device::fmx<char2, short2, float2, op>, device::fmx<char3, short3, float3, op>, device::fmx<char4, short4, float4, op>  },
                    { device::fmx<schar, short, double, op>, device::fmx<char2, short2, double2, op>, device::fmx<char3, short3, double3, op>, device::fmx<char4, short4, double4, op>  },
            },
            {
                    { device::fmx<schar, int, uchar, op>, device::fmx<char2, int2, uchar2, op>, device::fmx<char3, int3, uchar3, op>, device::fmx<char4, int4, uchar4, op>  },
                    { device::fmx<schar, int, schar, op>, device::fmx<char2, int2, char2, op>, device::fmx<char3, int3, char3, op>, device::fmx<char4, int4, char4, op>  },
                    { device::fmx<schar, int, ushort, op>, device::fmx<char2, int2, ushort2, op>, device::fmx<char3, int3, ushort3, op>, device::fmx<char4, int4, ushort4, op>  },
                    { device::fmx<schar, int, short, op>, device::fmx<char2, int2, short2, op>, device::fmx<char3, int3, short3, op>, device::fmx<char4, int4, short4, op>  },
                    { device::fmx<schar, int, int, op>, device::fmx<char2, int2, int2, op>, device::fmx<char3, int3, int3, op>, device::fmx<char4, int4, int4, op>  },
                    { device::fmx<schar, int, float, op>, device::fmx<char2, int2, float2, op>, device::fmx<char3, int3, float3, op>, device::fmx<char4, int4, float4, op>  },
                    { device::fmx<schar, int, double, op>, device::fmx<char2, int2, double2, op>, device::fmx<char3, int3, double3, op>, device::fmx<char4, int4, double4, op>  },
            },
            {
                    { device::fmx<schar, float, uchar, op>, device::fmx<char2, float2, uchar2, op>, device::fmx<char3, float3, uchar3, op>, device::fmx<char4, float4, uchar4, op>  },
                    { device::fmx<schar, float, schar, op>, device::fmx<char2, float2, char2, op>, device::fmx<char3, float3, char3, op>, device::fmx<char4, float4, char4, op>  },
                    { device::fmx<schar, float, ushort, op>, device::fmx<char2, float2, ushort2, op>, device::fmx<char3, float3, ushort3, op>, device::fmx<char4, float4, ushort4, op>  },
                    { device::fmx<schar, float, short, op>, device::fmx<char2, float2, short2, op>, device::fmx<char3, float3, short3, op>, device::fmx<char4, float4, short4, op>  },
                    { device::fmx<schar, float, int, op>, device::fmx<char2, float2, int2, op>, device::fmx<char3, float3, int3, op>, device::fmx<char4, float4, int4, op>  },
                    { device::fmx<schar, float, float, op>, device::fmx<char2, float2, float2, op>, device::fmx<char3, float3, float3, op>, device::fmx<char4, float4, float4, op>  },
                    { device::fmx<schar, float, double, op>, device::fmx<char2, float2, double2, op>, device::fmx<char3, float3, double3, op>, device::fmx<char4, float4, double4, op>  },
            },
            {
                    { device::fmx<schar, double, uchar, op>, device::fmx<char2, double2, uchar2, op>, device::fmx<char3, double3, uchar3, op>, device::fmx<char4, double4, uchar4, op>  },
                    { device::fmx<schar, double, schar, op>, device::fmx<char2, double2, char2, op>, device::fmx<char3, double3, char3, op>, device::fmx<char4, double4, char4, op>  },
                    { device::fmx<schar, double, ushort, op>, device::fmx<char2, double2, ushort2, op>, device::fmx<char3, double3, ushort3, op>, device::fmx<char4, double4, ushort4, op>  },
                    { device::fmx<schar, double, short, op>, device::fmx<char2, double2, short2, op>, device::fmx<char3, double3, short3, op>, device::fmx<char4, double4, short4, op>  },
                    { device::fmx<schar, double, int, op>, device::fmx<char2, double2, int2, op>, device::fmx<char3, double3, int3, op>, device::fmx<char4, double4, int4, op>  },
                    { device::fmx<schar, double, float, op>, device::fmx<char2, double2, float2, op>, device::fmx<char3, double3, float3, op>, device::fmx<char4, double4, float4, op>  },
                    { device::fmx<schar, double, double, op>, device::fmx<char2, double2, double2, op>, device::fmx<char3, double3, double3, op>, device::fmx<char4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<ushort, uchar, uchar, op>, device::fmx<ushort2, uchar2, uchar2, op>, device::fmx<ushort3, uchar3, uchar3, op>, device::fmx<ushort4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, uchar, schar, op>, device::fmx<ushort2, uchar2, char2, op>, device::fmx<ushort3, uchar3, char3, op>, device::fmx<ushort4, uchar4, char4, op>  },
                    { device::fmx<ushort, uchar, ushort, op>, device::fmx<ushort2, uchar2, ushort2, op>, device::fmx<ushort3, uchar3, ushort3, op>, device::fmx<ushort4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, uchar, short, op>, device::fmx<ushort2, uchar2, short2, op>, device::fmx<ushort3, uchar3, short3, op>, device::fmx<ushort4, uchar4, short4, op>  },
                    { device::fmx<ushort, uchar, int, op>, device::fmx<ushort2, uchar2, int2, op>, device::fmx<ushort3, uchar3, int3, op>, device::fmx<ushort4, uchar4, int4, op>  },
                    { device::fmx<ushort, uchar, float, op>, device::fmx<ushort2, uchar2, float2, op>, device::fmx<ushort3, uchar3, float3, op>, device::fmx<ushort4, uchar4, float4, op>  },
                    { device::fmx<ushort, uchar, double, op>, device::fmx<ushort2, uchar2, double2, op>, device::fmx<ushort3, uchar3, double3, op>, device::fmx<ushort4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<ushort, schar, uchar, op>, device::fmx<ushort2, char2, uchar2, op>, device::fmx<ushort3, char3, uchar3, op>, device::fmx<ushort4, char4, uchar4, op>  },
                    { device::fmx<ushort, schar, schar, op>, device::fmx<ushort2, char2, char2, op>, device::fmx<ushort3, char3, char3, op>, device::fmx<ushort4, char4, char4, op>  },
                    { device::fmx<ushort, schar, ushort, op>, device::fmx<ushort2, char2, ushort2, op>, device::fmx<ushort3, char3, ushort3, op>, device::fmx<ushort4, char4, ushort4, op>  },
                    { device::fmx<ushort, schar, short, op>, device::fmx<ushort2, char2, short2, op>, device::fmx<ushort3, char3, short3, op>, device::fmx<ushort4, char4, short4, op>  },
                    { device::fmx<ushort, schar, int, op>, device::fmx<ushort2, char2, int2, op>, device::fmx<ushort3, char3, int3, op>, device::fmx<ushort4, char4, int4, op>  },
                    { device::fmx<ushort, schar, float, op>, device::fmx<ushort2, char2, float2, op>, device::fmx<ushort3, char3, float3, op>, device::fmx<ushort4, char4, float4, op>  },
                    { device::fmx<ushort, schar, double, op>, device::fmx<ushort2, char2, double2, op>, device::fmx<ushort3, char3, double3, op>, device::fmx<ushort4, char4, double4, op>  },
            },
            {
                    { device::fmx<ushort, ushort, uchar, op>, device::fmx<ushort2, ushort2, uchar2, op>, device::fmx<ushort3, ushort3, uchar3, op>, device::fmx<ushort4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, ushort, schar, op>, device::fmx<ushort2, ushort2, char2, op>, device::fmx<ushort3, ushort3, char3, op>, device::fmx<ushort4, ushort4, char4, op>  },
                    { device::fmx<ushort, ushort, ushort, op>, device::fmx<ushort2, ushort2, ushort2, op>, device::fmx<ushort3, ushort3, ushort3, op>, device::fmx<ushort4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, ushort, short, op>, device::fmx<ushort2, ushort2, short2, op>, device::fmx<ushort3, ushort3, short3, op>, device::fmx<ushort4, ushort4, short4, op>  },
                    { device::fmx<ushort, ushort, int, op>, device::fmx<ushort2, ushort2, int2, op>, device::fmx<ushort3, ushort3, int3, op>, device::fmx<ushort4, ushort4, int4, op>  },
                    { device::fmx<ushort, ushort, float, op>, device::fmx<ushort2, ushort2, float2, op>, device::fmx<ushort3, ushort3, float3, op>, device::fmx<ushort4, ushort4, float4, op>  },
                    { device::fmx<ushort, ushort, double, op>, device::fmx<ushort2, ushort2, double2, op>, device::fmx<ushort3, ushort3, double3, op>, device::fmx<ushort4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<ushort, short, uchar, op>, device::fmx<ushort2, short2, uchar2, op>, device::fmx<ushort3, short3, uchar3, op>, device::fmx<ushort4, short4, uchar4, op>  },
                    { device::fmx<ushort, short, schar, op>, device::fmx<ushort2, short2, char2, op>, device::fmx<ushort3, short3, char3, op>, device::fmx<ushort4, short4, char4, op>  },
                    { device::fmx<ushort, short, ushort, op>, device::fmx<ushort2, short2, ushort2, op>, device::fmx<ushort3, short3, ushort3, op>, device::fmx<ushort4, short4, ushort4, op>  },
                    { device::fmx<ushort, short, short, op>, device::fmx<ushort2, short2, short2, op>, device::fmx<ushort3, short3, short3, op>, device::fmx<ushort4, short4, short4, op>  },
                    { device::fmx<ushort, short, int, op>, device::fmx<ushort2, short2, int2, op>, device::fmx<ushort3, short3, int3, op>, device::fmx<ushort4, short4, int4, op>  },
                    { device::fmx<ushort, short, float, op>, device::fmx<ushort2, short2, float2, op>, device::fmx<ushort3, short3, float3, op>, device::fmx<ushort4, short4, float4, op>  },
                    { device::fmx<ushort, short, double, op>, device::fmx<ushort2, short2, double2, op>, device::fmx<ushort3, short3, double3, op>, device::fmx<ushort4, short4, double4, op>  },
            },
            {
                    { device::fmx<ushort, int, uchar, op>, device::fmx<ushort2, int2, uchar2, op>, device::fmx<ushort3, int3, uchar3, op>, device::fmx<ushort4, int4, uchar4, op>  },
                    { device::fmx<ushort, int, schar, op>, device::fmx<ushort2, int2, char2, op>, device::fmx<ushort3, int3, char3, op>, device::fmx<ushort4, int4, char4, op>  },
                    { device::fmx<ushort, int, ushort, op>, device::fmx<ushort2, int2, ushort2, op>, device::fmx<ushort3, int3, ushort3, op>, device::fmx<ushort4, int4, ushort4, op>  },
                    { device::fmx<ushort, int, short, op>, device::fmx<ushort2, int2, short2, op>, device::fmx<ushort3, int3, short3, op>, device::fmx<ushort4, int4, short4, op>  },
                    { device::fmx<ushort, int, int, op>, device::fmx<ushort2, int2, int2, op>, device::fmx<ushort3, int3, int3, op>, device::fmx<ushort4, int4, int4, op>  },
                    { device::fmx<ushort, int, float, op>, device::fmx<ushort2, int2, float2, op>, device::fmx<ushort3, int3, float3, op>, device::fmx<ushort4, int4, float4, op>  },
                    { device::fmx<ushort, int, double, op>, device::fmx<ushort2, int2, double2, op>, device::fmx<ushort3, int3, double3, op>, device::fmx<ushort4, int4, double4, op>  },
            },
            {
                    { device::fmx<ushort, float, uchar, op>, device::fmx<ushort2, float2, uchar2, op>, device::fmx<ushort3, float3, uchar3, op>, device::fmx<ushort4, float4, uchar4, op>  },
                    { device::fmx<ushort, float, schar, op>, device::fmx<ushort2, float2, char2, op>, device::fmx<ushort3, float3, char3, op>, device::fmx<ushort4, float4, char4, op>  },
                    { device::fmx<ushort, float, ushort, op>, device::fmx<ushort2, float2, ushort2, op>, device::fmx<ushort3, float3, ushort3, op>, device::fmx<ushort4, float4, ushort4, op>  },
                    { device::fmx<ushort, float, short, op>, device::fmx<ushort2, float2, short2, op>, device::fmx<ushort3, float3, short3, op>, device::fmx<ushort4, float4, short4, op>  },
                    { device::fmx<ushort, float, int, op>, device::fmx<ushort2, float2, int2, op>, device::fmx<ushort3, float3, int3, op>, device::fmx<ushort4, float4, int4, op>  },
                    { device::fmx<ushort, float, float, op>, device::fmx<ushort2, float2, float2, op>, device::fmx<ushort3, float3, float3, op>, device::fmx<ushort4, float4, float4, op>  },
                    { device::fmx<ushort, float, double, op>, device::fmx<ushort2, float2, double2, op>, device::fmx<ushort3, float3, double3, op>, device::fmx<ushort4, float4, double4, op>  },
            },
            {
                    { device::fmx<ushort, double, uchar, op>, device::fmx<ushort2, double2, uchar2, op>, device::fmx<ushort3, double3, uchar3, op>, device::fmx<ushort4, double4, uchar4, op>  },
                    { device::fmx<ushort, double, schar, op>, device::fmx<ushort2, double2, char2, op>, device::fmx<ushort3, double3, char3, op>, device::fmx<ushort4, double4, char4, op>  },
                    { device::fmx<ushort, double, ushort, op>, device::fmx<ushort2, double2, ushort2, op>, device::fmx<ushort3, double3, ushort3, op>, device::fmx<ushort4, double4, ushort4, op>  },
                    { device::fmx<ushort, double, short, op>, device::fmx<ushort2, double2, short2, op>, device::fmx<ushort3, double3, short3, op>, device::fmx<ushort4, double4, short4, op>  },
                    { device::fmx<ushort, double, int, op>, device::fmx<ushort2, double2, int2, op>, device::fmx<ushort3, double3, int3, op>, device::fmx<ushort4, double4, int4, op>  },
                    { device::fmx<ushort, double, float, op>, device::fmx<ushort2, double2, float2, op>, device::fmx<ushort3, double3, float3, op>, device::fmx<ushort4, double4, float4, op>  },
                    { device::fmx<ushort, double, double, op>, device::fmx<ushort2, double2, double2, op>, device::fmx<ushort3, double3, double3, op>, device::fmx<ushort4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<short, uchar, uchar, op>, device::fmx<short2, uchar2, uchar2, op>, device::fmx<short3, uchar3, uchar3, op>, device::fmx<short4, uchar4, uchar4, op>  },
                    { device::fmx<short, uchar, schar, op>, device::fmx<short2, uchar2, char2, op>, device::fmx<short3, uchar3, char3, op>, device::fmx<short4, uchar4, char4, op>  },
                    { device::fmx<short, uchar, ushort, op>, device::fmx<short2, uchar2, ushort2, op>, device::fmx<short3, uchar3, ushort3, op>, device::fmx<short4, uchar4, ushort4, op>  },
                    { device::fmx<short, uchar, short, op>, device::fmx<short2, uchar2, short2, op>, device::fmx<short3, uchar3, short3, op>, device::fmx<short4, uchar4, short4, op>  },
                    { device::fmx<short, uchar, int, op>, device::fmx<short2, uchar2, int2, op>, device::fmx<short3, uchar3, int3, op>, device::fmx<short4, uchar4, int4, op>  },
                    { device::fmx<short, uchar, float, op>, device::fmx<short2, uchar2, float2, op>, device::fmx<short3, uchar3, float3, op>, device::fmx<short4, uchar4, float4, op>  },
                    { device::fmx<short, uchar, double, op>, device::fmx<short2, uchar2, double2, op>, device::fmx<short3, uchar3, double3, op>, device::fmx<short4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<short, schar, uchar, op>, device::fmx<short2, char2, uchar2, op>, device::fmx<short3, char3, uchar3, op>, device::fmx<short4, char4, uchar4, op>  },
                    { device::fmx<short, schar, schar, op>, device::fmx<short2, char2, char2, op>, device::fmx<short3, char3, char3, op>, device::fmx<short4, char4, char4, op>  },
                    { device::fmx<short, schar, ushort, op>, device::fmx<short2, char2, ushort2, op>, device::fmx<short3, char3, ushort3, op>, device::fmx<short4, char4, ushort4, op>  },
                    { device::fmx<short, schar, short, op>, device::fmx<short2, char2, short2, op>, device::fmx<short3, char3, short3, op>, device::fmx<short4, char4, short4, op>  },
                    { device::fmx<short, schar, int, op>, device::fmx<short2, char2, int2, op>, device::fmx<short3, char3, int3, op>, device::fmx<short4, char4, int4, op>  },
                    { device::fmx<short, schar, float, op>, device::fmx<short2, char2, float2, op>, device::fmx<short3, char3, float3, op>, device::fmx<short4, char4, float4, op>  },
                    { device::fmx<short, schar, double, op>, device::fmx<short2, char2, double2, op>, device::fmx<short3, char3, double3, op>, device::fmx<short4, char4, double4, op>  },
            },
            {
                    { device::fmx<short, ushort, uchar, op>, device::fmx<short2, ushort2, uchar2, op>, device::fmx<short3, ushort3, uchar3, op>, device::fmx<short4, ushort4, uchar4, op>  },
                    { device::fmx<short, ushort, schar, op>, device::fmx<short2, ushort2, char2, op>, device::fmx<short3, ushort3, char3, op>, device::fmx<short4, ushort4, char4, op>  },
                    { device::fmx<short, ushort, ushort, op>, device::fmx<short2, ushort2, ushort2, op>, device::fmx<short3, ushort3, ushort3, op>, device::fmx<short4, ushort4, ushort4, op>  },
                    { device::fmx<short, ushort, short, op>, device::fmx<short2, ushort2, short2, op>, device::fmx<short3, ushort3, short3, op>, device::fmx<short4, ushort4, short4, op>  },
                    { device::fmx<short, ushort, int, op>, device::fmx<short2, ushort2, int2, op>, device::fmx<short3, ushort3, int3, op>, device::fmx<short4, ushort4, int4, op>  },
                    { device::fmx<short, ushort, float, op>, device::fmx<short2, ushort2, float2, op>, device::fmx<short3, ushort3, float3, op>, device::fmx<short4, ushort4, float4, op>  },
                    { device::fmx<short, ushort, double, op>, device::fmx<short2, ushort2, double2, op>, device::fmx<short3, ushort3, double3, op>, device::fmx<short4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<short, short, uchar, op>, device::fmx<short2, short2, uchar2, op>, device::fmx<short3, short3, uchar3, op>, device::fmx<short4, short4, uchar4, op>  },
                    { device::fmx<short, short, schar, op>, device::fmx<short2, short2, char2, op>, device::fmx<short3, short3, char3, op>, device::fmx<short4, short4, char4, op>  },
                    { device::fmx<short, short, ushort, op>, device::fmx<short2, short2, ushort2, op>, device::fmx<short3, short3, ushort3, op>, device::fmx<short4, short4, ushort4, op>  },
                    { device::fmx<short, short, short, op>, device::fmx<short2, short2, short2, op>, device::fmx<short3, short3, short3, op>, device::fmx<short4, short4, short4, op>  },
                    { device::fmx<short, short, int, op>, device::fmx<short2, short2, int2, op>, device::fmx<short3, short3, int3, op>, device::fmx<short4, short4, int4, op>  },
                    { device::fmx<short, short, float, op>, device::fmx<short2, short2, float2, op>, device::fmx<short3, short3, float3, op>, device::fmx<short4, short4, float4, op>  },
                    { device::fmx<short, short, double, op>, device::fmx<short2, short2, double2, op>, device::fmx<short3, short3, double3, op>, device::fmx<short4, short4, double4, op>  },
            },
            {
                    { device::fmx<short, int, uchar, op>, device::fmx<short2, int2, uchar2, op>, device::fmx<short3, int3, uchar3, op>, device::fmx<short4, int4, uchar4, op>  },
                    { device::fmx<short, int, schar, op>, device::fmx<short2, int2, char2, op>, device::fmx<short3, int3, char3, op>, device::fmx<short4, int4, char4, op>  },
                    { device::fmx<short, int, ushort, op>, device::fmx<short2, int2, ushort2, op>, device::fmx<short3, int3, ushort3, op>, device::fmx<short4, int4, ushort4, op>  },
                    { device::fmx<short, int, short, op>, device::fmx<short2, int2, short2, op>, device::fmx<short3, int3, short3, op>, device::fmx<short4, int4, short4, op>  },
                    { device::fmx<short, int, int, op>, device::fmx<short2, int2, int2, op>, device::fmx<short3, int3, int3, op>, device::fmx<short4, int4, int4, op>  },
                    { device::fmx<short, int, float, op>, device::fmx<short2, int2, float2, op>, device::fmx<short3, int3, float3, op>, device::fmx<short4, int4, float4, op>  },
                    { device::fmx<short, int, double, op>, device::fmx<short2, int2, double2, op>, device::fmx<short3, int3, double3, op>, device::fmx<short4, int4, double4, op>  },
            },
            {
                    { device::fmx<short, float, uchar, op>, device::fmx<short2, float2, uchar2, op>, device::fmx<short3, float3, uchar3, op>, device::fmx<short4, float4, uchar4, op>  },
                    { device::fmx<short, float, schar, op>, device::fmx<short2, float2, char2, op>, device::fmx<short3, float3, char3, op>, device::fmx<short4, float4, char4, op>  },
                    { device::fmx<short, float, ushort, op>, device::fmx<short2, float2, ushort2, op>, device::fmx<short3, float3, ushort3, op>, device::fmx<short4, float4, ushort4, op>  },
                    { device::fmx<short, float, short, op>, device::fmx<short2, float2, short2, op>, device::fmx<short3, float3, short3, op>, device::fmx<short4, float4, short4, op>  },
                    { device::fmx<short, float, int, op>, device::fmx<short2, float2, int2, op>, device::fmx<short3, float3, int3, op>, device::fmx<short4, float4, int4, op>  },
                    { device::fmx<short, float, float, op>, device::fmx<short2, float2, float2, op>, device::fmx<short3, float3, float3, op>, device::fmx<short4, float4, float4, op>  },
                    { device::fmx<short, float, double, op>, device::fmx<short2, float2, double2, op>, device::fmx<short3, float3, double3, op>, device::fmx<short4, float4, double4, op>  },
            },
            {
                    { device::fmx<short, double, uchar, op>, device::fmx<short2, double2, uchar2, op>, device::fmx<short3, double3, uchar3, op>, device::fmx<short4, double4, uchar4, op>  },
                    { device::fmx<short, double, schar, op>, device::fmx<short2, double2, char2, op>, device::fmx<short3, double3, char3, op>, device::fmx<short4, double4, char4, op>  },
                    { device::fmx<short, double, ushort, op>, device::fmx<short2, double2, ushort2, op>, device::fmx<short3, double3, ushort3, op>, device::fmx<short4, double4, ushort4, op>  },
                    { device::fmx<short, double, short, op>, device::fmx<short2, double2, short2, op>, device::fmx<short3, double3, short3, op>, device::fmx<short4, double4, short4, op>  },
                    { device::fmx<short, double, int, op>, device::fmx<short2, double2, int2, op>, device::fmx<short3, double3, int3, op>, device::fmx<short4, double4, int4, op>  },
                    { device::fmx<short, double, float, op>, device::fmx<short2, double2, float2, op>, device::fmx<short3, double3, float3, op>, device::fmx<short4, double4, float4, op>  },
                    { device::fmx<short, double, double, op>, device::fmx<short2, double2, double2, op>, device::fmx<short3, double3, double3, op>, device::fmx<short4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<int, uchar, uchar, op>, device::fmx<int2, uchar2, uchar2, op>, device::fmx<int3, uchar3, uchar3, op>, device::fmx<int4, uchar4, uchar4, op>  },
                    { device::fmx<int, uchar, schar, op>, device::fmx<int2, uchar2, char2, op>, device::fmx<int3, uchar3, char3, op>, device::fmx<int4, uchar4, char4, op>  },
                    { device::fmx<int, uchar, ushort, op>, device::fmx<int2, uchar2, ushort2, op>, device::fmx<int3, uchar3, ushort3, op>, device::fmx<int4, uchar4, ushort4, op>  },
                    { device::fmx<int, uchar, short, op>, device::fmx<int2, uchar2, short2, op>, device::fmx<int3, uchar3, short3, op>, device::fmx<int4, uchar4, short4, op>  },
                    { device::fmx<int, uchar, int, op>, device::fmx<int2, uchar2, int2, op>, device::fmx<int3, uchar3, int3, op>, device::fmx<int4, uchar4, int4, op>  },
                    { device::fmx<int, uchar, float, op>, device::fmx<int2, uchar2, float2, op>, device::fmx<int3, uchar3, float3, op>, device::fmx<int4, uchar4, float4, op>  },
                    { device::fmx<int, uchar, double, op>, device::fmx<int2, uchar2, double2, op>, device::fmx<int3, uchar3, double3, op>, device::fmx<int4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<int, schar, uchar, op>, device::fmx<int2, char2, uchar2, op>, device::fmx<int3, char3, uchar3, op>, device::fmx<int4, char4, uchar4, op>  },
                    { device::fmx<int, schar, schar, op>, device::fmx<int2, char2, char2, op>, device::fmx<int3, char3, char3, op>, device::fmx<int4, char4, char4, op>  },
                    { device::fmx<int, schar, ushort, op>, device::fmx<int2, char2, ushort2, op>, device::fmx<int3, char3, ushort3, op>, device::fmx<int4, char4, ushort4, op>  },
                    { device::fmx<int, schar, short, op>, device::fmx<int2, char2, short2, op>, device::fmx<int3, char3, short3, op>, device::fmx<int4, char4, short4, op>  },
                    { device::fmx<int, schar, int, op>, device::fmx<int2, char2, int2, op>, device::fmx<int3, char3, int3, op>, device::fmx<int4, char4, int4, op>  },
                    { device::fmx<int, schar, float, op>, device::fmx<int2, char2, float2, op>, device::fmx<int3, char3, float3, op>, device::fmx<int4, char4, float4, op>  },
                    { device::fmx<int, schar, double, op>, device::fmx<int2, char2, double2, op>, device::fmx<int3, char3, double3, op>, device::fmx<int4, char4, double4, op>  },
            },
            {
                    { device::fmx<int, ushort, uchar, op>, device::fmx<int2, ushort2, uchar2, op>, device::fmx<int3, ushort3, uchar3, op>, device::fmx<int4, ushort4, uchar4, op>  },
                    { device::fmx<int, ushort, schar, op>, device::fmx<int2, ushort2, char2, op>, device::fmx<int3, ushort3, char3, op>, device::fmx<int4, ushort4, char4, op>  },
                    { device::fmx<int, ushort, ushort, op>, device::fmx<int2, ushort2, ushort2, op>, device::fmx<int3, ushort3, ushort3, op>, device::fmx<int4, ushort4, ushort4, op>  },
                    { device::fmx<int, ushort, short, op>, device::fmx<int2, ushort2, short2, op>, device::fmx<int3, ushort3, short3, op>, device::fmx<int4, ushort4, short4, op>  },
                    { device::fmx<int, ushort, int, op>, device::fmx<int2, ushort2, int2, op>, device::fmx<int3, ushort3, int3, op>, device::fmx<int4, ushort4, int4, op>  },
                    { device::fmx<int, ushort, float, op>, device::fmx<int2, ushort2, float2, op>, device::fmx<int3, ushort3, float3, op>, device::fmx<int4, ushort4, float4, op>  },
                    { device::fmx<int, ushort, double, op>, device::fmx<int2, ushort2, double2, op>, device::fmx<int3, ushort3, double3, op>, device::fmx<int4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<int, short, uchar, op>, device::fmx<int2, short2, uchar2, op>, device::fmx<int3, short3, uchar3, op>, device::fmx<int4, short4, uchar4, op>  },
                    { device::fmx<int, short, schar, op>, device::fmx<int2, short2, char2, op>, device::fmx<int3, short3, char3, op>, device::fmx<int4, short4, char4, op>  },
                    { device::fmx<int, short, ushort, op>, device::fmx<int2, short2, ushort2, op>, device::fmx<int3, short3, ushort3, op>, device::fmx<int4, short4, ushort4, op>  },
                    { device::fmx<int, short, short, op>, device::fmx<int2, short2, short2, op>, device::fmx<int3, short3, short3, op>, device::fmx<int4, short4, short4, op>  },
                    { device::fmx<int, short, int, op>, device::fmx<int2, short2, int2, op>, device::fmx<int3, short3, int3, op>, device::fmx<int4, short4, int4, op>  },
                    { device::fmx<int, short, float, op>, device::fmx<int2, short2, float2, op>, device::fmx<int3, short3, float3, op>, device::fmx<int4, short4, float4, op>  },
                    { device::fmx<int, short, double, op>, device::fmx<int2, short2, double2, op>, device::fmx<int3, short3, double3, op>, device::fmx<int4, short4, double4, op>  },
            },
            {
                    { device::fmx<int, int, uchar, op>, device::fmx<int2, int2, uchar2, op>, device::fmx<int3, int3, uchar3, op>, device::fmx<int4, int4, uchar4, op>  },
                    { device::fmx<int, int, schar, op>, device::fmx<int2, int2, char2, op>, device::fmx<int3, int3, char3, op>, device::fmx<int4, int4, char4, op>  },
                    { device::fmx<int, int, ushort, op>, device::fmx<int2, int2, ushort2, op>, device::fmx<int3, int3, ushort3, op>, device::fmx<int4, int4, ushort4, op>  },
                    { device::fmx<int, int, short, op>, device::fmx<int2, int2, short2, op>, device::fmx<int3, int3, short3, op>, device::fmx<int4, int4, short4, op>  },
                    { device::fmx<int, int, int, op>, device::fmx<int2, int2, int2, op>, device::fmx<int3, int3, int3, op>, device::fmx<int4, int4, int4, op>  },
                    { device::fmx<int, int, float, op>, device::fmx<int2, int2, float2, op>, device::fmx<int3, int3, float3, op>, device::fmx<int4, int4, float4, op>  },
                    { device::fmx<int, int, double, op>, device::fmx<int2, int2, double2, op>, device::fmx<int3, int3, double3, op>, device::fmx<int4, int4, double4, op>  },
            },
            {
                    { device::fmx<int, float, uchar, op>, device::fmx<int2, float2, uchar2, op>, device::fmx<int3, float3, uchar3, op>, device::fmx<int4, float4, uchar4, op>  },
                    { device::fmx<int, float, schar, op>, device::fmx<int2, float2, char2, op>, device::fmx<int3, float3, char3, op>, device::fmx<int4, float4, char4, op>  },
                    { device::fmx<int, float, ushort, op>, device::fmx<int2, float2, ushort2, op>, device::fmx<int3, float3, ushort3, op>, device::fmx<int4, float4, ushort4, op>  },
                    { device::fmx<int, float, short, op>, device::fmx<int2, float2, short2, op>, device::fmx<int3, float3, short3, op>, device::fmx<int4, float4, short4, op>  },
                    { device::fmx<int, float, int, op>, device::fmx<int2, float2, int2, op>, device::fmx<int3, float3, int3, op>, device::fmx<int4, float4, int4, op>  },
                    { device::fmx<int, float, float, op>, device::fmx<int2, float2, float2, op>, device::fmx<int3, float3, float3, op>, device::fmx<int4, float4, float4, op>  },
                    { device::fmx<int, float, double, op>, device::fmx<int2, float2, double2, op>, device::fmx<int3, float3, double3, op>, device::fmx<int4, float4, double4, op>  },
            },
            {
                    { device::fmx<int, double, uchar, op>, device::fmx<int2, double2, uchar2, op>, device::fmx<int3, double3, uchar3, op>, device::fmx<int4, double4, uchar4, op>  },
                    { device::fmx<int, double, schar, op>, device::fmx<int2, double2, char2, op>, device::fmx<int3, double3, char3, op>, device::fmx<int4, double4, char4, op>  },
                    { device::fmx<int, double, ushort, op>, device::fmx<int2, double2, ushort2, op>, device::fmx<int3, double3, ushort3, op>, device::fmx<int4, double4, ushort4, op>  },
                    { device::fmx<int, double, short, op>, device::fmx<int2, double2, short2, op>, device::fmx<int3, double3, short3, op>, device::fmx<int4, double4, short4, op>  },
                    { device::fmx<int, double, int, op>, device::fmx<int2, double2, int2, op>, device::fmx<int3, double3, int3, op>, device::fmx<int4, double4, int4, op>  },
                    { device::fmx<int, double, float, op>, device::fmx<int2, double2, float2, op>, device::fmx<int3, double3, float3, op>, device::fmx<int4, double4, float4, op>  },
                    { device::fmx<int, double, double, op>, device::fmx<int2, double2, double2, op>, device::fmx<int3, double3, double3, op>, device::fmx<int4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<float, uchar, uchar, op>, device::fmx<float2, uchar2, uchar2, op>, device::fmx<float3, uchar3, uchar3, op>, device::fmx<float4, uchar4, uchar4, op>  },
                    { device::fmx<float, uchar, schar, op>, device::fmx<float2, uchar2, char2, op>, device::fmx<float3, uchar3, char3, op>, device::fmx<float4, uchar4, char4, op>  },
                    { device::fmx<float, uchar, ushort, op>, device::fmx<float2, uchar2, ushort2, op>, device::fmx<float3, uchar3, ushort3, op>, device::fmx<float4, uchar4, ushort4, op>  },
                    { device::fmx<float, uchar, short, op>, device::fmx<float2, uchar2, short2, op>, device::fmx<float3, uchar3, short3, op>, device::fmx<float4, uchar4, short4, op>  },
                    { device::fmx<float, uchar, int, op>, device::fmx<float2, uchar2, int2, op>, device::fmx<float3, uchar3, int3, op>, device::fmx<float4, uchar4, int4, op>  },
                    { device::fmx<float, uchar, float, op>, device::fmx<float2, uchar2, float2, op>, device::fmx<float3, uchar3, float3, op>, device::fmx<float4, uchar4, float4, op>  },
                    { device::fmx<float, uchar, double, op>, device::fmx<float2, uchar2, double2, op>, device::fmx<float3, uchar3, double3, op>, device::fmx<float4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<float, schar, uchar, op>, device::fmx<float2, char2, uchar2, op>, device::fmx<float3, char3, uchar3, op>, device::fmx<float4, char4, uchar4, op>  },
                    { device::fmx<float, schar, schar, op>, device::fmx<float2, char2, char2, op>, device::fmx<float3, char3, char3, op>, device::fmx<float4, char4, char4, op>  },
                    { device::fmx<float, schar, ushort, op>, device::fmx<float2, char2, ushort2, op>, device::fmx<float3, char3, ushort3, op>, device::fmx<float4, char4, ushort4, op>  },
                    { device::fmx<float, schar, short, op>, device::fmx<float2, char2, short2, op>, device::fmx<float3, char3, short3, op>, device::fmx<float4, char4, short4, op>  },
                    { device::fmx<float, schar, int, op>, device::fmx<float2, char2, int2, op>, device::fmx<float3, char3, int3, op>, device::fmx<float4, char4, int4, op>  },
                    { device::fmx<float, schar, float, op>, device::fmx<float2, char2, float2, op>, device::fmx<float3, char3, float3, op>, device::fmx<float4, char4, float4, op>  },
                    { device::fmx<float, schar, double, op>, device::fmx<float2, char2, double2, op>, device::fmx<float3, char3, double3, op>, device::fmx<float4, char4, double4, op>  },
            },
            {
                    { device::fmx<float, ushort, uchar, op>, device::fmx<float2, ushort2, uchar2, op>, device::fmx<float3, ushort3, uchar3, op>, device::fmx<float4, ushort4, uchar4, op>  },
                    { device::fmx<float, ushort, schar, op>, device::fmx<float2, ushort2, char2, op>, device::fmx<float3, ushort3, char3, op>, device::fmx<float4, ushort4, char4, op>  },
                    { device::fmx<float, ushort, ushort, op>, device::fmx<float2, ushort2, ushort2, op>, device::fmx<float3, ushort3, ushort3, op>, device::fmx<float4, ushort4, ushort4, op>  },
                    { device::fmx<float, ushort, short, op>, device::fmx<float2, ushort2, short2, op>, device::fmx<float3, ushort3, short3, op>, device::fmx<float4, ushort4, short4, op>  },
                    { device::fmx<float, ushort, int, op>, device::fmx<float2, ushort2, int2, op>, device::fmx<float3, ushort3, int3, op>, device::fmx<float4, ushort4, int4, op>  },
                    { device::fmx<float, ushort, float, op>, device::fmx<float2, ushort2, float2, op>, device::fmx<float3, ushort3, float3, op>, device::fmx<float4, ushort4, float4, op>  },
                    { device::fmx<float, ushort, double, op>, device::fmx<float2, ushort2, double2, op>, device::fmx<float3, ushort3, double3, op>, device::fmx<float4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<float, short, uchar, op>, device::fmx<float2, short2, uchar2, op>, device::fmx<float3, short3, uchar3, op>, device::fmx<float4, short4, uchar4, op>  },
                    { device::fmx<float, short, schar, op>, device::fmx<float2, short2, char2, op>, device::fmx<float3, short3, char3, op>, device::fmx<float4, short4, char4, op>  },
                    { device::fmx<float, short, ushort, op>, device::fmx<float2, short2, ushort2, op>, device::fmx<float3, short3, ushort3, op>, device::fmx<float4, short4, ushort4, op>  },
                    { device::fmx<float, short, short, op>, device::fmx<float2, short2, short2, op>, device::fmx<float3, short3, short3, op>, device::fmx<float4, short4, short4, op>  },
                    { device::fmx<float, short, int, op>, device::fmx<float2, short2, int2, op>, device::fmx<float3, short3, int3, op>, device::fmx<float4, short4, int4, op>  },
                    { device::fmx<float, short, float, op>, device::fmx<float2, short2, float2, op>, device::fmx<float3, short3, float3, op>, device::fmx<float4, short4, float4, op>  },
                    { device::fmx<float, short, double, op>, device::fmx<float2, short2, double2, op>, device::fmx<float3, short3, double3, op>, device::fmx<float4, short4, double4, op>  },
            },
            {
                    { device::fmx<float, int, uchar, op>, device::fmx<float2, int2, uchar2, op>, device::fmx<float3, int3, uchar3, op>, device::fmx<float4, int4, uchar4, op>  },
                    { device::fmx<float, int, schar, op>, device::fmx<float2, int2, char2, op>, device::fmx<float3, int3, char3, op>, device::fmx<float4, int4, char4, op>  },
                    { device::fmx<float, int, ushort, op>, device::fmx<float2, int2, ushort2, op>, device::fmx<float3, int3, ushort3, op>, device::fmx<float4, int4, ushort4, op>  },
                    { device::fmx<float, int, short, op>, device::fmx<float2, int2, short2, op>, device::fmx<float3, int3, short3, op>, device::fmx<float4, int4, short4, op>  },
                    { device::fmx<float, int, int, op>, device::fmx<float2, int2, int2, op>, device::fmx<float3, int3, int3, op>, device::fmx<float4, int4, int4, op>  },
                    { device::fmx<float, int, float, op>, device::fmx<float2, int2, float2, op>, device::fmx<float3, int3, float3, op>, device::fmx<float4, int4, float4, op>  },
                    { device::fmx<float, int, double, op>, device::fmx<float2, int2, double2, op>, device::fmx<float3, int3, double3, op>, device::fmx<float4, int4, double4, op>  },
            },
            {
                    { device::fmx<float, float, uchar, op>, device::fmx<float2, float2, uchar2, op>, device::fmx<float3, float3, uchar3, op>, device::fmx<float4, float4, uchar4, op>  },
                    { device::fmx<float, float, schar, op>, device::fmx<float2, float2, char2, op>, device::fmx<float3, float3, char3, op>, device::fmx<float4, float4, char4, op>  },
                    { device::fmx<float, float, ushort, op>, device::fmx<float2, float2, ushort2, op>, device::fmx<float3, float3, ushort3, op>, device::fmx<float4, float4, ushort4, op>  },
                    { device::fmx<float, float, short, op>, device::fmx<float2, float2, short2, op>, device::fmx<float3, float3, short3, op>, device::fmx<float4, float4, short4, op>  },
                    { device::fmx<float, float, int, op>, device::fmx<float2, float2, int2, op>, device::fmx<float3, float3, int3, op>, device::fmx<float4, float4, int4, op>  },
                    { device::fmx<float, float, float, op>, device::fmx<float2, float2, float2, op>, device::fmx<float3, float3, float3, op>, device::fmx<float4, float4, float4, op>  },
                    { device::fmx<float, float, double, op>, device::fmx<float2, float2, double2, op>, device::fmx<float3, float3, double3, op>, device::fmx<float4, float4, double4, op>  },
            },
            {
                    { device::fmx<float, double, uchar, op>, device::fmx<float2, double2, uchar2, op>, device::fmx<float3, double3, uchar3, op>, device::fmx<float4, double4, uchar4, op>  },
                    { device::fmx<float, double, schar, op>, device::fmx<float2, double2, char2, op>, device::fmx<float3, double3, char3, op>, device::fmx<float4, double4, char4, op>  },
                    { device::fmx<float, double, ushort, op>, device::fmx<float2, double2, ushort2, op>, device::fmx<float3, double3, ushort3, op>, device::fmx<float4, double4, ushort4, op>  },
                    { device::fmx<float, double, short, op>, device::fmx<float2, double2, short2, op>, device::fmx<float3, double3, short3, op>, device::fmx<float4, double4, short4, op>  },
                    { device::fmx<float, double, int, op>, device::fmx<float2, double2, int2, op>, device::fmx<float3, double3, int3, op>, device::fmx<float4, double4, int4, op>  },
                    { device::fmx<float, double, float, op>, device::fmx<float2, double2, float2, op>, device::fmx<float3, double3, float3, op>, device::fmx<float4, double4, float4, op>  },
                    { device::fmx<float, double, double, op>, device::fmx<float2, double2, double2, op>, device::fmx<float3, double3, double3, op>, device::fmx<float4, double4, double4, op>  },
            }
        },
        {
            {
                    { device::fmx<double, uchar, uchar, op>, device::fmx<double2, uchar2, uchar2, op>, device::fmx<double3, uchar3, uchar3, op>, device::fmx<double4, uchar4, uchar4, op>  },
                    { device::fmx<double, uchar, schar, op>, device::fmx<double2, uchar2, char2, op>, device::fmx<double3, uchar3, char3, op>, device::fmx<double4, uchar4, char4, op>  },
                    { device::fmx<double, uchar, ushort, op>, device::fmx<double2, uchar2, ushort2, op>, device::fmx<double3, uchar3, ushort3, op>, device::fmx<double4, uchar4, ushort4, op>  },
                    { device::fmx<double, uchar, short, op>, device::fmx<double2, uchar2, short2, op>, device::fmx<double3, uchar3, short3, op>, device::fmx<double4, uchar4, short4, op>  },
                    { device::fmx<double, uchar, int, op>, device::fmx<double2, uchar2, int2, op>, device::fmx<double3, uchar3, int3, op>, device::fmx<double4, uchar4, int4, op>  },
                    { device::fmx<double, uchar, float, op>, device::fmx<double2, uchar2, float2, op>, device::fmx<double3, uchar3, float3, op>, device::fmx<double4, uchar4, float4, op>  },
                    { device::fmx<double, uchar, double, op>, device::fmx<double2, uchar2, double2, op>, device::fmx<double3, uchar3, double3, op>, device::fmx<double4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<double, schar, uchar, op>, device::fmx<double2, char2, uchar2, op>, device::fmx<double3, char3, uchar3, op>, device::fmx<double4, char4, uchar4, op>  },
                    { device::fmx<double, schar, schar, op>, device::fmx<double2, char2, char2, op>, device::fmx<double3, char3, char3, op>, device::fmx<double4, char4, char4, op>  },
                    { device::fmx<double, schar, ushort, op>, device::fmx<double2, char2, ushort2, op>, device::fmx<double3, char3, ushort3, op>, device::fmx<double4, char4, ushort4, op>  },
                    { device::fmx<double, schar, short, op>, device::fmx<double2, char2, short2, op>, device::fmx<double3, char3, short3, op>, device::fmx<double4, char4, short4, op>  },
                    { device::fmx<double, schar, int, op>, device::fmx<double2, char2, int2, op>, device::fmx<double3, char3, int3, op>, device::fmx<double4, char4, int4, op>  },
                    { device::fmx<double, schar, float, op>, device::fmx<double2, char2, float2, op>, device::fmx<double3, char3, float3, op>, device::fmx<double4, char4, float4, op>  },
                    { device::fmx<double, schar, double, op>, device::fmx<double2, char2, double2, op>, device::fmx<double3, char3, double3, op>, device::fmx<double4, char4, double4, op>  },
            },
            {
                    { device::fmx<double, ushort, uchar, op>, device::fmx<double2, ushort2, uchar2, op>, device::fmx<double3, ushort3, uchar3, op>, device::fmx<double4, ushort4, uchar4, op>  },
                    { device::fmx<double, ushort, schar, op>, device::fmx<double2, ushort2, char2, op>, device::fmx<double3, ushort3, char3, op>, device::fmx<double4, ushort4, char4, op>  },
                    { device::fmx<double, ushort, ushort, op>, device::fmx<double2, ushort2, ushort2, op>, device::fmx<double3, ushort3, ushort3, op>, device::fmx<double4, ushort4, ushort4, op>  },
                    { device::fmx<double, ushort, short, op>, device::fmx<double2, ushort2, short2, op>, device::fmx<double3, ushort3, short3, op>, device::fmx<double4, ushort4, short4, op>  },
                    { device::fmx<double, ushort, int, op>, device::fmx<double2, ushort2, int2, op>, device::fmx<double3, ushort3, int3, op>, device::fmx<double4, ushort4, int4, op>  },
                    { device::fmx<double, ushort, float, op>, device::fmx<double2, ushort2, float2, op>, device::fmx<double3, ushort3, float3, op>, device::fmx<double4, ushort4, float4, op>  },
                    { device::fmx<double, ushort, double, op>, device::fmx<double2, ushort2, double2, op>, device::fmx<double3, ushort3, double3, op>, device::fmx<double4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<double, short, uchar, op>, device::fmx<double2, short2, uchar2, op>, device::fmx<double3, short3, uchar3, op>, device::fmx<double4, short4, uchar4, op>  },
                    { device::fmx<double, short, schar, op>, device::fmx<double2, short2, char2, op>, device::fmx<double3, short3, char3, op>, device::fmx<double4, short4, char4, op>  },
                    { device::fmx<double, short, ushort, op>, device::fmx<double2, short2, ushort2, op>, device::fmx<double3, short3, ushort3, op>, device::fmx<double4, short4, ushort4, op>  },
                    { device::fmx<double, short, short, op>, device::fmx<double2, short2, short2, op>, device::fmx<double3, short3, short3, op>, device::fmx<double4, short4, short4, op>  },
                    { device::fmx<double, short, int, op>, device::fmx<double2, short2, int2, op>, device::fmx<double3, short3, int3, op>, device::fmx<double4, short4, int4, op>  },
                    { device::fmx<double, short, float, op>, device::fmx<double2, short2, float2, op>, device::fmx<double3, short3, float3, op>, device::fmx<double4, short4, float4, op>  },
                    { device::fmx<double, short, double, op>, device::fmx<double2, short2, double2, op>, device::fmx<double3, short3, double3, op>, device::fmx<double4, short4, double4, op>  },
            },
            {
                    { device::fmx<double, int, uchar, op>, device::fmx<double2, int2, uchar2, op>, device::fmx<double3, int3, uchar3, op>, device::fmx<double4, int4, uchar4, op>  },
                    { device::fmx<double, int, schar, op>, device::fmx<double2, int2, char2, op>, device::fmx<double3, int3, char3, op>, device::fmx<double4, int4, char4, op>  },
                    { device::fmx<double, int, ushort, op>, device::fmx<double2, int2, ushort2, op>, device::fmx<double3, int3, ushort3, op>, device::fmx<double4, int4, ushort4, op>  },
                    { device::fmx<double, int, short, op>, device::fmx<double2, int2, short2, op>, device::fmx<double3, int3, short3, op>, device::fmx<double4, int4, short4, op>  },
                    { device::fmx<double, int, int, op>, device::fmx<double2, int2, int2, op>, device::fmx<double3, int3, int3, op>, device::fmx<double4, int4, int4, op>  },
                    { device::fmx<double, int, float, op>, device::fmx<double2, int2, float2, op>, device::fmx<double3, int3, float3, op>, device::fmx<double4, int4, float4, op>  },
                    { device::fmx<double, int, double, op>, device::fmx<double2, int2, double2, op>, device::fmx<double3, int3, double3, op>, device::fmx<double4, int4, double4, op>  },
            },
            {
                    { device::fmx<double, float, uchar, op>, device::fmx<double2, float2, uchar2, op>, device::fmx<double3, float3, uchar3, op>, device::fmx<double4, float4, uchar4, op>  },
                    { device::fmx<double, float, schar, op>, device::fmx<double2, float2, char2, op>, device::fmx<double3, float3, char3, op>, device::fmx<double4, float4, char4, op>  },
                    { device::fmx<double, float, ushort, op>, device::fmx<double2, float2, ushort2, op>, device::fmx<double3, float3, ushort3, op>, device::fmx<double4, float4, ushort4, op>  },
                    { device::fmx<double, float, short, op>, device::fmx<double2, float2, short2, op>, device::fmx<double3, float3, short3, op>, device::fmx<double4, float4, short4, op>  },
                    { device::fmx<double, float, int, op>, device::fmx<double2, float2, int2, op>, device::fmx<double3, float3, int3, op>, device::fmx<double4, float4, int4, op>  },
                    { device::fmx<double, float, float, op>, device::fmx<double2, float2, float2, op>, device::fmx<double3, float3, float3, op>, device::fmx<double4, float4, float4, op>  },
                    { device::fmx<double, float, double, op>, device::fmx<double2, float2, double2, op>, device::fmx<double3, float3, double3, op>, device::fmx<double4, float4, double4, op>  },
            },
            {
                    { device::fmx<double, double, uchar, op>, device::fmx<double2, double2, uchar2, op>, device::fmx<double3, double3, uchar3, op>, device::fmx<double4, double4, uchar4, op>  },
                    { device::fmx<double, double, schar, op>, device::fmx<double2, double2, char2, op>, device::fmx<double3, double3, char3, op>, device::fmx<double4, double4, char4, op>  },
                    { device::fmx<double, double, ushort, op>, device::fmx<double2, double2, ushort2, op>, device::fmx<double3, double3, ushort3, op>, device::fmx<double4, double4, ushort4, op>  },
                    { device::fmx<double, double, short, op>, device::fmx<double2, double2, short2, op>, device::fmx<double3, double3, short3, op>, device::fmx<double4, double4, short4, op>  },
                    { device::fmx<double, double, int, op>, device::fmx<double2, double2, int2, op>, device::fmx<double3, double3, int3, op>, device::fmx<double4, double4, int4, op>  },
                    { device::fmx<double, double, float, op>, device::fmx<double2, double2, float2, op>, device::fmx<double3, double3, float3, op>, device::fmx<double4, double4, float4, op>  },
                    { device::fmx<double, double, double, op>, device::fmx<double2, double2, double2, op>, device::fmx<double3, double3, double3, op>, device::fmx<double4, double4, double4, op>  },
            }
          }
        };

    GpuMat src1(_src1), src3(_src3);
    GpuMat& dst = _dst;



//    int stype = std::min(_src1.type(), _src3.type());
    int stype = CV_MAKETYPE(std::min(_src1.depth(), _src3.depth()), _src1.channels());
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int wdepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruction_needed(false);

    if(cn>4)
    {
        reconstruction_needed = true;

        GpuMat tmp;

        if(!src1.isContinuous())
        {
            src1.copyTo(tmp, _stream);
            src1.release();
            src1 = tmp;
        }

        tmp = src1.reshape(1);
        src1 = tmp;

        if(!src3.isContinuous())
        {
            src3.copyTo(tmp, _stream);
            src3.release();
            src3 = tmp;
        }

        tmp = src3.reshape(1);
        src3 = tmp;
    }

    dst.create(src1.size(), !reconstruction_needed ? wtype : wdepth);

    function_type fun = functions[src1.depth()][src3.depth()][dst.depth()][dst.channels()-1];

    fun(src1, _src2, src3, dst, _mask, _stream);

    if(reconstruction_needed)
    {
        GpuMat tmp;

        tmp = dst.reshape(cn);
        dst = tmp;
    }

}

template<int op>
void fmx_soaoa(const Scalar& _src1, const GpuMat& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
        CV_Assert((_src2.size() == _src3.size()) && (_mask.empty() || (_mask.size() == _src2.size())) );

        fmx_axspa<op>(_src2, _src1, _src3, _dst, _mask, dtype, _stream);
}

template<int op>
void fmx_aoaos(const GpuMat& _src1, const GpuMat& _src2, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
    CV_Assert((_src1.size() == _src2.size()) && (_mask.empty() || (_mask.size() == _src1.size())) );

    typedef void (*function_type)(const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

    static const function_type functions[7][7][7][4] =
    {
        {
            {
                    { device::fmx<uchar, uchar, uchar, op>, device::fmx<uchar2, uchar2, uchar2, op>, device::fmx<uchar3, uchar3, uchar3, op>, device::fmx<uchar4, uchar4, uchar4, op>  },
                    { device::fmx<uchar, uchar, schar, op>, device::fmx<uchar2, uchar2, char2, op>, device::fmx<uchar3, uchar3, char3, op>, device::fmx<uchar4, uchar4, char4, op>  },
                    { device::fmx<uchar, uchar, ushort, op>, device::fmx<uchar2, uchar2, ushort2, op>, device::fmx<uchar3, uchar3, ushort3, op>, device::fmx<uchar4, uchar4, ushort4, op>  },
                    { device::fmx<uchar, uchar, short, op>, device::fmx<uchar2, uchar2, short2, op>, device::fmx<uchar3, uchar3, short3, op>, device::fmx<uchar4, uchar4, short4, op>  },
                    { device::fmx<uchar, uchar, int, op>, device::fmx<uchar2, uchar2, int2, op>, device::fmx<uchar3, uchar3, int3, op>, device::fmx<uchar4, uchar4, int4, op>  },
                    { device::fmx<uchar, uchar, float, op>, device::fmx<uchar2, uchar2, float2, op>, device::fmx<uchar3, uchar3, float3, op>, device::fmx<uchar4, uchar4, float4, op>  },
                    { device::fmx<uchar, uchar, double, op>, device::fmx<uchar2, uchar2, double2, op>, device::fmx<uchar3, uchar3, double3, op>, device::fmx<uchar4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<uchar, schar, uchar, op>, device::fmx<uchar2, char2, uchar2, op>, device::fmx<uchar3, char3, uchar3, op>, device::fmx<uchar4, char4, uchar4, op>  },
                    { device::fmx<uchar, schar, schar, op>, device::fmx<uchar2, char2, char2, op>, device::fmx<uchar3, char3, char3, op>, device::fmx<uchar4, char4, char4, op>  },
                    { device::fmx<uchar, schar, ushort, op>, device::fmx<uchar2, char2, ushort2, op>, device::fmx<uchar3, char3, ushort3, op>, device::fmx<uchar4, char4, ushort4, op>  },
                    { device::fmx<uchar, schar, short, op>, device::fmx<uchar2, char2, short2, op>, device::fmx<uchar3, char3, short3, op>, device::fmx<uchar4, char4, short4, op>  },
                    { device::fmx<uchar, schar, int, op>, device::fmx<uchar2, char2, int2, op>, device::fmx<uchar3, char3, int3, op>, device::fmx<uchar4, char4, int4, op>  },
                    { device::fmx<uchar, schar, float, op>, device::fmx<uchar2, char2, float2, op>, device::fmx<uchar3, char3, float3, op>, device::fmx<uchar4, char4, float4, op>  },
                    { device::fmx<uchar, schar, double, op>, device::fmx<uchar2, char2, double2, op>, device::fmx<uchar3, char3, double3, op>, device::fmx<uchar4, char4, double4, op>  },
            },
            {
                    { device::fmx<uchar, ushort, uchar, op>, device::fmx<uchar2, ushort2, uchar2, op>, device::fmx<uchar3, ushort3, uchar3, op>, device::fmx<uchar4, ushort4, uchar4, op>  },
                    { device::fmx<uchar, ushort, schar, op>, device::fmx<uchar2, ushort2, char2, op>, device::fmx<uchar3, ushort3, char3, op>, device::fmx<uchar4, ushort4, char4, op>  },
                    { device::fmx<uchar, ushort, ushort, op>, device::fmx<uchar2, ushort2, ushort2, op>, device::fmx<uchar3, ushort3, ushort3, op>, device::fmx<uchar4, ushort4, ushort4, op>  },
                    { device::fmx<uchar, ushort, short, op>, device::fmx<uchar2, ushort2, short2, op>, device::fmx<uchar3, ushort3, short3, op>, device::fmx<uchar4, ushort4, short4, op>  },
                    { device::fmx<uchar, ushort, int, op>, device::fmx<uchar2, ushort2, int2, op>, device::fmx<uchar3, ushort3, int3, op>, device::fmx<uchar4, ushort4, int4, op>  },
                    { device::fmx<uchar, ushort, float, op>, device::fmx<uchar2, ushort2, float2, op>, device::fmx<uchar3, ushort3, float3, op>, device::fmx<uchar4, ushort4, float4, op>  },
                    { device::fmx<uchar, ushort, double, op>, device::fmx<uchar2, ushort2, double2, op>, device::fmx<uchar3, ushort3, double3, op>, device::fmx<uchar4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<uchar, short, uchar, op>, device::fmx<uchar2, short2, uchar2, op>, device::fmx<uchar3, short3, uchar3, op>, device::fmx<uchar4, short4, uchar4, op>  },
                    { device::fmx<uchar, short, schar, op>, device::fmx<uchar2, short2, char2, op>, device::fmx<uchar3, short3, char3, op>, device::fmx<uchar4, short4, char4, op>  },
                    { device::fmx<uchar, short, ushort, op>, device::fmx<uchar2, short2, ushort2, op>, device::fmx<uchar3, short3, ushort3, op>, device::fmx<uchar4, short4, ushort4, op>  },
                    { device::fmx<uchar, short, short, op>, device::fmx<uchar2, short2, short2, op>, device::fmx<uchar3, short3, short3, op>, device::fmx<uchar4, short4, short4, op>  },
                    { device::fmx<uchar, short, int, op>, device::fmx<uchar2, short2, int2, op>, device::fmx<uchar3, short3, int3, op>, device::fmx<uchar4, short4, int4, op>  },
                    { device::fmx<uchar, short, float, op>, device::fmx<uchar2, short2, float2, op>, device::fmx<uchar3, short3, float3, op>, device::fmx<uchar4, short4, float4, op>  },
                    { device::fmx<uchar, short, double, op>, device::fmx<uchar2, short2, double2, op>, device::fmx<uchar3, short3, double3, op>, device::fmx<uchar4, short4, double4, op>  },
            },
            {
                    { device::fmx<uchar, int, uchar, op>, device::fmx<uchar2, int2, uchar2, op>, device::fmx<uchar3, int3, uchar3, op>, device::fmx<uchar4, int4, uchar4, op>  },
                    { device::fmx<uchar, int, schar, op>, device::fmx<uchar2, int2, char2, op>, device::fmx<uchar3, int3, char3, op>, device::fmx<uchar4, int4, char4, op>  },
                    { device::fmx<uchar, int, ushort, op>, device::fmx<uchar2, int2, ushort2, op>, device::fmx<uchar3, int3, ushort3, op>, device::fmx<uchar4, int4, ushort4, op>  },
                    { device::fmx<uchar, int, short, op>, device::fmx<uchar2, int2, short2, op>, device::fmx<uchar3, int3, short3, op>, device::fmx<uchar4, int4, short4, op>  },
                    { device::fmx<uchar, int, int, op>, device::fmx<uchar2, int2, int2, op>, device::fmx<uchar3, int3, int3, op>, device::fmx<uchar4, int4, int4, op>  },
                    { device::fmx<uchar, int, float, op>, device::fmx<uchar2, int2, float2, op>, device::fmx<uchar3, int3, float3, op>, device::fmx<uchar4, int4, float4, op>  },
                    { device::fmx<uchar, int, double, op>, device::fmx<uchar2, int2, double2, op>, device::fmx<uchar3, int3, double3, op>, device::fmx<uchar4, int4, double4, op>  },
            },
            {
                    { device::fmx<uchar, float, uchar, op>, device::fmx<uchar2, float2, uchar2, op>, device::fmx<uchar3, float3, uchar3, op>, device::fmx<uchar4, float4, uchar4, op>  },
                    { device::fmx<uchar, float, schar, op>, device::fmx<uchar2, float2, char2, op>, device::fmx<uchar3, float3, char3, op>, device::fmx<uchar4, float4, char4, op>  },
                    { device::fmx<uchar, float, ushort, op>, device::fmx<uchar2, float2, ushort2, op>, device::fmx<uchar3, float3, ushort3, op>, device::fmx<uchar4, float4, ushort4, op>  },
                    { device::fmx<uchar, float, short, op>, device::fmx<uchar2, float2, short2, op>, device::fmx<uchar3, float3, short3, op>, device::fmx<uchar4, float4, short4, op>  },
                    { device::fmx<uchar, float, int, op>, device::fmx<uchar2, float2, int2, op>, device::fmx<uchar3, float3, int3, op>, device::fmx<uchar4, float4, int4, op>  },
                    { device::fmx<uchar, float, float, op>, device::fmx<uchar2, float2, float2, op>, device::fmx<uchar3, float3, float3, op>, device::fmx<uchar4, float4, float4, op>  },
                    { device::fmx<uchar, float, double, op>, device::fmx<uchar2, float2, double2, op>, device::fmx<uchar3, float3, double3, op>, device::fmx<uchar4, float4, double4, op>  },
            },
            {
                    { device::fmx<uchar, double, uchar, op>, device::fmx<uchar2, double2, uchar2, op>, device::fmx<uchar3, double3, uchar3, op>, device::fmx<uchar4, double4, uchar4, op>  },
                    { device::fmx<uchar, double, schar, op>, device::fmx<uchar2, double2, char2, op>, device::fmx<uchar3, double3, char3, op>, device::fmx<uchar4, double4, char4, op>  },
                    { device::fmx<uchar, double, ushort, op>, device::fmx<uchar2, double2, ushort2, op>, device::fmx<uchar3, double3, ushort3, op>, device::fmx<uchar4, double4, ushort4, op>  },
                    { device::fmx<uchar, double, short, op>, device::fmx<uchar2, double2, short2, op>, device::fmx<uchar3, double3, short3, op>, device::fmx<uchar4, double4, short4, op>  },
                    { device::fmx<uchar, double, int, op>, device::fmx<uchar2, double2, int2, op>, device::fmx<uchar3, double3, int3, op>, device::fmx<uchar4, double4, int4, op>  },
                    { device::fmx<uchar, double, float, op>, device::fmx<uchar2, double2, float2, op>, device::fmx<uchar3, double3, float3, op>, device::fmx<uchar4, double4, float4, op>  },
                    { device::fmx<uchar, double, double, op>, device::fmx<uchar2, double2, double2, op>, device::fmx<uchar3, double3, double3, op>, device::fmx<uchar4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<schar, uchar, uchar, op>, device::fmx<char2, uchar2, uchar2, op>, device::fmx<char3, uchar3, uchar3, op>, device::fmx<char4, uchar4, uchar4, op>  },
                    { device::fmx<schar, uchar, schar, op>, device::fmx<char2, uchar2, char2, op>, device::fmx<char3, uchar3, char3, op>, device::fmx<char4, uchar4, char4, op>  },
                    { device::fmx<schar, uchar, ushort, op>, device::fmx<char2, uchar2, ushort2, op>, device::fmx<char3, uchar3, ushort3, op>, device::fmx<char4, uchar4, ushort4, op>  },
                    { device::fmx<schar, uchar, short, op>, device::fmx<char2, uchar2, short2, op>, device::fmx<char3, uchar3, short3, op>, device::fmx<char4, uchar4, short4, op>  },
                    { device::fmx<schar, uchar, int, op>, device::fmx<char2, uchar2, int2, op>, device::fmx<char3, uchar3, int3, op>, device::fmx<char4, uchar4, int4, op>  },
                    { device::fmx<schar, uchar, float, op>, device::fmx<char2, uchar2, float2, op>, device::fmx<char3, uchar3, float3, op>, device::fmx<char4, uchar4, float4, op>  },
                    { device::fmx<schar, uchar, double, op>, device::fmx<char2, uchar2, double2, op>, device::fmx<char3, uchar3, double3, op>, device::fmx<char4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<schar, schar, uchar, op>, device::fmx<char2, char2, uchar2, op>, device::fmx<char3, char3, uchar3, op>, device::fmx<char4, char4, uchar4, op>  },
                    { device::fmx<schar, schar, schar, op>, device::fmx<char2, char2, char2, op>, device::fmx<char3, char3, char3, op>, device::fmx<char4, char4, char4, op>  },
                    { device::fmx<schar, schar, ushort, op>, device::fmx<char2, char2, ushort2, op>, device::fmx<char3, char3, ushort3, op>, device::fmx<char4, char4, ushort4, op>  },
                    { device::fmx<schar, schar, short, op>, device::fmx<char2, char2, short2, op>, device::fmx<char3, char3, short3, op>, device::fmx<char4, char4, short4, op>  },
                    { device::fmx<schar, schar, int, op>, device::fmx<char2, char2, int2, op>, device::fmx<char3, char3, int3, op>, device::fmx<char4, char4, int4, op>  },
                    { device::fmx<schar, schar, float, op>, device::fmx<char2, char2, float2, op>, device::fmx<char3, char3, float3, op>, device::fmx<char4, char4, float4, op>  },
                    { device::fmx<schar, schar, double, op>, device::fmx<char2, char2, double2, op>, device::fmx<char3, char3, double3, op>, device::fmx<char4, char4, double4, op>  },
            },
            {
                    { device::fmx<schar, ushort, uchar, op>, device::fmx<char2, ushort2, uchar2, op>, device::fmx<char3, ushort3, uchar3, op>, device::fmx<char4, ushort4, uchar4, op>  },
                    { device::fmx<schar, ushort, schar, op>, device::fmx<char2, ushort2, char2, op>, device::fmx<char3, ushort3, char3, op>, device::fmx<char4, ushort4, char4, op>  },
                    { device::fmx<schar, ushort, ushort, op>, device::fmx<char2, ushort2, ushort2, op>, device::fmx<char3, ushort3, ushort3, op>, device::fmx<char4, ushort4, ushort4, op>  },
                    { device::fmx<schar, ushort, short, op>, device::fmx<char2, ushort2, short2, op>, device::fmx<char3, ushort3, short3, op>, device::fmx<char4, ushort4, short4, op>  },
                    { device::fmx<schar, ushort, int, op>, device::fmx<char2, ushort2, int2, op>, device::fmx<char3, ushort3, int3, op>, device::fmx<char4, ushort4, int4, op>  },
                    { device::fmx<schar, ushort, float, op>, device::fmx<char2, ushort2, float2, op>, device::fmx<char3, ushort3, float3, op>, device::fmx<char4, ushort4, float4, op>  },
                    { device::fmx<schar, ushort, double, op>, device::fmx<char2, ushort2, double2, op>, device::fmx<char3, ushort3, double3, op>, device::fmx<char4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<schar, short, uchar, op>, device::fmx<char2, short2, uchar2, op>, device::fmx<char3, short3, uchar3, op>, device::fmx<char4, short4, uchar4, op>  },
                    { device::fmx<schar, short, schar, op>, device::fmx<char2, short2, char2, op>, device::fmx<char3, short3, char3, op>, device::fmx<char4, short4, char4, op>  },
                    { device::fmx<schar, short, ushort, op>, device::fmx<char2, short2, ushort2, op>, device::fmx<char3, short3, ushort3, op>, device::fmx<char4, short4, ushort4, op>  },
                    { device::fmx<schar, short, short, op>, device::fmx<char2, short2, short2, op>, device::fmx<char3, short3, short3, op>, device::fmx<char4, short4, short4, op>  },
                    { device::fmx<schar, short, int, op>, device::fmx<char2, short2, int2, op>, device::fmx<char3, short3, int3, op>, device::fmx<char4, short4, int4, op>  },
                    { device::fmx<schar, short, float, op>, device::fmx<char2, short2, float2, op>, device::fmx<char3, short3, float3, op>, device::fmx<char4, short4, float4, op>  },
                    { device::fmx<schar, short, double, op>, device::fmx<char2, short2, double2, op>, device::fmx<char3, short3, double3, op>, device::fmx<char4, short4, double4, op>  },
            },
            {
                    { device::fmx<schar, int, uchar, op>, device::fmx<char2, int2, uchar2, op>, device::fmx<char3, int3, uchar3, op>, device::fmx<char4, int4, uchar4, op>  },
                    { device::fmx<schar, int, schar, op>, device::fmx<char2, int2, char2, op>, device::fmx<char3, int3, char3, op>, device::fmx<char4, int4, char4, op>  },
                    { device::fmx<schar, int, ushort, op>, device::fmx<char2, int2, ushort2, op>, device::fmx<char3, int3, ushort3, op>, device::fmx<char4, int4, ushort4, op>  },
                    { device::fmx<schar, int, short, op>, device::fmx<char2, int2, short2, op>, device::fmx<char3, int3, short3, op>, device::fmx<char4, int4, short4, op>  },
                    { device::fmx<schar, int, int, op>, device::fmx<char2, int2, int2, op>, device::fmx<char3, int3, int3, op>, device::fmx<char4, int4, int4, op>  },
                    { device::fmx<schar, int, float, op>, device::fmx<char2, int2, float2, op>, device::fmx<char3, int3, float3, op>, device::fmx<char4, int4, float4, op>  },
                    { device::fmx<schar, int, double, op>, device::fmx<char2, int2, double2, op>, device::fmx<char3, int3, double3, op>, device::fmx<char4, int4, double4, op>  },
            },
            {
                    { device::fmx<schar, float, uchar, op>, device::fmx<char2, float2, uchar2, op>, device::fmx<char3, float3, uchar3, op>, device::fmx<char4, float4, uchar4, op>  },
                    { device::fmx<schar, float, schar, op>, device::fmx<char2, float2, char2, op>, device::fmx<char3, float3, char3, op>, device::fmx<char4, float4, char4, op>  },
                    { device::fmx<schar, float, ushort, op>, device::fmx<char2, float2, ushort2, op>, device::fmx<char3, float3, ushort3, op>, device::fmx<char4, float4, ushort4, op>  },
                    { device::fmx<schar, float, short, op>, device::fmx<char2, float2, short2, op>, device::fmx<char3, float3, short3, op>, device::fmx<char4, float4, short4, op>  },
                    { device::fmx<schar, float, int, op>, device::fmx<char2, float2, int2, op>, device::fmx<char3, float3, int3, op>, device::fmx<char4, float4, int4, op>  },
                    { device::fmx<schar, float, float, op>, device::fmx<char2, float2, float2, op>, device::fmx<char3, float3, float3, op>, device::fmx<char4, float4, float4, op>  },
                    { device::fmx<schar, float, double, op>, device::fmx<char2, float2, double2, op>, device::fmx<char3, float3, double3, op>, device::fmx<char4, float4, double4, op>  },
            },
            {
                    { device::fmx<schar, double, uchar, op>, device::fmx<char2, double2, uchar2, op>, device::fmx<char3, double3, uchar3, op>, device::fmx<char4, double4, uchar4, op>  },
                    { device::fmx<schar, double, schar, op>, device::fmx<char2, double2, char2, op>, device::fmx<char3, double3, char3, op>, device::fmx<char4, double4, char4, op>  },
                    { device::fmx<schar, double, ushort, op>, device::fmx<char2, double2, ushort2, op>, device::fmx<char3, double3, ushort3, op>, device::fmx<char4, double4, ushort4, op>  },
                    { device::fmx<schar, double, short, op>, device::fmx<char2, double2, short2, op>, device::fmx<char3, double3, short3, op>, device::fmx<char4, double4, short4, op>  },
                    { device::fmx<schar, double, int, op>, device::fmx<char2, double2, int2, op>, device::fmx<char3, double3, int3, op>, device::fmx<char4, double4, int4, op>  },
                    { device::fmx<schar, double, float, op>, device::fmx<char2, double2, float2, op>, device::fmx<char3, double3, float3, op>, device::fmx<char4, double4, float4, op>  },
                    { device::fmx<schar, double, double, op>, device::fmx<char2, double2, double2, op>, device::fmx<char3, double3, double3, op>, device::fmx<char4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<ushort, uchar, uchar, op>, device::fmx<ushort2, uchar2, uchar2, op>, device::fmx<ushort3, uchar3, uchar3, op>, device::fmx<ushort4, uchar4, uchar4, op>  },
                    { device::fmx<ushort, uchar, schar, op>, device::fmx<ushort2, uchar2, char2, op>, device::fmx<ushort3, uchar3, char3, op>, device::fmx<ushort4, uchar4, char4, op>  },
                    { device::fmx<ushort, uchar, ushort, op>, device::fmx<ushort2, uchar2, ushort2, op>, device::fmx<ushort3, uchar3, ushort3, op>, device::fmx<ushort4, uchar4, ushort4, op>  },
                    { device::fmx<ushort, uchar, short, op>, device::fmx<ushort2, uchar2, short2, op>, device::fmx<ushort3, uchar3, short3, op>, device::fmx<ushort4, uchar4, short4, op>  },
                    { device::fmx<ushort, uchar, int, op>, device::fmx<ushort2, uchar2, int2, op>, device::fmx<ushort3, uchar3, int3, op>, device::fmx<ushort4, uchar4, int4, op>  },
                    { device::fmx<ushort, uchar, float, op>, device::fmx<ushort2, uchar2, float2, op>, device::fmx<ushort3, uchar3, float3, op>, device::fmx<ushort4, uchar4, float4, op>  },
                    { device::fmx<ushort, uchar, double, op>, device::fmx<ushort2, uchar2, double2, op>, device::fmx<ushort3, uchar3, double3, op>, device::fmx<ushort4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<ushort, schar, uchar, op>, device::fmx<ushort2, char2, uchar2, op>, device::fmx<ushort3, char3, uchar3, op>, device::fmx<ushort4, char4, uchar4, op>  },
                    { device::fmx<ushort, schar, schar, op>, device::fmx<ushort2, char2, char2, op>, device::fmx<ushort3, char3, char3, op>, device::fmx<ushort4, char4, char4, op>  },
                    { device::fmx<ushort, schar, ushort, op>, device::fmx<ushort2, char2, ushort2, op>, device::fmx<ushort3, char3, ushort3, op>, device::fmx<ushort4, char4, ushort4, op>  },
                    { device::fmx<ushort, schar, short, op>, device::fmx<ushort2, char2, short2, op>, device::fmx<ushort3, char3, short3, op>, device::fmx<ushort4, char4, short4, op>  },
                    { device::fmx<ushort, schar, int, op>, device::fmx<ushort2, char2, int2, op>, device::fmx<ushort3, char3, int3, op>, device::fmx<ushort4, char4, int4, op>  },
                    { device::fmx<ushort, schar, float, op>, device::fmx<ushort2, char2, float2, op>, device::fmx<ushort3, char3, float3, op>, device::fmx<ushort4, char4, float4, op>  },
                    { device::fmx<ushort, schar, double, op>, device::fmx<ushort2, char2, double2, op>, device::fmx<ushort3, char3, double3, op>, device::fmx<ushort4, char4, double4, op>  },
            },
            {
                    { device::fmx<ushort, ushort, uchar, op>, device::fmx<ushort2, ushort2, uchar2, op>, device::fmx<ushort3, ushort3, uchar3, op>, device::fmx<ushort4, ushort4, uchar4, op>  },
                    { device::fmx<ushort, ushort, schar, op>, device::fmx<ushort2, ushort2, char2, op>, device::fmx<ushort3, ushort3, char3, op>, device::fmx<ushort4, ushort4, char4, op>  },
                    { device::fmx<ushort, ushort, ushort, op>, device::fmx<ushort2, ushort2, ushort2, op>, device::fmx<ushort3, ushort3, ushort3, op>, device::fmx<ushort4, ushort4, ushort4, op>  },
                    { device::fmx<ushort, ushort, short, op>, device::fmx<ushort2, ushort2, short2, op>, device::fmx<ushort3, ushort3, short3, op>, device::fmx<ushort4, ushort4, short4, op>  },
                    { device::fmx<ushort, ushort, int, op>, device::fmx<ushort2, ushort2, int2, op>, device::fmx<ushort3, ushort3, int3, op>, device::fmx<ushort4, ushort4, int4, op>  },
                    { device::fmx<ushort, ushort, float, op>, device::fmx<ushort2, ushort2, float2, op>, device::fmx<ushort3, ushort3, float3, op>, device::fmx<ushort4, ushort4, float4, op>  },
                    { device::fmx<ushort, ushort, double, op>, device::fmx<ushort2, ushort2, double2, op>, device::fmx<ushort3, ushort3, double3, op>, device::fmx<ushort4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<ushort, short, uchar, op>, device::fmx<ushort2, short2, uchar2, op>, device::fmx<ushort3, short3, uchar3, op>, device::fmx<ushort4, short4, uchar4, op>  },
                    { device::fmx<ushort, short, schar, op>, device::fmx<ushort2, short2, char2, op>, device::fmx<ushort3, short3, char3, op>, device::fmx<ushort4, short4, char4, op>  },
                    { device::fmx<ushort, short, ushort, op>, device::fmx<ushort2, short2, ushort2, op>, device::fmx<ushort3, short3, ushort3, op>, device::fmx<ushort4, short4, ushort4, op>  },
                    { device::fmx<ushort, short, short, op>, device::fmx<ushort2, short2, short2, op>, device::fmx<ushort3, short3, short3, op>, device::fmx<ushort4, short4, short4, op>  },
                    { device::fmx<ushort, short, int, op>, device::fmx<ushort2, short2, int2, op>, device::fmx<ushort3, short3, int3, op>, device::fmx<ushort4, short4, int4, op>  },
                    { device::fmx<ushort, short, float, op>, device::fmx<ushort2, short2, float2, op>, device::fmx<ushort3, short3, float3, op>, device::fmx<ushort4, short4, float4, op>  },
                    { device::fmx<ushort, short, double, op>, device::fmx<ushort2, short2, double2, op>, device::fmx<ushort3, short3, double3, op>, device::fmx<ushort4, short4, double4, op>  },
            },
            {
                    { device::fmx<ushort, int, uchar, op>, device::fmx<ushort2, int2, uchar2, op>, device::fmx<ushort3, int3, uchar3, op>, device::fmx<ushort4, int4, uchar4, op>  },
                    { device::fmx<ushort, int, schar, op>, device::fmx<ushort2, int2, char2, op>, device::fmx<ushort3, int3, char3, op>, device::fmx<ushort4, int4, char4, op>  },
                    { device::fmx<ushort, int, ushort, op>, device::fmx<ushort2, int2, ushort2, op>, device::fmx<ushort3, int3, ushort3, op>, device::fmx<ushort4, int4, ushort4, op>  },
                    { device::fmx<ushort, int, short, op>, device::fmx<ushort2, int2, short2, op>, device::fmx<ushort3, int3, short3, op>, device::fmx<ushort4, int4, short4, op>  },
                    { device::fmx<ushort, int, int, op>, device::fmx<ushort2, int2, int2, op>, device::fmx<ushort3, int3, int3, op>, device::fmx<ushort4, int4, int4, op>  },
                    { device::fmx<ushort, int, float, op>, device::fmx<ushort2, int2, float2, op>, device::fmx<ushort3, int3, float3, op>, device::fmx<ushort4, int4, float4, op>  },
                    { device::fmx<ushort, int, double, op>, device::fmx<ushort2, int2, double2, op>, device::fmx<ushort3, int3, double3, op>, device::fmx<ushort4, int4, double4, op>  },
            },
            {
                    { device::fmx<ushort, float, uchar, op>, device::fmx<ushort2, float2, uchar2, op>, device::fmx<ushort3, float3, uchar3, op>, device::fmx<ushort4, float4, uchar4, op>  },
                    { device::fmx<ushort, float, schar, op>, device::fmx<ushort2, float2, char2, op>, device::fmx<ushort3, float3, char3, op>, device::fmx<ushort4, float4, char4, op>  },
                    { device::fmx<ushort, float, ushort, op>, device::fmx<ushort2, float2, ushort2, op>, device::fmx<ushort3, float3, ushort3, op>, device::fmx<ushort4, float4, ushort4, op>  },
                    { device::fmx<ushort, float, short, op>, device::fmx<ushort2, float2, short2, op>, device::fmx<ushort3, float3, short3, op>, device::fmx<ushort4, float4, short4, op>  },
                    { device::fmx<ushort, float, int, op>, device::fmx<ushort2, float2, int2, op>, device::fmx<ushort3, float3, int3, op>, device::fmx<ushort4, float4, int4, op>  },
                    { device::fmx<ushort, float, float, op>, device::fmx<ushort2, float2, float2, op>, device::fmx<ushort3, float3, float3, op>, device::fmx<ushort4, float4, float4, op>  },
                    { device::fmx<ushort, float, double, op>, device::fmx<ushort2, float2, double2, op>, device::fmx<ushort3, float3, double3, op>, device::fmx<ushort4, float4, double4, op>  },
            },
            {
                    { device::fmx<ushort, double, uchar, op>, device::fmx<ushort2, double2, uchar2, op>, device::fmx<ushort3, double3, uchar3, op>, device::fmx<ushort4, double4, uchar4, op>  },
                    { device::fmx<ushort, double, schar, op>, device::fmx<ushort2, double2, char2, op>, device::fmx<ushort3, double3, char3, op>, device::fmx<ushort4, double4, char4, op>  },
                    { device::fmx<ushort, double, ushort, op>, device::fmx<ushort2, double2, ushort2, op>, device::fmx<ushort3, double3, ushort3, op>, device::fmx<ushort4, double4, ushort4, op>  },
                    { device::fmx<ushort, double, short, op>, device::fmx<ushort2, double2, short2, op>, device::fmx<ushort3, double3, short3, op>, device::fmx<ushort4, double4, short4, op>  },
                    { device::fmx<ushort, double, int, op>, device::fmx<ushort2, double2, int2, op>, device::fmx<ushort3, double3, int3, op>, device::fmx<ushort4, double4, int4, op>  },
                    { device::fmx<ushort, double, float, op>, device::fmx<ushort2, double2, float2, op>, device::fmx<ushort3, double3, float3, op>, device::fmx<ushort4, double4, float4, op>  },
                    { device::fmx<ushort, double, double, op>, device::fmx<ushort2, double2, double2, op>, device::fmx<ushort3, double3, double3, op>, device::fmx<ushort4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<short, uchar, uchar, op>, device::fmx<short2, uchar2, uchar2, op>, device::fmx<short3, uchar3, uchar3, op>, device::fmx<short4, uchar4, uchar4, op>  },
                    { device::fmx<short, uchar, schar, op>, device::fmx<short2, uchar2, char2, op>, device::fmx<short3, uchar3, char3, op>, device::fmx<short4, uchar4, char4, op>  },
                    { device::fmx<short, uchar, ushort, op>, device::fmx<short2, uchar2, ushort2, op>, device::fmx<short3, uchar3, ushort3, op>, device::fmx<short4, uchar4, ushort4, op>  },
                    { device::fmx<short, uchar, short, op>, device::fmx<short2, uchar2, short2, op>, device::fmx<short3, uchar3, short3, op>, device::fmx<short4, uchar4, short4, op>  },
                    { device::fmx<short, uchar, int, op>, device::fmx<short2, uchar2, int2, op>, device::fmx<short3, uchar3, int3, op>, device::fmx<short4, uchar4, int4, op>  },
                    { device::fmx<short, uchar, float, op>, device::fmx<short2, uchar2, float2, op>, device::fmx<short3, uchar3, float3, op>, device::fmx<short4, uchar4, float4, op>  },
                    { device::fmx<short, uchar, double, op>, device::fmx<short2, uchar2, double2, op>, device::fmx<short3, uchar3, double3, op>, device::fmx<short4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<short, schar, uchar, op>, device::fmx<short2, char2, uchar2, op>, device::fmx<short3, char3, uchar3, op>, device::fmx<short4, char4, uchar4, op>  },
                    { device::fmx<short, schar, schar, op>, device::fmx<short2, char2, char2, op>, device::fmx<short3, char3, char3, op>, device::fmx<short4, char4, char4, op>  },
                    { device::fmx<short, schar, ushort, op>, device::fmx<short2, char2, ushort2, op>, device::fmx<short3, char3, ushort3, op>, device::fmx<short4, char4, ushort4, op>  },
                    { device::fmx<short, schar, short, op>, device::fmx<short2, char2, short2, op>, device::fmx<short3, char3, short3, op>, device::fmx<short4, char4, short4, op>  },
                    { device::fmx<short, schar, int, op>, device::fmx<short2, char2, int2, op>, device::fmx<short3, char3, int3, op>, device::fmx<short4, char4, int4, op>  },
                    { device::fmx<short, schar, float, op>, device::fmx<short2, char2, float2, op>, device::fmx<short3, char3, float3, op>, device::fmx<short4, char4, float4, op>  },
                    { device::fmx<short, schar, double, op>, device::fmx<short2, char2, double2, op>, device::fmx<short3, char3, double3, op>, device::fmx<short4, char4, double4, op>  },
            },
            {
                    { device::fmx<short, ushort, uchar, op>, device::fmx<short2, ushort2, uchar2, op>, device::fmx<short3, ushort3, uchar3, op>, device::fmx<short4, ushort4, uchar4, op>  },
                    { device::fmx<short, ushort, schar, op>, device::fmx<short2, ushort2, char2, op>, device::fmx<short3, ushort3, char3, op>, device::fmx<short4, ushort4, char4, op>  },
                    { device::fmx<short, ushort, ushort, op>, device::fmx<short2, ushort2, ushort2, op>, device::fmx<short3, ushort3, ushort3, op>, device::fmx<short4, ushort4, ushort4, op>  },
                    { device::fmx<short, ushort, short, op>, device::fmx<short2, ushort2, short2, op>, device::fmx<short3, ushort3, short3, op>, device::fmx<short4, ushort4, short4, op>  },
                    { device::fmx<short, ushort, int, op>, device::fmx<short2, ushort2, int2, op>, device::fmx<short3, ushort3, int3, op>, device::fmx<short4, ushort4, int4, op>  },
                    { device::fmx<short, ushort, float, op>, device::fmx<short2, ushort2, float2, op>, device::fmx<short3, ushort3, float3, op>, device::fmx<short4, ushort4, float4, op>  },
                    { device::fmx<short, ushort, double, op>, device::fmx<short2, ushort2, double2, op>, device::fmx<short3, ushort3, double3, op>, device::fmx<short4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<short, short, uchar, op>, device::fmx<short2, short2, uchar2, op>, device::fmx<short3, short3, uchar3, op>, device::fmx<short4, short4, uchar4, op>  },
                    { device::fmx<short, short, schar, op>, device::fmx<short2, short2, char2, op>, device::fmx<short3, short3, char3, op>, device::fmx<short4, short4, char4, op>  },
                    { device::fmx<short, short, ushort, op>, device::fmx<short2, short2, ushort2, op>, device::fmx<short3, short3, ushort3, op>, device::fmx<short4, short4, ushort4, op>  },
                    { device::fmx<short, short, short, op>, device::fmx<short2, short2, short2, op>, device::fmx<short3, short3, short3, op>, device::fmx<short4, short4, short4, op>  },
                    { device::fmx<short, short, int, op>, device::fmx<short2, short2, int2, op>, device::fmx<short3, short3, int3, op>, device::fmx<short4, short4, int4, op>  },
                    { device::fmx<short, short, float, op>, device::fmx<short2, short2, float2, op>, device::fmx<short3, short3, float3, op>, device::fmx<short4, short4, float4, op>  },
                    { device::fmx<short, short, double, op>, device::fmx<short2, short2, double2, op>, device::fmx<short3, short3, double3, op>, device::fmx<short4, short4, double4, op>  },
            },
            {
                    { device::fmx<short, int, uchar, op>, device::fmx<short2, int2, uchar2, op>, device::fmx<short3, int3, uchar3, op>, device::fmx<short4, int4, uchar4, op>  },
                    { device::fmx<short, int, schar, op>, device::fmx<short2, int2, char2, op>, device::fmx<short3, int3, char3, op>, device::fmx<short4, int4, char4, op>  },
                    { device::fmx<short, int, ushort, op>, device::fmx<short2, int2, ushort2, op>, device::fmx<short3, int3, ushort3, op>, device::fmx<short4, int4, ushort4, op>  },
                    { device::fmx<short, int, short, op>, device::fmx<short2, int2, short2, op>, device::fmx<short3, int3, short3, op>, device::fmx<short4, int4, short4, op>  },
                    { device::fmx<short, int, int, op>, device::fmx<short2, int2, int2, op>, device::fmx<short3, int3, int3, op>, device::fmx<short4, int4, int4, op>  },
                    { device::fmx<short, int, float, op>, device::fmx<short2, int2, float2, op>, device::fmx<short3, int3, float3, op>, device::fmx<short4, int4, float4, op>  },
                    { device::fmx<short, int, double, op>, device::fmx<short2, int2, double2, op>, device::fmx<short3, int3, double3, op>, device::fmx<short4, int4, double4, op>  },
            },
            {
                    { device::fmx<short, float, uchar, op>, device::fmx<short2, float2, uchar2, op>, device::fmx<short3, float3, uchar3, op>, device::fmx<short4, float4, uchar4, op>  },
                    { device::fmx<short, float, schar, op>, device::fmx<short2, float2, char2, op>, device::fmx<short3, float3, char3, op>, device::fmx<short4, float4, char4, op>  },
                    { device::fmx<short, float, ushort, op>, device::fmx<short2, float2, ushort2, op>, device::fmx<short3, float3, ushort3, op>, device::fmx<short4, float4, ushort4, op>  },
                    { device::fmx<short, float, short, op>, device::fmx<short2, float2, short2, op>, device::fmx<short3, float3, short3, op>, device::fmx<short4, float4, short4, op>  },
                    { device::fmx<short, float, int, op>, device::fmx<short2, float2, int2, op>, device::fmx<short3, float3, int3, op>, device::fmx<short4, float4, int4, op>  },
                    { device::fmx<short, float, float, op>, device::fmx<short2, float2, float2, op>, device::fmx<short3, float3, float3, op>, device::fmx<short4, float4, float4, op>  },
                    { device::fmx<short, float, double, op>, device::fmx<short2, float2, double2, op>, device::fmx<short3, float3, double3, op>, device::fmx<short4, float4, double4, op>  },
            },
            {
                    { device::fmx<short, double, uchar, op>, device::fmx<short2, double2, uchar2, op>, device::fmx<short3, double3, uchar3, op>, device::fmx<short4, double4, uchar4, op>  },
                    { device::fmx<short, double, schar, op>, device::fmx<short2, double2, char2, op>, device::fmx<short3, double3, char3, op>, device::fmx<short4, double4, char4, op>  },
                    { device::fmx<short, double, ushort, op>, device::fmx<short2, double2, ushort2, op>, device::fmx<short3, double3, ushort3, op>, device::fmx<short4, double4, ushort4, op>  },
                    { device::fmx<short, double, short, op>, device::fmx<short2, double2, short2, op>, device::fmx<short3, double3, short3, op>, device::fmx<short4, double4, short4, op>  },
                    { device::fmx<short, double, int, op>, device::fmx<short2, double2, int2, op>, device::fmx<short3, double3, int3, op>, device::fmx<short4, double4, int4, op>  },
                    { device::fmx<short, double, float, op>, device::fmx<short2, double2, float2, op>, device::fmx<short3, double3, float3, op>, device::fmx<short4, double4, float4, op>  },
                    { device::fmx<short, double, double, op>, device::fmx<short2, double2, double2, op>, device::fmx<short3, double3, double3, op>, device::fmx<short4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<int, uchar, uchar, op>, device::fmx<int2, uchar2, uchar2, op>, device::fmx<int3, uchar3, uchar3, op>, device::fmx<int4, uchar4, uchar4, op>  },
                    { device::fmx<int, uchar, schar, op>, device::fmx<int2, uchar2, char2, op>, device::fmx<int3, uchar3, char3, op>, device::fmx<int4, uchar4, char4, op>  },
                    { device::fmx<int, uchar, ushort, op>, device::fmx<int2, uchar2, ushort2, op>, device::fmx<int3, uchar3, ushort3, op>, device::fmx<int4, uchar4, ushort4, op>  },
                    { device::fmx<int, uchar, short, op>, device::fmx<int2, uchar2, short2, op>, device::fmx<int3, uchar3, short3, op>, device::fmx<int4, uchar4, short4, op>  },
                    { device::fmx<int, uchar, int, op>, device::fmx<int2, uchar2, int2, op>, device::fmx<int3, uchar3, int3, op>, device::fmx<int4, uchar4, int4, op>  },
                    { device::fmx<int, uchar, float, op>, device::fmx<int2, uchar2, float2, op>, device::fmx<int3, uchar3, float3, op>, device::fmx<int4, uchar4, float4, op>  },
                    { device::fmx<int, uchar, double, op>, device::fmx<int2, uchar2, double2, op>, device::fmx<int3, uchar3, double3, op>, device::fmx<int4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<int, schar, uchar, op>, device::fmx<int2, char2, uchar2, op>, device::fmx<int3, char3, uchar3, op>, device::fmx<int4, char4, uchar4, op>  },
                    { device::fmx<int, schar, schar, op>, device::fmx<int2, char2, char2, op>, device::fmx<int3, char3, char3, op>, device::fmx<int4, char4, char4, op>  },
                    { device::fmx<int, schar, ushort, op>, device::fmx<int2, char2, ushort2, op>, device::fmx<int3, char3, ushort3, op>, device::fmx<int4, char4, ushort4, op>  },
                    { device::fmx<int, schar, short, op>, device::fmx<int2, char2, short2, op>, device::fmx<int3, char3, short3, op>, device::fmx<int4, char4, short4, op>  },
                    { device::fmx<int, schar, int, op>, device::fmx<int2, char2, int2, op>, device::fmx<int3, char3, int3, op>, device::fmx<int4, char4, int4, op>  },
                    { device::fmx<int, schar, float, op>, device::fmx<int2, char2, float2, op>, device::fmx<int3, char3, float3, op>, device::fmx<int4, char4, float4, op>  },
                    { device::fmx<int, schar, double, op>, device::fmx<int2, char2, double2, op>, device::fmx<int3, char3, double3, op>, device::fmx<int4, char4, double4, op>  },
            },
            {
                    { device::fmx<int, ushort, uchar, op>, device::fmx<int2, ushort2, uchar2, op>, device::fmx<int3, ushort3, uchar3, op>, device::fmx<int4, ushort4, uchar4, op>  },
                    { device::fmx<int, ushort, schar, op>, device::fmx<int2, ushort2, char2, op>, device::fmx<int3, ushort3, char3, op>, device::fmx<int4, ushort4, char4, op>  },
                    { device::fmx<int, ushort, ushort, op>, device::fmx<int2, ushort2, ushort2, op>, device::fmx<int3, ushort3, ushort3, op>, device::fmx<int4, ushort4, ushort4, op>  },
                    { device::fmx<int, ushort, short, op>, device::fmx<int2, ushort2, short2, op>, device::fmx<int3, ushort3, short3, op>, device::fmx<int4, ushort4, short4, op>  },
                    { device::fmx<int, ushort, int, op>, device::fmx<int2, ushort2, int2, op>, device::fmx<int3, ushort3, int3, op>, device::fmx<int4, ushort4, int4, op>  },
                    { device::fmx<int, ushort, float, op>, device::fmx<int2, ushort2, float2, op>, device::fmx<int3, ushort3, float3, op>, device::fmx<int4, ushort4, float4, op>  },
                    { device::fmx<int, ushort, double, op>, device::fmx<int2, ushort2, double2, op>, device::fmx<int3, ushort3, double3, op>, device::fmx<int4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<int, short, uchar, op>, device::fmx<int2, short2, uchar2, op>, device::fmx<int3, short3, uchar3, op>, device::fmx<int4, short4, uchar4, op>  },
                    { device::fmx<int, short, schar, op>, device::fmx<int2, short2, char2, op>, device::fmx<int3, short3, char3, op>, device::fmx<int4, short4, char4, op>  },
                    { device::fmx<int, short, ushort, op>, device::fmx<int2, short2, ushort2, op>, device::fmx<int3, short3, ushort3, op>, device::fmx<int4, short4, ushort4, op>  },
                    { device::fmx<int, short, short, op>, device::fmx<int2, short2, short2, op>, device::fmx<int3, short3, short3, op>, device::fmx<int4, short4, short4, op>  },
                    { device::fmx<int, short, int, op>, device::fmx<int2, short2, int2, op>, device::fmx<int3, short3, int3, op>, device::fmx<int4, short4, int4, op>  },
                    { device::fmx<int, short, float, op>, device::fmx<int2, short2, float2, op>, device::fmx<int3, short3, float3, op>, device::fmx<int4, short4, float4, op>  },
                    { device::fmx<int, short, double, op>, device::fmx<int2, short2, double2, op>, device::fmx<int3, short3, double3, op>, device::fmx<int4, short4, double4, op>  },
            },
            {
                    { device::fmx<int, int, uchar, op>, device::fmx<int2, int2, uchar2, op>, device::fmx<int3, int3, uchar3, op>, device::fmx<int4, int4, uchar4, op>  },
                    { device::fmx<int, int, schar, op>, device::fmx<int2, int2, char2, op>, device::fmx<int3, int3, char3, op>, device::fmx<int4, int4, char4, op>  },
                    { device::fmx<int, int, ushort, op>, device::fmx<int2, int2, ushort2, op>, device::fmx<int3, int3, ushort3, op>, device::fmx<int4, int4, ushort4, op>  },
                    { device::fmx<int, int, short, op>, device::fmx<int2, int2, short2, op>, device::fmx<int3, int3, short3, op>, device::fmx<int4, int4, short4, op>  },
                    { device::fmx<int, int, int, op>, device::fmx<int2, int2, int2, op>, device::fmx<int3, int3, int3, op>, device::fmx<int4, int4, int4, op>  },
                    { device::fmx<int, int, float, op>, device::fmx<int2, int2, float2, op>, device::fmx<int3, int3, float3, op>, device::fmx<int4, int4, float4, op>  },
                    { device::fmx<int, int, double, op>, device::fmx<int2, int2, double2, op>, device::fmx<int3, int3, double3, op>, device::fmx<int4, int4, double4, op>  },
            },
            {
                    { device::fmx<int, float, uchar, op>, device::fmx<int2, float2, uchar2, op>, device::fmx<int3, float3, uchar3, op>, device::fmx<int4, float4, uchar4, op>  },
                    { device::fmx<int, float, schar, op>, device::fmx<int2, float2, char2, op>, device::fmx<int3, float3, char3, op>, device::fmx<int4, float4, char4, op>  },
                    { device::fmx<int, float, ushort, op>, device::fmx<int2, float2, ushort2, op>, device::fmx<int3, float3, ushort3, op>, device::fmx<int4, float4, ushort4, op>  },
                    { device::fmx<int, float, short, op>, device::fmx<int2, float2, short2, op>, device::fmx<int3, float3, short3, op>, device::fmx<int4, float4, short4, op>  },
                    { device::fmx<int, float, int, op>, device::fmx<int2, float2, int2, op>, device::fmx<int3, float3, int3, op>, device::fmx<int4, float4, int4, op>  },
                    { device::fmx<int, float, float, op>, device::fmx<int2, float2, float2, op>, device::fmx<int3, float3, float3, op>, device::fmx<int4, float4, float4, op>  },
                    { device::fmx<int, float, double, op>, device::fmx<int2, float2, double2, op>, device::fmx<int3, float3, double3, op>, device::fmx<int4, float4, double4, op>  },
            },
            {
                    { device::fmx<int, double, uchar, op>, device::fmx<int2, double2, uchar2, op>, device::fmx<int3, double3, uchar3, op>, device::fmx<int4, double4, uchar4, op>  },
                    { device::fmx<int, double, schar, op>, device::fmx<int2, double2, char2, op>, device::fmx<int3, double3, char3, op>, device::fmx<int4, double4, char4, op>  },
                    { device::fmx<int, double, ushort, op>, device::fmx<int2, double2, ushort2, op>, device::fmx<int3, double3, ushort3, op>, device::fmx<int4, double4, ushort4, op>  },
                    { device::fmx<int, double, short, op>, device::fmx<int2, double2, short2, op>, device::fmx<int3, double3, short3, op>, device::fmx<int4, double4, short4, op>  },
                    { device::fmx<int, double, int, op>, device::fmx<int2, double2, int2, op>, device::fmx<int3, double3, int3, op>, device::fmx<int4, double4, int4, op>  },
                    { device::fmx<int, double, float, op>, device::fmx<int2, double2, float2, op>, device::fmx<int3, double3, float3, op>, device::fmx<int4, double4, float4, op>  },
                    { device::fmx<int, double, double, op>, device::fmx<int2, double2, double2, op>, device::fmx<int3, double3, double3, op>, device::fmx<int4, double4, double4, op>  },
            },
        },
        {
            {
                    { device::fmx<float, uchar, uchar, op>, device::fmx<float2, uchar2, uchar2, op>, device::fmx<float3, uchar3, uchar3, op>, device::fmx<float4, uchar4, uchar4, op>  },
                    { device::fmx<float, uchar, schar, op>, device::fmx<float2, uchar2, char2, op>, device::fmx<float3, uchar3, char3, op>, device::fmx<float4, uchar4, char4, op>  },
                    { device::fmx<float, uchar, ushort, op>, device::fmx<float2, uchar2, ushort2, op>, device::fmx<float3, uchar3, ushort3, op>, device::fmx<float4, uchar4, ushort4, op>  },
                    { device::fmx<float, uchar, short, op>, device::fmx<float2, uchar2, short2, op>, device::fmx<float3, uchar3, short3, op>, device::fmx<float4, uchar4, short4, op>  },
                    { device::fmx<float, uchar, int, op>, device::fmx<float2, uchar2, int2, op>, device::fmx<float3, uchar3, int3, op>, device::fmx<float4, uchar4, int4, op>  },
                    { device::fmx<float, uchar, float, op>, device::fmx<float2, uchar2, float2, op>, device::fmx<float3, uchar3, float3, op>, device::fmx<float4, uchar4, float4, op>  },
                    { device::fmx<float, uchar, double, op>, device::fmx<float2, uchar2, double2, op>, device::fmx<float3, uchar3, double3, op>, device::fmx<float4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<float, schar, uchar, op>, device::fmx<float2, char2, uchar2, op>, device::fmx<float3, char3, uchar3, op>, device::fmx<float4, char4, uchar4, op>  },
                    { device::fmx<float, schar, schar, op>, device::fmx<float2, char2, char2, op>, device::fmx<float3, char3, char3, op>, device::fmx<float4, char4, char4, op>  },
                    { device::fmx<float, schar, ushort, op>, device::fmx<float2, char2, ushort2, op>, device::fmx<float3, char3, ushort3, op>, device::fmx<float4, char4, ushort4, op>  },
                    { device::fmx<float, schar, short, op>, device::fmx<float2, char2, short2, op>, device::fmx<float3, char3, short3, op>, device::fmx<float4, char4, short4, op>  },
                    { device::fmx<float, schar, int, op>, device::fmx<float2, char2, int2, op>, device::fmx<float3, char3, int3, op>, device::fmx<float4, char4, int4, op>  },
                    { device::fmx<float, schar, float, op>, device::fmx<float2, char2, float2, op>, device::fmx<float3, char3, float3, op>, device::fmx<float4, char4, float4, op>  },
                    { device::fmx<float, schar, double, op>, device::fmx<float2, char2, double2, op>, device::fmx<float3, char3, double3, op>, device::fmx<float4, char4, double4, op>  },
            },
            {
                    { device::fmx<float, ushort, uchar, op>, device::fmx<float2, ushort2, uchar2, op>, device::fmx<float3, ushort3, uchar3, op>, device::fmx<float4, ushort4, uchar4, op>  },
                    { device::fmx<float, ushort, schar, op>, device::fmx<float2, ushort2, char2, op>, device::fmx<float3, ushort3, char3, op>, device::fmx<float4, ushort4, char4, op>  },
                    { device::fmx<float, ushort, ushort, op>, device::fmx<float2, ushort2, ushort2, op>, device::fmx<float3, ushort3, ushort3, op>, device::fmx<float4, ushort4, ushort4, op>  },
                    { device::fmx<float, ushort, short, op>, device::fmx<float2, ushort2, short2, op>, device::fmx<float3, ushort3, short3, op>, device::fmx<float4, ushort4, short4, op>  },
                    { device::fmx<float, ushort, int, op>, device::fmx<float2, ushort2, int2, op>, device::fmx<float3, ushort3, int3, op>, device::fmx<float4, ushort4, int4, op>  },
                    { device::fmx<float, ushort, float, op>, device::fmx<float2, ushort2, float2, op>, device::fmx<float3, ushort3, float3, op>, device::fmx<float4, ushort4, float4, op>  },
                    { device::fmx<float, ushort, double, op>, device::fmx<float2, ushort2, double2, op>, device::fmx<float3, ushort3, double3, op>, device::fmx<float4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<float, short, uchar, op>, device::fmx<float2, short2, uchar2, op>, device::fmx<float3, short3, uchar3, op>, device::fmx<float4, short4, uchar4, op>  },
                    { device::fmx<float, short, schar, op>, device::fmx<float2, short2, char2, op>, device::fmx<float3, short3, char3, op>, device::fmx<float4, short4, char4, op>  },
                    { device::fmx<float, short, ushort, op>, device::fmx<float2, short2, ushort2, op>, device::fmx<float3, short3, ushort3, op>, device::fmx<float4, short4, ushort4, op>  },
                    { device::fmx<float, short, short, op>, device::fmx<float2, short2, short2, op>, device::fmx<float3, short3, short3, op>, device::fmx<float4, short4, short4, op>  },
                    { device::fmx<float, short, int, op>, device::fmx<float2, short2, int2, op>, device::fmx<float3, short3, int3, op>, device::fmx<float4, short4, int4, op>  },
                    { device::fmx<float, short, float, op>, device::fmx<float2, short2, float2, op>, device::fmx<float3, short3, float3, op>, device::fmx<float4, short4, float4, op>  },
                    { device::fmx<float, short, double, op>, device::fmx<float2, short2, double2, op>, device::fmx<float3, short3, double3, op>, device::fmx<float4, short4, double4, op>  },
            },
            {
                    { device::fmx<float, int, uchar, op>, device::fmx<float2, int2, uchar2, op>, device::fmx<float3, int3, uchar3, op>, device::fmx<float4, int4, uchar4, op>  },
                    { device::fmx<float, int, schar, op>, device::fmx<float2, int2, char2, op>, device::fmx<float3, int3, char3, op>, device::fmx<float4, int4, char4, op>  },
                    { device::fmx<float, int, ushort, op>, device::fmx<float2, int2, ushort2, op>, device::fmx<float3, int3, ushort3, op>, device::fmx<float4, int4, ushort4, op>  },
                    { device::fmx<float, int, short, op>, device::fmx<float2, int2, short2, op>, device::fmx<float3, int3, short3, op>, device::fmx<float4, int4, short4, op>  },
                    { device::fmx<float, int, int, op>, device::fmx<float2, int2, int2, op>, device::fmx<float3, int3, int3, op>, device::fmx<float4, int4, int4, op>  },
                    { device::fmx<float, int, float, op>, device::fmx<float2, int2, float2, op>, device::fmx<float3, int3, float3, op>, device::fmx<float4, int4, float4, op>  },
                    { device::fmx<float, int, double, op>, device::fmx<float2, int2, double2, op>, device::fmx<float3, int3, double3, op>, device::fmx<float4, int4, double4, op>  },
            },
            {
                    { device::fmx<float, float, uchar, op>, device::fmx<float2, float2, uchar2, op>, device::fmx<float3, float3, uchar3, op>, device::fmx<float4, float4, uchar4, op>  },
                    { device::fmx<float, float, schar, op>, device::fmx<float2, float2, char2, op>, device::fmx<float3, float3, char3, op>, device::fmx<float4, float4, char4, op>  },
                    { device::fmx<float, float, ushort, op>, device::fmx<float2, float2, ushort2, op>, device::fmx<float3, float3, ushort3, op>, device::fmx<float4, float4, ushort4, op>  },
                    { device::fmx<float, float, short, op>, device::fmx<float2, float2, short2, op>, device::fmx<float3, float3, short3, op>, device::fmx<float4, float4, short4, op>  },
                    { device::fmx<float, float, int, op>, device::fmx<float2, float2, int2, op>, device::fmx<float3, float3, int3, op>, device::fmx<float4, float4, int4, op>  },
                    { device::fmx<float, float, float, op>, device::fmx<float2, float2, float2, op>, device::fmx<float3, float3, float3, op>, device::fmx<float4, float4, float4, op>  },
                    { device::fmx<float, float, double, op>, device::fmx<float2, float2, double2, op>, device::fmx<float3, float3, double3, op>, device::fmx<float4, float4, double4, op>  },
            },
            {
                    { device::fmx<float, double, uchar, op>, device::fmx<float2, double2, uchar2, op>, device::fmx<float3, double3, uchar3, op>, device::fmx<float4, double4, uchar4, op>  },
                    { device::fmx<float, double, schar, op>, device::fmx<float2, double2, char2, op>, device::fmx<float3, double3, char3, op>, device::fmx<float4, double4, char4, op>  },
                    { device::fmx<float, double, ushort, op>, device::fmx<float2, double2, ushort2, op>, device::fmx<float3, double3, ushort3, op>, device::fmx<float4, double4, ushort4, op>  },
                    { device::fmx<float, double, short, op>, device::fmx<float2, double2, short2, op>, device::fmx<float3, double3, short3, op>, device::fmx<float4, double4, short4, op>  },
                    { device::fmx<float, double, int, op>, device::fmx<float2, double2, int2, op>, device::fmx<float3, double3, int3, op>, device::fmx<float4, double4, int4, op>  },
                    { device::fmx<float, double, float, op>, device::fmx<float2, double2, float2, op>, device::fmx<float3, double3, float3, op>, device::fmx<float4, double4, float4, op>  },
                    { device::fmx<float, double, double, op>, device::fmx<float2, double2, double2, op>, device::fmx<float3, double3, double3, op>, device::fmx<float4, double4, double4, op>  },
            }
        },
        {
            {
                    { device::fmx<double, uchar, uchar, op>, device::fmx<double2, uchar2, uchar2, op>, device::fmx<double3, uchar3, uchar3, op>, device::fmx<double4, uchar4, uchar4, op>  },
                    { device::fmx<double, uchar, schar, op>, device::fmx<double2, uchar2, char2, op>, device::fmx<double3, uchar3, char3, op>, device::fmx<double4, uchar4, char4, op>  },
                    { device::fmx<double, uchar, ushort, op>, device::fmx<double2, uchar2, ushort2, op>, device::fmx<double3, uchar3, ushort3, op>, device::fmx<double4, uchar4, ushort4, op>  },
                    { device::fmx<double, uchar, short, op>, device::fmx<double2, uchar2, short2, op>, device::fmx<double3, uchar3, short3, op>, device::fmx<double4, uchar4, short4, op>  },
                    { device::fmx<double, uchar, int, op>, device::fmx<double2, uchar2, int2, op>, device::fmx<double3, uchar3, int3, op>, device::fmx<double4, uchar4, int4, op>  },
                    { device::fmx<double, uchar, float, op>, device::fmx<double2, uchar2, float2, op>, device::fmx<double3, uchar3, float3, op>, device::fmx<double4, uchar4, float4, op>  },
                    { device::fmx<double, uchar, double, op>, device::fmx<double2, uchar2, double2, op>, device::fmx<double3, uchar3, double3, op>, device::fmx<double4, uchar4, double4, op>  },
            },
            {
                    { device::fmx<double, schar, uchar, op>, device::fmx<double2, char2, uchar2, op>, device::fmx<double3, char3, uchar3, op>, device::fmx<double4, char4, uchar4, op>  },
                    { device::fmx<double, schar, schar, op>, device::fmx<double2, char2, char2, op>, device::fmx<double3, char3, char3, op>, device::fmx<double4, char4, char4, op>  },
                    { device::fmx<double, schar, ushort, op>, device::fmx<double2, char2, ushort2, op>, device::fmx<double3, char3, ushort3, op>, device::fmx<double4, char4, ushort4, op>  },
                    { device::fmx<double, schar, short, op>, device::fmx<double2, char2, short2, op>, device::fmx<double3, char3, short3, op>, device::fmx<double4, char4, short4, op>  },
                    { device::fmx<double, schar, int, op>, device::fmx<double2, char2, int2, op>, device::fmx<double3, char3, int3, op>, device::fmx<double4, char4, int4, op>  },
                    { device::fmx<double, schar, float, op>, device::fmx<double2, char2, float2, op>, device::fmx<double3, char3, float3, op>, device::fmx<double4, char4, float4, op>  },
                    { device::fmx<double, schar, double, op>, device::fmx<double2, char2, double2, op>, device::fmx<double3, char3, double3, op>, device::fmx<double4, char4, double4, op>  },
            },
            {
                    { device::fmx<double, ushort, uchar, op>, device::fmx<double2, ushort2, uchar2, op>, device::fmx<double3, ushort3, uchar3, op>, device::fmx<double4, ushort4, uchar4, op>  },
                    { device::fmx<double, ushort, schar, op>, device::fmx<double2, ushort2, char2, op>, device::fmx<double3, ushort3, char3, op>, device::fmx<double4, ushort4, char4, op>  },
                    { device::fmx<double, ushort, ushort, op>, device::fmx<double2, ushort2, ushort2, op>, device::fmx<double3, ushort3, ushort3, op>, device::fmx<double4, ushort4, ushort4, op>  },
                    { device::fmx<double, ushort, short, op>, device::fmx<double2, ushort2, short2, op>, device::fmx<double3, ushort3, short3, op>, device::fmx<double4, ushort4, short4, op>  },
                    { device::fmx<double, ushort, int, op>, device::fmx<double2, ushort2, int2, op>, device::fmx<double3, ushort3, int3, op>, device::fmx<double4, ushort4, int4, op>  },
                    { device::fmx<double, ushort, float, op>, device::fmx<double2, ushort2, float2, op>, device::fmx<double3, ushort3, float3, op>, device::fmx<double4, ushort4, float4, op>  },
                    { device::fmx<double, ushort, double, op>, device::fmx<double2, ushort2, double2, op>, device::fmx<double3, ushort3, double3, op>, device::fmx<double4, ushort4, double4, op>  },
            },
            {
                    { device::fmx<double, short, uchar, op>, device::fmx<double2, short2, uchar2, op>, device::fmx<double3, short3, uchar3, op>, device::fmx<double4, short4, uchar4, op>  },
                    { device::fmx<double, short, schar, op>, device::fmx<double2, short2, char2, op>, device::fmx<double3, short3, char3, op>, device::fmx<double4, short4, char4, op>  },
                    { device::fmx<double, short, ushort, op>, device::fmx<double2, short2, ushort2, op>, device::fmx<double3, short3, ushort3, op>, device::fmx<double4, short4, ushort4, op>  },
                    { device::fmx<double, short, short, op>, device::fmx<double2, short2, short2, op>, device::fmx<double3, short3, short3, op>, device::fmx<double4, short4, short4, op>  },
                    { device::fmx<double, short, int, op>, device::fmx<double2, short2, int2, op>, device::fmx<double3, short3, int3, op>, device::fmx<double4, short4, int4, op>  },
                    { device::fmx<double, short, float, op>, device::fmx<double2, short2, float2, op>, device::fmx<double3, short3, float3, op>, device::fmx<double4, short4, float4, op>  },
                    { device::fmx<double, short, double, op>, device::fmx<double2, short2, double2, op>, device::fmx<double3, short3, double3, op>, device::fmx<double4, short4, double4, op>  },
            },
            {
                    { device::fmx<double, int, uchar, op>, device::fmx<double2, int2, uchar2, op>, device::fmx<double3, int3, uchar3, op>, device::fmx<double4, int4, uchar4, op>  },
                    { device::fmx<double, int, schar, op>, device::fmx<double2, int2, char2, op>, device::fmx<double3, int3, char3, op>, device::fmx<double4, int4, char4, op>  },
                    { device::fmx<double, int, ushort, op>, device::fmx<double2, int2, ushort2, op>, device::fmx<double3, int3, ushort3, op>, device::fmx<double4, int4, ushort4, op>  },
                    { device::fmx<double, int, short, op>, device::fmx<double2, int2, short2, op>, device::fmx<double3, int3, short3, op>, device::fmx<double4, int4, short4, op>  },
                    { device::fmx<double, int, int, op>, device::fmx<double2, int2, int2, op>, device::fmx<double3, int3, int3, op>, device::fmx<double4, int4, int4, op>  },
                    { device::fmx<double, int, float, op>, device::fmx<double2, int2, float2, op>, device::fmx<double3, int3, float3, op>, device::fmx<double4, int4, float4, op>  },
                    { device::fmx<double, int, double, op>, device::fmx<double2, int2, double2, op>, device::fmx<double3, int3, double3, op>, device::fmx<double4, int4, double4, op>  },
            },
            {
                    { device::fmx<double, float, uchar, op>, device::fmx<double2, float2, uchar2, op>, device::fmx<double3, float3, uchar3, op>, device::fmx<double4, float4, uchar4, op>  },
                    { device::fmx<double, float, schar, op>, device::fmx<double2, float2, char2, op>, device::fmx<double3, float3, char3, op>, device::fmx<double4, float4, char4, op>  },
                    { device::fmx<double, float, ushort, op>, device::fmx<double2, float2, ushort2, op>, device::fmx<double3, float3, ushort3, op>, device::fmx<double4, float4, ushort4, op>  },
                    { device::fmx<double, float, short, op>, device::fmx<double2, float2, short2, op>, device::fmx<double3, float3, short3, op>, device::fmx<double4, float4, short4, op>  },
                    { device::fmx<double, float, int, op>, device::fmx<double2, float2, int2, op>, device::fmx<double3, float3, int3, op>, device::fmx<double4, float4, int4, op>  },
                    { device::fmx<double, float, float, op>, device::fmx<double2, float2, float2, op>, device::fmx<double3, float3, float3, op>, device::fmx<double4, float4, float4, op>  },
                    { device::fmx<double, float, double, op>, device::fmx<double2, float2, double2, op>, device::fmx<double3, float3, double3, op>, device::fmx<double4, float4, double4, op>  },
            },
            {
                    { device::fmx<double, double, uchar, op>, device::fmx<double2, double2, uchar2, op>, device::fmx<double3, double3, uchar3, op>, device::fmx<double4, double4, uchar4, op>  },
                    { device::fmx<double, double, schar, op>, device::fmx<double2, double2, char2, op>, device::fmx<double3, double3, char3, op>, device::fmx<double4, double4, char4, op>  },
                    { device::fmx<double, double, ushort, op>, device::fmx<double2, double2, ushort2, op>, device::fmx<double3, double3, ushort3, op>, device::fmx<double4, double4, ushort4, op>  },
                    { device::fmx<double, double, short, op>, device::fmx<double2, double2, short2, op>, device::fmx<double3, double3, short3, op>, device::fmx<double4, double4, short4, op>  },
                    { device::fmx<double, double, int, op>, device::fmx<double2, double2, int2, op>, device::fmx<double3, double3, int3, op>, device::fmx<double4, double4, int4, op>  },
                    { device::fmx<double, double, float, op>, device::fmx<double2, double2, float2, op>, device::fmx<double3, double3, float3, op>, device::fmx<double4, double4, float4, op>  },
                    { device::fmx<double, double, double, op>, device::fmx<double2, double2, double2, op>, device::fmx<double3, double3, double3, op>, device::fmx<double4, double4, double4, op>  },
            }
          }
        };

    GpuMat src1(_src1), src2(_src2);
    GpuMat& dst = _dst;



    int stype = CV_MAKETYPE(std::min(_src1.depth(), _src2.depth()), _src1.channels());
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int wdepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruction_needed(false);

    if(cn>4)
    {
        reconstruction_needed = true;

        GpuMat tmp;

        if(!src1.isContinuous())
        {
            src1.copyTo(tmp, _stream);
            src1.release();
            src1 = tmp;
        }

        tmp = src1.reshape(1);
        src1 = tmp;

        if(!src2.isContinuous())
        {
            src2.copyTo(tmp, _stream);
            src2.release();
            src2 = tmp;
        }

        tmp = src2.reshape(1);
        src2 = tmp;
    }

    dst.create(src1.size(), !reconstruction_needed ? wtype : wdepth);

    function_type fun = functions[src1.depth()][src2.depth()][dst.depth()][dst.channels()-1];

    fun(src1, src2, _src3, dst, _mask, _stream);

    if(reconstruction_needed)
    {
        GpuMat tmp;

        tmp = dst.reshape(cn);
        dst = tmp;
    }

}

template<int op>
void fmx_soaos(const Scalar& _src1, const GpuMat& _src2, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
    CV_Assert(_mask.empty() || (_mask.size() == _src2.size()));

   typedef void (*function_type)(const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

    static const function_type functions[7][7][4] =
    {
        {
            {device::fmx<uchar, uchar,  op >, device::fmx<uchar2, uchar2,  op >, device::fmx<uchar3, uchar3,  op >, device::fmx<uchar4, uchar4,  op >},
            {device::fmx<uchar, schar,  op >, device::fmx<uchar2, char2,   op >, device::fmx<uchar3, char3,   op >, device::fmx<uchar4, char4,   op >},
            {device::fmx<uchar, ushort, op >, device::fmx<uchar2, ushort2, op >, device::fmx<uchar3, ushort3, op >, device::fmx<uchar4, ushort4, op >},
            {device::fmx<uchar, short,  op >, device::fmx<uchar2, short2,  op >, device::fmx<uchar3, short3,  op >, device::fmx<uchar4, short4,  op >},
            {device::fmx<uchar, int,    op >, device::fmx<uchar2, int2,    op >, device::fmx<uchar3, int3,    op >, device::fmx<uchar4, int4,    op >},
            {device::fmx<uchar, float,  op >, device::fmx<uchar2, float2,  op >, device::fmx<uchar3, float3,  op >, device::fmx<uchar4, float4,  op >},
            {device::fmx<uchar, double, op >, device::fmx<uchar2, double2, op >, device::fmx<uchar3, double3, op >, device::fmx<uchar4, double4, op >}
        },
        {
            {device::fmx<schar, uchar,  op >, device::fmx<char2, uchar2,  op >, device::fmx<char3, uchar3,  op >, device::fmx<char4, uchar4,  op >},
            {device::fmx<schar, schar,  op >, device::fmx<char2, char2,   op >, device::fmx<char3, char3,   op >, device::fmx<char4, char4,   op >},
            {device::fmx<schar, ushort, op >, device::fmx<char2, ushort2, op >, device::fmx<char3, ushort3, op >, device::fmx<char4, ushort4, op >},
            {device::fmx<schar, short,  op >, device::fmx<char2, short2,  op >, device::fmx<char3, short3,  op >, device::fmx<char4, short4,  op >},
            {device::fmx<schar, int,    op >, device::fmx<char2, int2,    op >, device::fmx<char3, int3,    op >, device::fmx<char4, int4,    op >},
            {device::fmx<schar, float,  op >, device::fmx<char2, float2,  op >, device::fmx<char3, float3,  op >, device::fmx<char4, float4,  op >},
            {device::fmx<schar, double, op >, device::fmx<char2, double2, op >, device::fmx<char3, double3, op >, device::fmx<char4, double4, op >}
        },
        {
            {device::fmx<ushort, uchar,  op >, device::fmx<ushort2, uchar2,  op >, device::fmx<ushort3, uchar3,  op >, device::fmx<ushort4, uchar4,  op >},
            {device::fmx<ushort, schar,  op >, device::fmx<ushort2, char2,   op >, device::fmx<ushort3, char3,   op >, device::fmx<ushort4, char4,   op >},
            {device::fmx<ushort, ushort, op >, device::fmx<ushort2, ushort2, op >, device::fmx<ushort3, ushort3, op >, device::fmx<ushort4, ushort4, op >},
            {device::fmx<ushort, short,  op >, device::fmx<ushort2, short2,  op >, device::fmx<ushort3, short3,  op >, device::fmx<ushort4, short4,  op >},
            {device::fmx<ushort, int,    op >, device::fmx<ushort2, int2,    op >, device::fmx<ushort3, int3,    op >, device::fmx<ushort4, int4,    op >},
            {device::fmx<ushort, float,  op >, device::fmx<ushort2, float2,  op >, device::fmx<ushort3, float3,  op >, device::fmx<ushort4, float4,  op >},
            {device::fmx<ushort, double, op >, device::fmx<ushort2, double2, op >, device::fmx<ushort3, double3, op >, device::fmx<ushort4, double4, op >}
        },
        {
            {device::fmx<short, uchar,  op >, device::fmx<short2, uchar2,  op >, device::fmx<short3, uchar3,  op >, device::fmx<short4, uchar4,  op >},
            {device::fmx<short, schar,  op >, device::fmx<short2, char2,   op >, device::fmx<short3, char3,   op >, device::fmx<short4, char4,   op >},
            {device::fmx<short, ushort, op >, device::fmx<short2, ushort2, op >, device::fmx<short3, ushort3, op >, device::fmx<short4, ushort4, op >},
            {device::fmx<short, short,  op >, device::fmx<short2, short2,  op >, device::fmx<short3, short3,  op >, device::fmx<short4, short4,  op >},
            {device::fmx<short, int,    op >, device::fmx<short2, int2,    op >, device::fmx<short3, int3,    op >, device::fmx<short4, int4,    op >},
            {device::fmx<short, float,  op >, device::fmx<short2, float2,  op >, device::fmx<short3, float3,  op >, device::fmx<short4, float4,  op >},
            {device::fmx<short, double, op >, device::fmx<short2, double2, op >, device::fmx<short3, double3, op >, device::fmx<short4, double4, op >}
        },
        {
            {device::fmx<int, uchar,  op >, device::fmx<int2, uchar2,  op >, device::fmx<int3, uchar3,  op >, device::fmx<int4, uchar4,  op >},
            {device::fmx<int, schar,  op >, device::fmx<int2, char2,   op >, device::fmx<int3, char3,   op >, device::fmx<int4, char4,   op >},
            {device::fmx<int, ushort, op >, device::fmx<int2, ushort2, op >, device::fmx<int3, ushort3, op >, device::fmx<int4, ushort4, op >},
            {device::fmx<int, short,  op >, device::fmx<int2, short2,  op >, device::fmx<int3, short3,  op >, device::fmx<int4, short4,  op >},
            {device::fmx<int, int,    op >, device::fmx<int2, int2,    op >, device::fmx<int3, int3,    op >, device::fmx<int4, int4,    op >},
            {device::fmx<int, float,  op >, device::fmx<int2, float2,  op >, device::fmx<int3, float3,  op >, device::fmx<int4, float4,  op >},
            {device::fmx<int, double, op >, device::fmx<int2, double2, op >, device::fmx<int3, double3, op >, device::fmx<int4, double4, op >}
        },
        {
            {device::fmx<float, uchar,  op >, device::fmx<float2, uchar2,  op >, device::fmx<float3, uchar3,  op >, device::fmx<float4, uchar4,  op >},
            {device::fmx<float, schar,  op >, device::fmx<float2, char2,   op >, device::fmx<float3, char3,   op >, device::fmx<float4, char4,   op >},
            {device::fmx<float, ushort, op >, device::fmx<float2, ushort2, op >, device::fmx<float3, ushort3, op >, device::fmx<float4, ushort4, op >},
            {device::fmx<float, short,  op >, device::fmx<float2, short2,  op >, device::fmx<float3, short3,  op >, device::fmx<float4, short4,  op >},
            {device::fmx<float, int,    op >, device::fmx<float2, int2,    op >, device::fmx<float3, int3,    op >, device::fmx<float4, int4,    op >},
            {device::fmx<float, float,  op >, device::fmx<float2, float2,  op >, device::fmx<float3, float3,  op >, device::fmx<float4, float4,  op >},
            {device::fmx<float, double, op >, device::fmx<float2, double2, op >, device::fmx<float3, double3, op >, device::fmx<float4, double4, op >}
        },
        {
            {device::fmx<double, uchar,  op >, device::fmx<double2, uchar2,  op >, device::fmx<double3, uchar3,  op >, device::fmx<double4, uchar4,  op >},
            {device::fmx<double, schar,  op >, device::fmx<double2, char2,   op >, device::fmx<double3, char3,   op >, device::fmx<double4, char4,   op >},
            {device::fmx<double, ushort, op >, device::fmx<double2, ushort2, op >, device::fmx<double3, ushort3, op >, device::fmx<double4, ushort4, op >},
            {device::fmx<double, short,  op >, device::fmx<double2, short2,  op >, device::fmx<double3, short3,  op >, device::fmx<double4, short4,  op >},
            {device::fmx<double, int,    op >, device::fmx<double2, int2,    op >, device::fmx<double3, int3,    op >, device::fmx<double4, int4,    op >},
            {device::fmx<double, float,  op >, device::fmx<double2, float2,  op >, device::fmx<double3, float3,  op >, device::fmx<double4, float4,  op >},
            {device::fmx<double, double, op >, device::fmx<double2, double2, op >, device::fmx<double3, double3, op >, device::fmx<double4, double4, op >}
        }
    };


    GpuMat src(_src2);
    GpuMat& dst = _dst;


    int stype = src.type();
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int wdepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruction_needed(false);

    if(cn>4)
    {
        reconstruction_needed = true;

        GpuMat tmp;

        if(!src.isContinuous())
        {
            src.copyTo(tmp, _stream);
            src.release();
            src = tmp;
        }

        tmp = src.reshape(1);
        src = tmp;
    }

    dst.create(src.size(), !reconstruction_needed ? wtype : wdepth);

    function_type fun = functions[src.depth()][dst.depth()][dst.channels()-1];

    fun(_src1, src, _src3, dst, _mask, _stream);

    if(reconstruction_needed)
    {
        GpuMat tmp;

        tmp = dst.reshape(cn);
        dst = tmp;
    }
}

template<int op>
void fmx_aosos(const GpuMat& _src1, const Scalar& _src2, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
    CV_Assert(_mask.empty() || (_mask.size() == _src1.size()));


    fmx_sxaps<op>(_src2, _src1, _src3, _dst, _mask, dtype, _stream);
}

template<int op>
void fmx_sosoa(const Scalar& _src1, const Scalar& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{
    CV_Assert(_mask.empty() || (_mask.size() == _src3.size()));

   typedef void (*function_type)(const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

    static const function_type functions[7][7][4] =
    {
        {
            {device::fmx<uchar, uchar,  op >, device::fmx<uchar2, uchar2,  op >, device::fmx<uchar3, uchar3,  op >, device::fmx<uchar4, uchar4,  op >},
            {device::fmx<uchar, schar,  op >, device::fmx<uchar2, char2,   op >, device::fmx<uchar3, char3,   op >, device::fmx<uchar4, char4,   op >},
            {device::fmx<uchar, ushort, op >, device::fmx<uchar2, ushort2, op >, device::fmx<uchar3, ushort3, op >, device::fmx<uchar4, ushort4, op >},
            {device::fmx<uchar, short,  op >, device::fmx<uchar2, short2,  op >, device::fmx<uchar3, short3,  op >, device::fmx<uchar4, short4,  op >},
            {device::fmx<uchar, int,    op >, device::fmx<uchar2, int2,    op >, device::fmx<uchar3, int3,    op >, device::fmx<uchar4, int4,    op >},
            {device::fmx<uchar, float,  op >, device::fmx<uchar2, float2,  op >, device::fmx<uchar3, float3,  op >, device::fmx<uchar4, float4,  op >},
            {device::fmx<uchar, double, op >, device::fmx<uchar2, double2, op >, device::fmx<uchar3, double3, op >, device::fmx<uchar4, double4, op >}
        },
        {
            {device::fmx<schar, uchar,  op >, device::fmx<char2, uchar2,  op >, device::fmx<char3, uchar3,  op >, device::fmx<char4, uchar4,  op >},
            {device::fmx<schar, schar,  op >, device::fmx<char2, char2,   op >, device::fmx<char3, char3,   op >, device::fmx<char4, char4,   op >},
            {device::fmx<schar, ushort, op >, device::fmx<char2, ushort2, op >, device::fmx<char3, ushort3, op >, device::fmx<char4, ushort4, op >},
            {device::fmx<schar, short,  op >, device::fmx<char2, short2,  op >, device::fmx<char3, short3,  op >, device::fmx<char4, short4,  op >},
            {device::fmx<schar, int,    op >, device::fmx<char2, int2,    op >, device::fmx<char3, int3,    op >, device::fmx<char4, int4,    op >},
            {device::fmx<schar, float,  op >, device::fmx<char2, float2,  op >, device::fmx<char3, float3,  op >, device::fmx<char4, float4,  op >},
            {device::fmx<schar, double, op >, device::fmx<char2, double2, op >, device::fmx<char3, double3, op >, device::fmx<char4, double4, op >}
        },
        {
            {device::fmx<ushort, uchar,  op >, device::fmx<ushort2, uchar2,  op >, device::fmx<ushort3, uchar3,  op >, device::fmx<ushort4, uchar4,  op >},
            {device::fmx<ushort, schar,  op >, device::fmx<ushort2, char2,   op >, device::fmx<ushort3, char3,   op >, device::fmx<ushort4, char4,   op >},
            {device::fmx<ushort, ushort, op >, device::fmx<ushort2, ushort2, op >, device::fmx<ushort3, ushort3, op >, device::fmx<ushort4, ushort4, op >},
            {device::fmx<ushort, short,  op >, device::fmx<ushort2, short2,  op >, device::fmx<ushort3, short3,  op >, device::fmx<ushort4, short4,  op >},
            {device::fmx<ushort, int,    op >, device::fmx<ushort2, int2,    op >, device::fmx<ushort3, int3,    op >, device::fmx<ushort4, int4,    op >},
            {device::fmx<ushort, float,  op >, device::fmx<ushort2, float2,  op >, device::fmx<ushort3, float3,  op >, device::fmx<ushort4, float4,  op >},
            {device::fmx<ushort, double, op >, device::fmx<ushort2, double2, op >, device::fmx<ushort3, double3, op >, device::fmx<ushort4, double4, op >}
        },
        {
            {device::fmx<short, uchar,  op >, device::fmx<short2, uchar2,  op >, device::fmx<short3, uchar3,  op >, device::fmx<short4, uchar4,  op >},
            {device::fmx<short, schar,  op >, device::fmx<short2, char2,   op >, device::fmx<short3, char3,   op >, device::fmx<short4, char4,   op >},
            {device::fmx<short, ushort, op >, device::fmx<short2, ushort2, op >, device::fmx<short3, ushort3, op >, device::fmx<short4, ushort4, op >},
            {device::fmx<short, short,  op >, device::fmx<short2, short2,  op >, device::fmx<short3, short3,  op >, device::fmx<short4, short4,  op >},
            {device::fmx<short, int,    op >, device::fmx<short2, int2,    op >, device::fmx<short3, int3,    op >, device::fmx<short4, int4,    op >},
            {device::fmx<short, float,  op >, device::fmx<short2, float2,  op >, device::fmx<short3, float3,  op >, device::fmx<short4, float4,  op >},
            {device::fmx<short, double, op >, device::fmx<short2, double2, op >, device::fmx<short3, double3, op >, device::fmx<short4, double4, op >}
        },
        {
            {device::fmx<int, uchar,  op >, device::fmx<int2, uchar2,  op >, device::fmx<int3, uchar3,  op >, device::fmx<int4, uchar4,  op >},
            {device::fmx<int, schar,  op >, device::fmx<int2, char2,   op >, device::fmx<int3, char3,   op >, device::fmx<int4, char4,   op >},
            {device::fmx<int, ushort, op >, device::fmx<int2, ushort2, op >, device::fmx<int3, ushort3, op >, device::fmx<int4, ushort4, op >},
            {device::fmx<int, short,  op >, device::fmx<int2, short2,  op >, device::fmx<int3, short3,  op >, device::fmx<int4, short4,  op >},
            {device::fmx<int, int,    op >, device::fmx<int2, int2,    op >, device::fmx<int3, int3,    op >, device::fmx<int4, int4,    op >},
            {device::fmx<int, float,  op >, device::fmx<int2, float2,  op >, device::fmx<int3, float3,  op >, device::fmx<int4, float4,  op >},
            {device::fmx<int, double, op >, device::fmx<int2, double2, op >, device::fmx<int3, double3, op >, device::fmx<int4, double4, op >}
        },
        {
            {device::fmx<float, uchar,  op >, device::fmx<float2, uchar2,  op >, device::fmx<float3, uchar3,  op >, device::fmx<float4, uchar4,  op >},
            {device::fmx<float, schar,  op >, device::fmx<float2, char2,   op >, device::fmx<float3, char3,   op >, device::fmx<float4, char4,   op >},
            {device::fmx<float, ushort, op >, device::fmx<float2, ushort2, op >, device::fmx<float3, ushort3, op >, device::fmx<float4, ushort4, op >},
            {device::fmx<float, short,  op >, device::fmx<float2, short2,  op >, device::fmx<float3, short3,  op >, device::fmx<float4, short4,  op >},
            {device::fmx<float, int,    op >, device::fmx<float2, int2,    op >, device::fmx<float3, int3,    op >, device::fmx<float4, int4,    op >},
            {device::fmx<float, float,  op >, device::fmx<float2, float2,  op >, device::fmx<float3, float3,  op >, device::fmx<float4, float4,  op >},
            {device::fmx<float, double, op >, device::fmx<float2, double2, op >, device::fmx<float3, double3, op >, device::fmx<float4, double4, op >}
        },
        {
            {device::fmx<double, uchar,  op >, device::fmx<double2, uchar2,  op >, device::fmx<double3, uchar3,  op >, device::fmx<double4, uchar4,  op >},
            {device::fmx<double, schar,  op >, device::fmx<double2, char2,   op >, device::fmx<double3, char3,   op >, device::fmx<double4, char4,   op >},
            {device::fmx<double, ushort, op >, device::fmx<double2, ushort2, op >, device::fmx<double3, ushort3, op >, device::fmx<double4, ushort4, op >},
            {device::fmx<double, short,  op >, device::fmx<double2, short2,  op >, device::fmx<double3, short3,  op >, device::fmx<double4, short4,  op >},
            {device::fmx<double, int,    op >, device::fmx<double2, int2,    op >, device::fmx<double3, int3,    op >, device::fmx<double4, int4,    op >},
            {device::fmx<double, float,  op >, device::fmx<double2, float2,  op >, device::fmx<double3, float3,  op >, device::fmx<double4, float4,  op >},
            {device::fmx<double, double, op >, device::fmx<double2, double2, op >, device::fmx<double3, double3, op >, device::fmx<double4, double4, op >}
        }
    };


    GpuMat src(_src3);
    GpuMat& dst = _dst;


    int stype = src.type();
    int sdepth = CV_MAT_DEPTH(stype);
    int cn = CV_MAT_CN(stype);
    int wdepth = dtype == -1 ? sdepth : CV_MAT_DEPTH(dtype);
    int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruction_needed(false);

    if(cn>4)
    {
        reconstruction_needed = true;

        GpuMat tmp;

        if(!src.isContinuous())
        {
            src.copyTo(tmp, _stream);
            src.release();
            src = tmp;
        }

        tmp = src.reshape(1);
        src = tmp;
    }

    dst.create(src.size(), !reconstruction_needed ? wtype : wdepth);

    function_type fun = functions[src.depth()][dst.depth()][dst.channels()-1];

    fun(_src1, _src2, src, dst, _mask, _stream);

    if(reconstruction_needed)
    {
        GpuMat tmp;

        tmp = dst.reshape(cn);
        dst = tmp;
    }
}


template<int op>
void fmx_(InputArray& _src1, InputArray& _src2, InputArray& _src3, OutputArray& _dst, InputArray& _mask, int dtype, Stream& _stream )
{

    enum
    {
      SOSOS,
      SOSOA=0x1, // Scalar OP1 Scalar OP2 Array
      SOAOS=0xA,
      SOAOA=0xB,
      AOSOS=0x64,
      AOSOA=0x65,
      AOAOS=0x6E,
      AOAOA=0x6F
    };


    CV_Assert_5(_src1.isGpuMat() || isScalar(_src1),
                _src2.isGpuMat() || isScalar(_src2),
                _src3.isGpuMat() || isScalar(_src3),
                _mask.empty() || (_mask.isGpuMat() && (_mask.type() == CV_8UC1)),
                _dst.isGpuMat()
                );
    GpuMat dst;

    int flag = (_src1.isGpuMat() ? 100 : 0) + (_src2.isGpuMat() ? 10 : 0) + (_src3.isGpuMat() ? 1 : 0) ;

    GpuMat mask;

    if(!_mask.empty())
        mask = _mask.getGpuMat();

    if(!flag) // i.e. flag == SOSOS
        CV_Error(Error::StsBadArg,"At least one the input arguments must be a matrix.");

    switch (flag)
    {
//    case SOSOS:
//        fmx_sosos_caller<op>(getScalar(_src1), getScalar(_src2), getScalar(_src3), mask, _stream);
//        break;

    case SOSOA:
        fmx_sosoa<op>(getScalar(_src1), getScalar(_src2), _src3.getGpuMat(), _dst.getGpuMatRef(), mask, dtype, _stream);
        break;

    case SOAOS:
        fmx_soaos<op>(getScalar(_src1), _src2.getGpuMat(), getScalar(_src3), _dst.getGpuMatRef(), mask, dtype, _stream);
        break;

    case SOAOA:
        fmx_soaoa<op>(getScalar(_src1), _src2.getGpuMat(), _src3.getGpuMat(), _dst.getGpuMatRef(), mask, dtype, _stream);
        break;

    case AOSOS:
        fmx_aosoa<op>(_src1.getGpuMat(), getScalar(_src2), _src3.getGpuMat(), _dst.getGpuMatRef(), mask, dtype, _stream);
        break;

    case AOSOA:
        fmx_aosoa<op>(_src1.getGpuMat(), getScalar(_src2), _src3.getGpuMat(), _dst.getGpuMatRef(), mask, dtype, _stream);
        break;

    case AOAOA:
        fmx_aoaoa<op>(_src1.getGpuMat(), _src2.getGpuMat(), _src3.getGpuMat(), _dst.getGpuMatRef(), mask, dtype, _stream);
        break;
    }
}

}//anonymous

void fma(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    fmx_<0>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}

void fms(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream&_stream)
{
    fmx_<1>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}

void nfma(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream&_stream)
{
    fmx_<2>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}

void nfms(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream&_stream)
{
    fmx_<3>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}

void fda(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream&_stream)
{
    fmx_<4>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}

void fds(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream&_stream)
{
    fmx_<5>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}

void nfda(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream&_stream)
{
    fmx_<6>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}

void nfds(InputArray _src1, InputArray _src2, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream&_stream)
{
    fmx_<7>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
}



namespace device
{

//template<class DstType, int op>
//void wfmx(const Scalar&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class SrcType5,class DstType, int op>
void wfmx(const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class SrcType5,class DstType, int op>
void wfmx(const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class SrcType5,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class SrcType5,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class SrcType5,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class SrcType5,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const Scalar&, GpuMat&, const GpuMat&, Stream&);

template<class SrcType1,class SrcType2,class SrcType3,class SrcType4,class SrcType5,class SrcType6,class DstType, int op>
void wfmx(const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);


} // device

namespace
{


#define IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2(fun_name, fun_call, type1, type2, type3, type4)\
    template<int op> void wfmx_ ## fun_name ## _ss(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oss <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _sa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## osa <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _as(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oas <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _aa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oaa <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}

#define IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_3_4(fun_name, fun_call, type1, type2, type3, type4)\
    template<int op> void wfmx_ ## fun_name ## _ss(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oss <op>(_w1, _src1, _src2, _w2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _sa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## osa <op>(_w1, _src1, _src2, _w2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _as(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oas <op>(_w1, _src1, _src2, _w2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _aa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oaa <op>(_w1, _src1, _src2, _w2, _w3, _src3, _dst, _mask, _stream);}

#define IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__2_1(fun_name, fun_call, type1, type2, type3, type4)\
    template<int op> void wfmx_ ## fun_name ## _ss(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oss <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _sa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## osa <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _as(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oas <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _aa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oaa <op>(_src1, _w1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);}

#define IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__3_4(fun_name, fun_call, type1, type2, type3, type4)\
    template<int op> void wfmx_ ## fun_name ## _ss(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oss <op>(_w2, _src2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _sa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## osa <op>(_w2, _src2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _as(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oas <op>(_w2, _src2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _aa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oaa <op>(_w2, _src2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}

#define IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__4_3(fun_name, fun_call, type1, type2, type3, type4)\
    template<int op> void wfmx_ ## fun_name ## _ss(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oss <op>(_src2, _w2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _sa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## osa <op>(_src2, _w2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _as(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oas <op>(_src2, _w2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _aa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oaa <op>(_src2, _w2, _w1, _src1, _w3, _src3, _dst, _mask, _stream);}

#define IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_2_1__3_4(fun_name, fun_call, type1, type2, type3, type4)\
    template<int op> void wfmx_ ## fun_name ## _ss(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oss <op>(_w2, _src2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _sa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## osa <op>(_w2, _src2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _as(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oas <op>(_w2, _src2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _aa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oaa <op>(_w2, _src2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}

#define IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_2_1__4_3(fun_name, fun_call, type1, type2, type3, type4)\
    template<int op> void wfmx_ ## fun_name ## _ss(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oss <op>(_src2, _w2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _sa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## osa <op>(_src2, _w2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _as(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oas <op>(_src2, _w2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}\
    template<int op> void wfmx_ ## fun_name ## _aa(const type1& _w1, const type2& _src1, const type3& _w2, const type4& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream) { wfmx_ ## fun_call ## oaa <op>(_src2, _w2, _src1, _w1, _w3, _src3, _dst, _mask, _stream);}

// SSOSA

template<int op>
void wfmx_ssossosa(const Scalar& _w1, const Scalar& _src1, const Scalar& _w2, const Scalar& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
inline void wfmx_ssossoas(const Scalar& _w1, const Scalar& _src1, const Scalar& _w2, const Scalar& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{
       wfmx_ssossosa<op>(_w1, _src1, _w2, _src2, _src3, _w3, _dst, _mask, _stream);
}

template<int op>
void wfmx_ssossoaa(const Scalar& _w1, const Scalar& _src1, const Scalar& _w2, const Scalar& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

// AAOAA
template<int op>
void wfmx_aaoaaoss(const GpuMat& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
void wfmx_aaoaaoas(const GpuMat& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
inline void wfmx_aaoaaosa(const GpuMat& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{
    wfmx_aaoaaoas<op>(_w1, _src1, _w2, _src2, _src3, _w3, _dst, _mask, _stream);
}

template<int op>
void wfmx_aaoaaoaa(const GpuMat& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}


// SSOSA
template<int op>
void wfmx_ssosaoss(const Scalar& _w1, const Scalar& _src1, const Scalar& _w2, const GpuMat& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
void wfmx_ssosaosa(const Scalar& _w1, const Scalar& _src1, const Scalar& _w2, const GpuMat& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
inline void wfmx_ssosaoas(const Scalar& _w1, const Scalar& _src1, const Scalar& _w2, const GpuMat& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{
    wfmx_ssosaosa<op>(_w1, _src1, _w2, _src2, _src3, _w3, _dst, _mask, _stream);
}

template<int op>
void wfmx_ssosaoaa(const Scalar& _w1, const Scalar& _src1, const Scalar& _w2, const GpuMat& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

// SSOAS

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_3_4(ssoas, ssosa, Scalar, Scalar, Scalar, GpuMat)

// SAOSS

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__3_4(saoss, ssosa, Scalar, GpuMat, Scalar, Scalar)

// ASOSS

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__4_3(asoss, ssosa, GpuMat, Scalar, Scalar, Scalar)

// SAOAA

template<int op>
void wfmx_saoaaoss(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
void wfmx_saoaaosa(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
inline void wfmx_saoaaoas(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{
    wfmx_saoaaosa<op>(_w1, _src1, _w2, _src2, _src3, _w3, _dst, _mask, _stream);
}

template<int op>
void wfmx_saoaaoaa(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

// AAOSA

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__3_4(aaosa, saoaa, GpuMat, GpuMat, Scalar, GpuMat)

//ASOAA

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__2_1(asoaa, saoaa, GpuMat, Scalar, GpuMat, GpuMat)

// AAOAS

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_2_1__3_4(aaoas, saoaa, GpuMat, GpuMat, Scalar, GpuMat)

// SSOAA

template<int op>
void wfmx_ssoaaoss(const Scalar& _w1, const Scalar& _src1, const GpuMat& _w2, const GpuMat& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
void wfmx_ssoaaosa(const Scalar& _w1, const Scalar& _src1, const GpuMat& _w2, const GpuMat& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
inline void wfmx_ssoaaoas(const Scalar& _w1, const Scalar& _src1, const GpuMat& _w2, const GpuMat& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{
    wfmx_ssoaaosa<op>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);
}

template<int op>
void wfmx_ssoaaoaa(const Scalar& _w1, const Scalar& _src1, const GpuMat& _w2, const GpuMat& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}


// AAOSS

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_1_2__3_4(aaoss, ssoaa, GpuMat, GpuMat, Scalar, Scalar)

// SAOAS

template<int op>
void wfmx_saoasoss(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const Scalar& _src2, const Scalar& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
void wfmx_saoasosa(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const Scalar& _src2, const Scalar& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}

template<int op>
inline void wfmx_saoasoas(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const GpuMat& _src2, const GpuMat& _w3, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{
    wfmx_saoasosa<op>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, _stream);
}

template<int op>
void wfmx_saoasoaa(const Scalar& _w1, const GpuMat& _src1, const GpuMat& _w2, const Scalar& _src2, const GpuMat& _w3, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, Stream& _stream)
{

}


// ASOSA

IMPL_CALL_PREVIOUS_FUN__SWAP_ARGS_2_1__3_4(asosa,saoas,GpuMat, Scalar, Scalar, GpuMat)


template<int op>
void wfmx_(InputArray& _w1, InputArray& _src1, InputArray& _w2, InputArray& _src2, InputArray& _w3, InputArray& _src3, OutputArray& _dst, InputArray& _mask, int dtype, Stream& _stream)
{
    enum
    {
        SSOSSOSS,
        SSOSSOSA,
        SSOSSOAS=0xa,
        SSOSSOAA=0xb,
        SSOSAOSS=0x64,
        SSOSAOSA=0x65,
        SSOSAOAS=0x6e,
        SSOSAOAA=0x6f,
        SSOASOSS=0x3e8,
        SSOASOSA=0x3e9,
        SSOASOAS=0x3f2,
        SSOASOAA=0x3f3,
        SSOAAOSS=0x44c,
        SSOAAOSA=0x44d,
        SSOAAOAS=0x456,
        SSOAAOAA=0x457,
        SAOSSOSS=0x2710,
        SAOSSOSA=0x2711,
        SAOSSOAS=0x271a,
        SAOSSOAA=0x271b,
        SAOSAOSS=0x2774,
        SAOSAOSA=0x2775,
        SAOSAOAS=0x277e,
        SAOSAOAA=0x277f,
        SAOASOSS=0x2af8,
        SAOASOSA=0x2af9,
        SAOASOAS=0x2b02,
        SAOASOAA=0x2b03,
        SAOAAOSS=0x2b5c,
        SAOAAOSA=0x2b5d,
        SAOAAOAS=0x2b66,
        SAOAAOAA=0x2b67,
        ASOSSOSS=0x186a0,
        ASOSSOSA=0x186a1,
        ASOSSOAS=0x186aa,
        ASOSSOAA=0x186ab,
        ASOSAOSS=0x18704,
        ASOSAOSA=0x18705,
        ASOSAOAS=0x1870e,
        ASOSAOAA=0x1870f,
        ASOASOSS=0x18a88,
        ASOASOSA=0x18a89,
        ASOASOAS=0x18a92,
        ASOASOAA=0x18a93,
        ASOAAOSS=0x18aec,
        ASOAAOSA=0x18aed,
        ASOAAOAS=0x18af6,
        ASOAAOAA=0x18af7,
        AAOSSOSS=0x1adb0,
        AAOSSOSA=0x1adb1,
        AAOSSOAS=0x1adba,
        AAOSSOAA=0x1adbb,
        AAOSAOSS=0x1ae14,
        AAOSAOSA=0x1ae15,
        AAOSAOAS=0x1ae1e,
        AAOSAOAA=0x1ae1f,
        AAOASOSS=0x1b198,
        AAOASOSA=0x1b199,
        AAOASOAS=0x1b1a2,
        AAOASOAA=0x1b1a3,
        AAOAAOSS=0x1b1fc,
        AAOAAOSA=0x1b1fd,
        AAOAAOAS=0x1b206,
        AAOAAOAA=0x1b207
    };

    CV_Assert_7(_w1.empty() || isScalar(_w1) || _w1.isGpuMat(),
                _w2.empty() || isScalar(_w2) || _w2.isGpuMat(),
                _w3.empty() || isScalar(_w3) || _w3.isGpuMat(),
                isScalar(_src1) || _src1.isGpuMat(),
                isScalar(_src2) || _src2.isGpuMat(),
                isScalar(_src3) || _src3.isGpuMat(),
                _mask.empty() || (_mask.type() == CV_8UC1)
                );

    if(_w1.empty() && _w2.empty() && _w3.empty())
    {
        fmx_<op>(_src1, _src2, _src3, _dst, _mask, dtype, _stream);
        return;
    }

    int flag = (_w1.isGpuMat() ? 100000 : 0) + (_src1.isGpuMat() ? 10000 : 0) + (_w2.isGpuMat() ? 1000 : 0) + (_src2.isGpuMat() ? 100 : 0) + (_w3.isGpuMat() ? 10 : 0) + (_src3.isGpuMat() ? 1 : 0);

    if(!flag) // i.e. flag == SSOSSOSS
        CV_Error(Error::StsBadArg,"At least one the input arguments must be a matrix.");

    switch(flag)
    {
    case SSOSSOSA:
        wfmx_ssossosa(getScalar(_w1), getScalar())
        break;
    }
}

} // anonymous


void wfma(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<0>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}

void wfms(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<1>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}

void wnfma(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<2>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}

void wnfms(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<3>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}

void wfda(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<4>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}

void wfds(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<5>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}

void wnfda(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<6>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}

void wnfds(InputArray _w1, InputArray _src1, InputArray _w2, InputArray _src2, InputArray _w3, InputArray _src3, OutputArray _dst, InputArray _mask, int dtype, Stream& _stream)
{
    wfmx_<7>(_w1, _src1, _w2, _src2, _w3, _src3, _dst, _mask, dtype, _stream);
}


#endif // if 0

#endif // HAVE_CUDA

} // cuda

} // cv
