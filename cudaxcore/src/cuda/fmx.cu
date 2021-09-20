#if 0
#include "../precomp.hpp"

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/cudev.hpp"

#include "cudev.functional.ext.hpp"
#include "cudev.transform.ext.hpp"

using namespace cv::cudev;

namespace cv
{

namespace cuda
{

namespace device
{

namespace
{

__device__ __forceinline__ float fmsf(const float& a, const float& b, const float& c){ return ::fmaf(a, b, -c);}
__device__ __forceinline__ double fms(const double& a, const double& b, const double& c){ return ::fma(a, b, -c);}

__device__ __forceinline__ float nfmaf(const float& a, const float& b, const float& c){ return ::fmaf(-a, b, c);}
__device__ __forceinline__ double nfma(const double& a, const double& b, const double& c){ return ::fma(-a, b, c);}

__device__ __forceinline__ float nfmsf(const float& a, const float& b, const float& c){ return ::fmaf(-a, b, -c);}
__device__ __forceinline__ double nfms(const double& a, const double& b, const double& c){ return ::fma(-a, b, -c);}


__device__ __forceinline__ float fda_rnf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rn(b), c);}
__device__ __forceinline__ float fda_rzf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rz(b), c);}
__device__ __forceinline__ float fda_ruf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_ru(b), c);}
__device__ __forceinline__ float fda_rdf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rd(b), c);}
__device__ __forceinline__ float fda_apf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::fdividef(1.f, b), c);}

__device__ __forceinline__ double fda_rn(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_rn(b), c);}
__device__ __forceinline__ double fda_rz(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_rz(b), c);}
__device__ __forceinline__ double fda_ru(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_ru(b), c);}
__device__ __forceinline__ double fda_rd(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_rd(b), c);}
__device__ __forceinline__ double fda_ap(const double& a, const double& b, const double& c){ return ::fma(a, 1./ b, c);}

__device__ __forceinline__ float fds_rnf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rn(b), -c);}
__device__ __forceinline__ float fds_rzf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rz(b), -c);}
__device__ __forceinline__ float fds_ruf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_ru(b), -c);}
__device__ __forceinline__ float fds_rdf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rd(b), -c);}
__device__ __forceinline__ float fds_apf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::fdividef(1.f, b), -c);}

__device__ __forceinline__ double fds_rn(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_rn(b), -c);}
__device__ __forceinline__ double fds_rz(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_rz(b), -c);}
__device__ __forceinline__ double fds_ru(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_ru(b), -c);}
__device__ __forceinline__ double fds_rd(const double& a, const double& b, const double& c){ return ::fma(a, ::__drcp_rd(b), -c);}
__device__ __forceinline__ double fds_ap(const double& a, const double& b, const double& c){ return ::fma(a, 1./ b, -c);}

__device__ __forceinline__ float nfda_rnf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rn(b), c);}
__device__ __forceinline__ float nfda_rzf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rz(b), c);}
__device__ __forceinline__ float nfda_ruf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_ru(b), c);}
__device__ __forceinline__ float nfda_rdf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rd(b), c);}
__device__ __forceinline__ float nfda_apf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::fdividef(1.f, b), c);}

__device__ __forceinline__ double nfda_rn(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_rn(b), c);}
__device__ __forceinline__ double nfda_rz(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_rz(b), c);}
__device__ __forceinline__ double nfda_ru(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_ru(b), c);}
__device__ __forceinline__ double nfda_rd(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_rd(b), c);}
__device__ __forceinline__ double nfda_ap(const double& a, const double& b, const double& c){ return ::fma(-a, 1./ b, c);}

__device__ __forceinline__ float nfds_rnf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rn(b), -c);}
__device__ __forceinline__ float nfds_rzf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rz(b), -c);}
__device__ __forceinline__ float nfds_ruf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_ru(b), -c);}
__device__ __forceinline__ float nfds_rdf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rd(b), -c);}
__device__ __forceinline__ float nfds_apf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::fdividef(1.f, b), -c);}

__device__ __forceinline__ double nfds_rn(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_rn(b), -c);}
__device__ __forceinline__ double nfds_rz(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_rz(b), -c);}
__device__ __forceinline__ double nfds_ru(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_ru(b), -c);}
__device__ __forceinline__ double nfds_rd(const double& a, const double& b, const double& c){ return ::fma(-a, ::__drcp_rd(b), -c);}
__device__ __forceinline__ double nfds_ap(const double& a, const double& b, const double& c){ return ::fma(-a, 1./ b, -c);}




__device__ __forceinline__ float frsa_rnf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frsqrt_rn(b), c);}
__device__ __forceinline__ float frsa_rzf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rz(::__fsqrt_rz(b)), c);}
__device__ __forceinline__ float frsa_ruf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_ru(::__fsqrt_ru(b)), c);}
__device__ __forceinline__ float frsa_rdf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rd(::__fsqrt_rd(b)), c);}
__device__ __forceinline__ float frsa_apf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::rsqrtf(b), c);}


__device__ __forceinline__ double frsa_rn(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_rn(1.,::__dsqrt_rn(b)), c);}
__device__ __forceinline__ double frsa_rz(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_rz(1.,::__dsqrt_rz(b)), c);}
__device__ __forceinline__ double frsa_ru(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_ru(1.,::__dsqrt_ru(b)), c);}
__device__ __forceinline__ double frsa_rd(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_rd(1.,::__dsqrt_rd(b)), c);}
__device__ __forceinline__ double frsa_ap(const double& a, const double& b, const double& c){ return ::fma(a, ::rsqrt(b)), c);}




__device__ __forceinline__ float frss_rnf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frsqrt_rn(b), -c);}
__device__ __forceinline__ float frss_rzf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rz(::__fsqrt_rz(b)), -c);}
__device__ __forceinline__ float frss_ruf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_ru(::__fsqrt_ru(b)), -c);}
__device__ __forceinline__ float frss_rdf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::__frcp_rd(::__fsqrt_rd(b)), -c);}
__device__ __forceinline__ float frss_apf(const float& a, const float& b, const float& c){ return ::fmaf(a, ::rsqrtf(b), -c);}


__device__ __forceinline__ double frss_rn(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_rn(1.,::__dsqrt_rn(b)), -c);}
__device__ __forceinline__ double frss_rz(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_rz(1.,::__dsqrt_rz(b)), -c);}
__device__ __forceinline__ double frss_ru(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_ru(1.,::__dsqrt_ru(b)), -c);}
__device__ __forceinline__ double frss_rd(const double& a, const double& b, const double& c){ return ::fma(a, ::__ddiv_rd(1.,::__dsqrt_rd(b)), -c);}
__device__ __forceinline__ double frss_ap(const double& a, const double& b, const double& c){ return ::fma(a, ::rsqrt(b)), -c);}




__device__ __forceinline__ float nfrsa_rnf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frsqrt_rn(b), c);}
__device__ __forceinline__ float nfrsa_rzf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rz(::__fsqrt_rz(b)), c);}
__device__ __forceinline__ float nfrsa_ruf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_ru(::__fsqrt_ru(b)), c);}
__device__ __forceinline__ float nfrsa_rdf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rd(::__fsqrt_rd(b)), c);}
__device__ __forceinline__ float nfrsa_apf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::rsqrtf(b), c);}


__device__ __forceinline__ double nfrsa_rn(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_rn(1.,::__dsqrt_rn(b)), c);}
__device__ __forceinline__ double nfrsa_rz(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_rz(1.,::__dsqrt_rz(b)), c);}
__device__ __forceinline__ double nfrsa_ru(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_ru(1.,::__dsqrt_ru(b)), c);}
__device__ __forceinline__ double nfrsa_rd(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_rd(1.,::__dsqrt_rd(b)), c);}
__device__ __forceinline__ double nfrsa_ap(const double& a, const double& b, const double& c){ return ::fma(-a, ::rsqrt(b)), c);}




__device__ __forceinline__ float nfrss_rnf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frsqrt_rn(b), c);}
__device__ __forceinline__ float nfrss_rzf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rz(::__fsqrt_rz(b)), c);}
__device__ __forceinline__ float nfrss_ruf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_ru(::__fsqrt_ru(b)), c);}
__device__ __forceinline__ float nfrss_rdf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::__frcp_rd(::__fsqrt_rd(b)), c);}
__device__ __forceinline__ float nfrss_apf(const float& a, const float& b, const float& c){ return ::fmaf(-a, ::rsqrtf(b), c);}


__device__ __forceinline__ double nfrss_rn(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_rn(1.,::__dsqrt_rn(b)), c);}
__device__ __forceinline__ double nfrss_rz(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_rz(1.,::__dsqrt_rz(b)), c);}
__device__ __forceinline__ double nfrss_ru(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_ru(1.,::__dsqrt_ru(b)), c);}
__device__ __forceinline__ double nfrss_rd(const double& a, const double& b, const double& c){ return ::fma(-a, ::__ddiv_rd(1.,::__dsqrt_rd(b)), c);}
__device__ __forceinline__ double nfrss_ap(const double& a, const double& b, const double& c){ return ::fma(-a, ::rsqrt(b)), c);}



#define IMPL_FUNCS_TERNARY(name1, name2)\
    __device__ __forceinline__ float name1(const float& a, const float& b, const float& c){ return name2 ## f (a,b,c);} \
    __device__ __forceinline__ float2 name1(const float2& a, const float2& b, const float2& c){ return make_float2(name2 ## f(a.x, b.x, c.x), name2 ## f(a.y, b.y, c.y) );} \
    __device__ __forceinline__ float3 name1(const float3& a, const float3& b, const float3& c){ return make_float3(name2 ## f(a.x, b.x, c.x), name2 ## f(a.y, b.y, c.y), name2 ## f(a.z, b.z, c.z) );} \
    __device__ __forceinline__ float4 name1(const float4& a, const float4& b, const float4& c){ return make_float4(name2 ## f(a.x, b.x, c.x), name2 ## f(a.y, b.y, c.y), name2 ## f(a.z, b.z, c.z), name2 ## f(a.w, b.w, c.w) );} \
\
    __device__ __forceinline__ double name1(const double& a, const double& b, const double& c){ return name2(a,b,c);} \
    __device__ __forceinline__ double2 name1(const double2& a, const double2& b, const double2& c){ return make_double2(name2(a.x, b.x, c.x), name2(a.y, b.y, c.y) );} \
    __device__ __forceinline__ double3 name1(const double3& a, const double3& b, const double3& c){ return make_double3(name2(a.x, b.x, c.x), name2(a.y, b.y, c.y), name2(a.z, b.z, c.z) );} \
    __device__ __forceinline__ double4 name1(const double4& a, const double4& b, const double4& c){ return make_double4(name2(a.x, b.x, c.x), name2(a.y, b.y, c.y), name2(a.z, b.z, c.z), name2(a.w, b.w, c.w) );}

IMPL_FUNCS_TERNARY(fused_multiply_add, fma)
IMPL_FUNCS_TERNARY(fused_multiply_sub, fms)
IMPL_FUNCS_TERNARY(negative_fused_multiply_add, nfma)
IMPL_FUNCS_TERNARY(negative_fused_multiply_sub, nfms)

IMPL_FUNCS_TERNARY(fused_divide_add_round_near_even, fda_rn)
IMPL_FUNCS_TERNARY(fused_divide_add_round_near_zero, fda_rz)
IMPL_FUNCS_TERNARY(fused_divide_add_round_up, fda_ru)
IMPL_FUNCS_TERNARY(fused_divide_add_round_down, fda_rd)
IMPL_FUNCS_TERNARY(fused_divide_add_approx, fda_ap)

IMPL_FUNCS_TERNARY(fused_divide_sub_round_near_even, fds_rn)
IMPL_FUNCS_TERNARY(fused_divide_sub_round_near_zero, fds_rz)
IMPL_FUNCS_TERNARY(fused_divide_sub_round_up, fds_ru)
IMPL_FUNCS_TERNARY(fused_divide_sub_round_down, fds_rd)
IMPL_FUNCS_TERNARY(fused_divide_sub_approx, fds_ap)

IMPL_FUNCS_TERNARY(negative_fused_divide_add_round_near_even, fda_rn)
IMPL_FUNCS_TERNARY(negative_fused_divide_add_round_near_zero, fda_rz)
IMPL_FUNCS_TERNARY(negative_fused_divide_add_round_up, fda_ru)
IMPL_FUNCS_TERNARY(negative_fused_divide_add_round_down, fda_rd)
IMPL_FUNCS_TERNARY(negative_fused_divide_add_approx, fda_ap)

IMPL_FUNCS_TERNARY(negative_fused_divide_sub_round_near_even, fds_rn)
IMPL_FUNCS_TERNARY(negative_fused_divide_sub_round_near_zero, fds_rz)
IMPL_FUNCS_TERNARY(negative_fused_divide_sub_round_up, fds_ru)
IMPL_FUNCS_TERNARY(negative_fused_divide_sub_round_down, fds_rd)
IMPL_FUNCS_TERNARY(negative_fused_divide_sub_approx, fds_ap)


IMPL_FUNCS_TERNARY(fused_reciprocal_square_add_round_near_even, frsa_rn)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_add_round_near_zero, frsa_rz)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_add_round_down, frsa_rd)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_add_round_up, frsa_ru)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_add_approximate, frsa_ap)

IMPL_FUNCS_TERNARY(fused_reciprocal_square_sub_round_near_even, frss_rn)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_sub_round_near_zero, frss_rz)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_sub_round_down, frss_rd)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_sub_round_up, frss_ru)
IMPL_FUNCS_TERNARY(fused_reciprocal_square_sub_approximate, frss_ap)

IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_add_round_near_even, nfrsa_rn)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_add_round_near_zero, nfrsa_rz)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_add_round_down, nfrsa_rd)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_add_round_up, nfrsa_ru)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_add_approximate, nfrsa_ap)

IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_sub_round_near_even, frss_rn)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_sub_round_near_zero, frss_rz)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_sub_round_down, frss_rd)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_sub_round_up, frss_ru)
IMPL_FUNCS_TERNARY(negative_fused_reciprocal_square_sub_approximate, frss_ap)


#undef IMPL_FUNCS_TERNARY

template<class SrcType, class WrkType, class DstType, int op>
struct FMXOP_sosoa : unary_function<SrcType, DstType>{};

#define SPEC_FMXOP_SOSOA(fun, value)\
\
template<class SrcType, class WrkType, class DstType>\
struct FMXOP_sosoa<SrcType, WrkType, DstType, value> : unary_function<SrcType, DstType>\
{\
    WrkType _s1, _s2;\
\
    __device__ __forceinline__ DstType operator()(const SrcType& v)const\
    {\
        WrkType s3 = saturate_cast<WrkType>(v);\
\
        WrkType dst = fun(this->_s1, this->_s2, s3);\
\
        return saturate_cast<DstType>(dst);\
    }    \
};

SPEC_FMXOP_SOSOA(fused_multiply_add, 0)
SPEC_FMXOP_SOSOA(fused_multiply_sub, 1)
SPEC_FMXOP_SOSOA(negative_fused_multiply_add, 2)
SPEC_FMXOP_SOSOA(negative_fused_multiply_sub, 3)
SPEC_FMXOP_SOSOA(fused_divide_add_round_near_even, 4)
SPEC_FMXOP_SOSOA(fused_divide_add_round_near_zero, 5)
SPEC_FMXOP_SOSOA(fused_divide_add_round_up, 6)
SPEC_FMXOP_SOSOA(fused_divide_add_round_down, 7)
SPEC_FMXOP_SOSOA(fused_divide_add_approximate, 8)
SPEC_FMXOP_SOSOA(fused_divide_sub_round_near_even, 9)
SPEC_FMXOP_SOSOA(fused_divide_sub_round_near_zero, 10)
SPEC_FMXOP_SOSOA(fused_divide_sub_round_up, 11)
SPEC_FMXOP_SOSOA(fused_divide_sub_round_down, 12)
SPEC_FMXOP_SOSOA(fused_divide_sub_approximate, 13)
SPEC_FMXOP_SOSOA(negative_fused_divide_add_round_near_even, 14)
SPEC_FMXOP_SOSOA(negative_fused_divide_add_round_near_zero, 15)
SPEC_FMXOP_SOSOA(negative_fused_divide_add_round_up, 16)
SPEC_FMXOP_SOSOA(negative_fused_divide_add_round_down, 17)
SPEC_FMXOP_SOSOA(negative_fused_divide_add_approximate, 18)
SPEC_FMXOP_SOSOA(negative_fused_divide_sub_round_near_even, 19)
SPEC_FMXOP_SOSOA(negative_fused_divide_sub_round_near_zero, 20)
SPEC_FMXOP_SOSOA(negative_fused_divide_sub_round_up, 21)
SPEC_FMXOP_SOSOA(negative_fused_divide_sub_round_down, 22)
SPEC_FMXOP_SOSOA(negative_fused_divide_sub_approximate, 23)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_add_round_near_even, 24)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_add_round_near_zero, 25)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_add_round_up, 26)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_add_round_down, 27)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_add_approximate, 28)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_sub_round_near_even, 29)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_sub_round_near_zero, 30)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_sub_round_up, 31)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_sub_round_down, 32)
SPEC_FMXOP_SOSOA(fused_reciprocal_square_sub_approximate, 33)
SPEC_FMXOP_SOSOA(negative_negative_fused_reciprocal_square_add_round_near_even, 34)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_add_round_near_zero, 35)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_add_round_up, 36)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_add_round_down, 37)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_add_approximate, 38)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_sub_round_near_even, 39)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_sub_round_near_zero, 40)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_sub_round_up, 41)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_sub_round_down, 42)
SPEC_FMXOP_SOSOA(negative_fused_reciprocal_square_sub_approximate, 43)

#undef SPEC_FMXOP_SOSOA


//////////////////////////////////////////////////////////////////////////////

template<class SrcType, class WrkType, class DstType, int op>
struct FMXOP_soaos : unary_function<SrcType, DstType>{};

#define SPEC_FMXOP_SOAOS(name, value)\
    template<class SrcType, class WrkType, class DstType>\
    struct FMXOP_soaos<SrcType, WrkType, DstType, value> : unary_function<SrcType, DstType>\
    {\
        WrkType _s1, _s3;\
    \
        __device__ __forceinline__ DstType operator()(const SrcType& v)const\
        {\
            WrkType s2 = saturate_cast<WrkType>(v);\
    \
            WrkType dst = name(this->_s1, s2, this->_s3);\
    \
            return saturate_cast<DstType>(dst);\
        }\
    \
    };

SPEC_FMXOP_SOAOS(fused_multiply_add, 0)
SPEC_FMXOP_SOAOS(fused_multiply_sub, 1)
SPEC_FMXOP_SOAOS(negative_fused_multiply_add, 2)
SPEC_FMXOP_SOAOS(negative_fused_multiply_sub, 3)
SPEC_FMXOP_SOAOS(fused_divide_add_round_near_even, 4)
SPEC_FMXOP_SOAOS(fused_divide_add_round_near_zero, 5)
SPEC_FMXOP_SOAOS(fused_divide_add_round_up, 6)
SPEC_FMXOP_SOAOS(fused_divide_add_round_down, 7)
SPEC_FMXOP_SOAOS(fused_divide_add_approximate, 8)
SPEC_FMXOP_SOAOS(fused_divide_sub_round_near_even, 9)
SPEC_FMXOP_SOAOS(fused_divide_sub_round_near_zero, 10)
SPEC_FMXOP_SOAOS(fused_divide_sub_round_up, 11)
SPEC_FMXOP_SOAOS(fused_divide_sub_round_down, 12)
SPEC_FMXOP_SOAOS(fused_divide_sub_approximate, 13)
SPEC_FMXOP_SOAOS(negative_fused_divide_add_round_near_even, 14)
SPEC_FMXOP_SOAOS(negative_fused_divide_add_round_near_zero, 15)
SPEC_FMXOP_SOAOS(negative_fused_divide_add_round_up, 16)
SPEC_FMXOP_SOAOS(negative_fused_divide_add_round_down, 17)
SPEC_FMXOP_SOAOS(negative_fused_divide_add_approximate, 18)
SPEC_FMXOP_SOAOS(negative_fused_divide_sub_round_near_even, 19)
SPEC_FMXOP_SOAOS(negative_fused_divide_sub_round_near_zero, 20)
SPEC_FMXOP_SOAOS(negative_fused_divide_sub_round_up, 21)
SPEC_FMXOP_SOAOS(negative_fused_divide_sub_round_down, 22)
SPEC_FMXOP_SOAOS(negative_fused_divide_sub_approximate, 23)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_add_round_near_even, 24)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_add_round_near_zero, 25)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_add_round_up, 26)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_add_round_down, 27)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_add_approximate, 28)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_sub_round_near_even, 29)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_sub_round_near_zero, 30)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_sub_round_up, 31)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_sub_round_down, 32)
SPEC_FMXOP_SOAOS(fused_reciprocal_square_sub_approximate, 33)
SPEC_FMXOP_SOAOS(negative_negative_fused_reciprocal_square_add_round_near_even, 34)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_add_round_near_zero, 35)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_add_round_up, 36)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_add_round_down, 37)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_add_approximate, 38)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_sub_round_near_even, 39)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_sub_round_near_zero, 40)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_sub_round_up, 41)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_sub_round_down, 42)
SPEC_FMXOP_SOAOS(negative_fused_reciprocal_square_sub_approximate, 43)

#undef SPEC_FMXOP_SOAOS

//////////////////////////////////////////////////////////////////////////////////////

template<class SrcType, class WrkType, class DstType, int op>
struct FMXOP_aosos : unary_function<SrcType, DstType> {};

#define SPEC_FMXOP_AOSOS(name, value)\
    template<class SrcType, class WrkType, class DstType>\
    struct FMXOP_aosos<SrcType, WrkType, DstType, value> : unary_function<SrcType, DstType>\
    {\
        WrkType _s2, _s3;\
    \
        __device__ __forceinline__ DstType operator()(const SrcType& v)const\
        {\
            WrkType s1 = saturate_cast<WrkType>(v);\
    \
            WrkType dst = name(s1, this->_s2, this->_s3);\
    \
            return saturate_cast<DstType>(dst);\
        }\
    \
    };

SPEC_FMXOP_AOSOS(fused_multiply_add, 0)
SPEC_FMXOP_AOSOS(fused_multiply_sub, 1)
SPEC_FMXOP_AOSOS(negative_fused_multiply_add, 2)
SPEC_FMXOP_AOSOS(negative_fused_multiply_sub, 3)
SPEC_FMXOP_AOSOS(fused_divide_add_round_near_even, 4)
SPEC_FMXOP_AOSOS(fused_divide_add_round_near_zero, 5)
SPEC_FMXOP_AOSOS(fused_divide_add_round_up, 6)
SPEC_FMXOP_AOSOS(fused_divide_add_round_down, 7)
SPEC_FMXOP_AOSOS(fused_divide_add_approximate, 8)
SPEC_FMXOP_AOSOS(fused_divide_sub_round_near_even, 9)
SPEC_FMXOP_AOSOS(fused_divide_sub_round_near_zero, 10)
SPEC_FMXOP_AOSOS(fused_divide_sub_round_up, 11)
SPEC_FMXOP_AOSOS(fused_divide_sub_round_down, 12)
SPEC_FMXOP_AOSOS(fused_divide_sub_approximate, 13)
SPEC_FMXOP_AOSOS(negative_fused_divide_add_round_near_even, 14)
SPEC_FMXOP_AOSOS(negative_fused_divide_add_round_near_zero, 15)
SPEC_FMXOP_AOSOS(negative_fused_divide_add_round_up, 16)
SPEC_FMXOP_AOSOS(negative_fused_divide_add_round_down, 17)
SPEC_FMXOP_AOSOS(negative_fused_divide_add_approximate, 18)
SPEC_FMXOP_AOSOS(negative_fused_divide_sub_round_near_even, 19)
SPEC_FMXOP_AOSOS(negative_fused_divide_sub_round_near_zero, 20)
SPEC_FMXOP_AOSOS(negative_fused_divide_sub_round_up, 21)
SPEC_FMXOP_AOSOS(negative_fused_divide_sub_round_down, 22)
SPEC_FMXOP_AOSOS(negative_fused_divide_sub_approximate, 23)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_add_round_near_even, 24)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_add_round_near_zero, 25)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_add_round_up, 26)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_add_round_down, 27)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_add_approximate, 28)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_sub_round_near_even, 29)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_sub_round_near_zero, 30)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_sub_round_up, 31)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_sub_round_down, 32)
SPEC_FMXOP_AOSOS(fused_reciprocal_square_sub_approximate, 33)
SPEC_FMXOP_AOSOS(negative_negative_fused_reciprocal_square_add_round_near_even, 34)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_add_round_near_zero, 35)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_add_round_up, 36)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_add_round_down, 37)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_add_approximate, 38)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_sub_round_near_even, 39)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_sub_round_near_zero, 40)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_sub_round_up, 41)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_sub_round_down, 42)
SPEC_FMXOP_AOSOS(negative_fused_reciprocal_square_sub_approximate, 43)

#undef SPEC_FMXOP_AOSOS

//////////////////////////////////////////////////////////////////////////////////////

template<class SrcType1, class SrcType2, class WrkType, class DstType, int op>
struct FMXOP_soaoa : binary_function<SrcType1, SrcType2, DstType>{};

#define SPEC_FMXOP_SOAOA(name, value)\
    template<class SrcType1, class SrcType2, class WrkType, class DstType> \
    struct FMXOP_soaoa<SrcType1, SrcType2, WrkType, DstType, value> : binary_function<SrcType1, SrcType2, DstType>\
    {\
        WrkType _s1;\
    \
        __device__ __forceinline__ DstType operator()(const SrcType1& vb, const SrcType2& vc)const\
        {\
            WrkType s2 = saturate_cast<WrkType>(vb);\
            WrkType s3 = saturate_cast<WrkType>(vc);\
    \
            WrkType dst = name(this->_s1, s2, s3);\
    \
            return saturate_cast<DstType>(dst);\
        }\
    \
    };

SPEC_FMXOP_SOAOA(fused_multiply_add, 0)
SPEC_FMXOP_SOAOA(fused_multiply_sub, 1)
SPEC_FMXOP_SOAOA(negative_fused_multiply_add, 2)
SPEC_FMXOP_SOAOA(negative_fused_multiply_sub, 3)
SPEC_FMXOP_SOAOA(fused_divide_add_round_near_even, 4)
SPEC_FMXOP_SOAOA(fused_divide_add_round_near_zero, 5)
SPEC_FMXOP_SOAOA(fused_divide_add_round_up, 6)
SPEC_FMXOP_SOAOA(fused_divide_add_round_down, 7)
SPEC_FMXOP_SOAOA(fused_divide_add_approximate, 8)
SPEC_FMXOP_SOAOA(fused_divide_sub_round_near_even, 9)
SPEC_FMXOP_SOAOA(fused_divide_sub_round_near_zero, 10)
SPEC_FMXOP_SOAOA(fused_divide_sub_round_up, 11)
SPEC_FMXOP_SOAOA(fused_divide_sub_round_down, 12)
SPEC_FMXOP_SOAOA(fused_divide_sub_approximate, 13)
SPEC_FMXOP_SOAOA(negative_fused_divide_add_round_near_even, 14)
SPEC_FMXOP_SOAOA(negative_fused_divide_add_round_near_zero, 15)
SPEC_FMXOP_SOAOA(negative_fused_divide_add_round_up, 16)
SPEC_FMXOP_SOAOA(negative_fused_divide_add_round_down, 17)
SPEC_FMXOP_SOAOA(negative_fused_divide_add_approximate, 18)
SPEC_FMXOP_SOAOA(negative_fused_divide_sub_round_near_even, 19)
SPEC_FMXOP_SOAOA(negative_fused_divide_sub_round_near_zero, 20)
SPEC_FMXOP_SOAOA(negative_fused_divide_sub_round_up, 21)
SPEC_FMXOP_SOAOA(negative_fused_divide_sub_round_down, 22)
SPEC_FMXOP_SOAOA(negative_fused_divide_sub_approximate, 23)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_add_round_near_even, 24)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_add_round_near_zero, 25)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_add_round_up, 26)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_add_round_down, 27)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_add_approximate, 28)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_sub_round_near_even, 29)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_sub_round_near_zero, 30)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_sub_round_up, 31)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_sub_round_down, 32)
SPEC_FMXOP_SOAOA(fused_reciprocal_square_sub_approximate, 33)
SPEC_FMXOP_SOAOA(negative_negative_fused_reciprocal_square_add_round_near_even, 34)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_add_round_near_zero, 35)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_add_round_up, 36)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_add_round_down, 37)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_add_approximate, 38)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_sub_round_near_even, 39)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_sub_round_near_zero, 40)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_sub_round_up, 41)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_sub_round_down, 42)
SPEC_FMXOP_SOAOA(negative_fused_reciprocal_square_sub_approximate, 43)

#undef SPEC_FMXOP_SOAOA

//////////////////////////////////////////////////////////////////////////////////////

template<class SrcType1, class SrcType2, class WrkType, class DstType, int op>
struct FMXOP_aosoa : binary_function<SrcType1, SrcType2, DstType> {};

#define SPEC_FMXOP_AOSOA(name, value)\
    template<class SrcType1, class SrcType2, class WrkType, class DstType>\
    struct FMXOP_aosoa<SrcType1, SrcType2, WrkType, DstType, value> : binary_function<SrcType1, SrcType2, DstType>\
{\
    WrkType _s2;\
    \
    __device__ __forceinline__ DstType operator()(const SrcType1& va, const SrcType2& vc)const\
{\
    WrkType s1 = saturate_cast<WrkType>(va);\
    WrkType s3 = saturate_cast<WrkType>(vc);\
    \
    WrkType dst = name(s1, this->_s2, s3);\
    \
    return saturate_cast<DstType>(dst);\
}\
};

SPEC_FMXOP_AOSOA(fused_multiply_add, 0)
SPEC_FMXOP_AOSOA(fused_multiply_sub, 1)
SPEC_FMXOP_AOSOA(negative_fused_multiply_add, 2)
SPEC_FMXOP_AOSOA(negative_fused_multiply_sub, 3)
SPEC_FMXOP_AOSOA(fused_divide_add_round_near_even, 4)
SPEC_FMXOP_AOSOA(fused_divide_add_round_near_zero, 5)
SPEC_FMXOP_AOSOA(fused_divide_add_round_up, 6)
SPEC_FMXOP_AOSOA(fused_divide_add_round_down, 7)
SPEC_FMXOP_AOSOA(fused_divide_add_approximate, 8)
SPEC_FMXOP_AOSOA(fused_divide_sub_round_near_even, 9)
SPEC_FMXOP_AOSOA(fused_divide_sub_round_near_zero, 10)
SPEC_FMXOP_AOSOA(fused_divide_sub_round_up, 11)
SPEC_FMXOP_AOSOA(fused_divide_sub_round_down, 12)
SPEC_FMXOP_AOSOA(fused_divide_sub_approximate, 13)
SPEC_FMXOP_AOSOA(negative_fused_divide_add_round_near_even, 14)
SPEC_FMXOP_AOSOA(negative_fused_divide_add_round_near_zero, 15)
SPEC_FMXOP_AOSOA(negative_fused_divide_add_round_up, 16)
SPEC_FMXOP_AOSOA(negative_fused_divide_add_round_down, 17)
SPEC_FMXOP_AOSOA(negative_fused_divide_add_approximate, 18)
SPEC_FMXOP_AOSOA(negative_fused_divide_sub_round_near_even, 19)
SPEC_FMXOP_AOSOA(negative_fused_divide_sub_round_near_zero, 20)
SPEC_FMXOP_AOSOA(negative_fused_divide_sub_round_up, 21)
SPEC_FMXOP_AOSOA(negative_fused_divide_sub_round_down, 22)
SPEC_FMXOP_AOSOA(negative_fused_divide_sub_approximate, 23)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_add_round_near_even, 24)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_add_round_near_zero, 25)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_add_round_up, 26)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_add_round_down, 27)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_add_approximate, 28)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_sub_round_near_even, 29)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_sub_round_near_zero, 30)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_sub_round_up, 31)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_sub_round_down, 32)
SPEC_FMXOP_AOSOA(fused_reciprocal_square_sub_approximate, 33)
SPEC_FMXOP_AOSOA(negative_negative_fused_reciprocal_square_add_round_near_even, 34)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_add_round_near_zero, 35)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_add_round_up, 36)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_add_round_down, 37)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_add_approximate, 38)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_sub_round_near_even, 39)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_sub_round_near_zero, 40)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_sub_round_up, 41)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_sub_round_down, 42)
SPEC_FMXOP_AOSOA(negative_fused_reciprocal_square_sub_approximate, 43)

#undef SPEC_FMXOP_AOSOA


//////////////////////////////////////////////////////////////////////////////////////

template<class SrcType1, class SrcType2, class WrkType, class DstType, int op>
struct FMXOP_aoaos : binary_function<SrcType1, SrcType2, DstType> {};

#define SPEC_FMXOP_AOAOS(name, value)\
    template<class SrcType1, class SrcType2, class WrkType, class DstType>\
    struct FMXOP_aoaos<SrcType1, SrcType2, WrkType, DstType, value> : binary_function<SrcType1, SrcType2, DstType>\
    {\
        WrkType _s3;\
    \
        __device__ __forceinline__ DstType operator()(const SrcType1& va, const SrcType2& vb)const\
        {\
            WrkType s1 = saturate_cast<WrkType>(va);\
            WrkType s2 = saturate_cast<WrkType>(vb);\
    \
            WrkType dst = name(s1, s2, this->_s3);\
    \
            return saturate_cast<DstType>(dst);\
        }\
    };

SPEC_FMXOP_AOAOS(fused_multiply_add, 0)
SPEC_FMXOP_AOAOS(fused_multiply_sub, 1)
SPEC_FMXOP_AOAOS(negative_fused_multiply_add, 2)
SPEC_FMXOP_AOAOS(negative_fused_multiply_sub, 3)
SPEC_FMXOP_AOAOS(fused_divide_add_round_near_even, 4)
SPEC_FMXOP_AOAOS(fused_divide_add_round_near_zero, 5)
SPEC_FMXOP_AOAOS(fused_divide_add_round_up, 6)
SPEC_FMXOP_AOAOS(fused_divide_add_round_down, 7)
SPEC_FMXOP_AOAOS(fused_divide_add_approximate, 8)
SPEC_FMXOP_AOAOS(fused_divide_sub_round_near_even, 9)
SPEC_FMXOP_AOAOS(fused_divide_sub_round_near_zero, 10)
SPEC_FMXOP_AOAOS(fused_divide_sub_round_up, 11)
SPEC_FMXOP_AOAOS(fused_divide_sub_round_down, 12)
SPEC_FMXOP_AOAOS(fused_divide_sub_approximate, 13)
SPEC_FMXOP_AOAOS(negative_fused_divide_add_round_near_even, 14)
SPEC_FMXOP_AOAOS(negative_fused_divide_add_round_near_zero, 15)
SPEC_FMXOP_AOAOS(negative_fused_divide_add_round_up, 16)
SPEC_FMXOP_AOAOS(negative_fused_divide_add_round_down, 17)
SPEC_FMXOP_AOAOS(negative_fused_divide_add_approximate, 18)
SPEC_FMXOP_AOAOS(negative_fused_divide_sub_round_near_even, 19)
SPEC_FMXOP_AOAOS(negative_fused_divide_sub_round_near_zero, 20)
SPEC_FMXOP_AOAOS(negative_fused_divide_sub_round_up, 21)
SPEC_FMXOP_AOAOS(negative_fused_divide_sub_round_down, 22)
SPEC_FMXOP_AOAOS(negative_fused_divide_sub_approximate, 23)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_add_round_near_even, 24)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_add_round_near_zero, 25)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_add_round_up, 26)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_add_round_down, 27)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_add_approximate, 28)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_sub_round_near_even, 29)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_sub_round_near_zero, 30)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_sub_round_up, 31)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_sub_round_down, 32)
SPEC_FMXOP_AOAOS(fused_reciprocal_square_sub_approximate, 33)
SPEC_FMXOP_AOAOS(negative_negative_fused_reciprocal_square_add_round_near_even, 34)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_add_round_near_zero, 35)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_add_round_up, 36)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_add_round_down, 37)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_add_approximate, 38)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_sub_round_near_even, 39)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_sub_round_near_zero, 40)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_sub_round_up, 41)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_sub_round_down, 42)
SPEC_FMXOP_AOAOS(negative_fused_reciprocal_square_sub_approximate, 43)

#undef SPEC_FMXOP_AOAOS

//////////////////////////////////////////////////////////////////////////////////////

template<class SrcType1, class SrcType2, class SrcType3, class WrkType, class DstType, int op>
struct FMXOP_aoaoa : ternary_function<SrcType1, SrcType2, SrcType3, DstType> {};


#define SPEC_FMXOP_AOAOA(name, value)\
template<class SrcType1, class SrcType2, class SrcType3, class WrkType, class DstType> \
struct FMXOP_aoaoa<SrcType1, SrcType2, SrcType3, WrkType, DstType, value> : ternary_function<SrcType1, SrcType2, SrcType3, DstType>\
{\
    __device__ __forceinline__ DstType operator()(const SrcType1& va, const SrcType2& vb, const SrcType3& vc)const\
    {\
        WrkType s1 = saturate_cast<WrkType>(va);\
        WrkType s2 = saturate_cast<WrkType>(vb);\
        WrkType s3 = saturate_cast<WrkType>(vc);\
\
        WrkType dst = name(s1, s2, s3);\
\
        return saturate_cast<DstType>(dst);\
    }\
};

SPEC_FMXOP_AOAOA(fused_multiply_add, 0)
SPEC_FMXOP_AOAOA(fused_multiply_sub, 1)
SPEC_FMXOP_AOAOA(negative_fused_multiply_add, 2)
SPEC_FMXOP_AOAOA(negative_fused_multiply_sub, 3)
SPEC_FMXOP_AOAOA(fused_divide_add_round_near_even, 4)
SPEC_FMXOP_AOAOA(fused_divide_add_round_near_zero, 5)
SPEC_FMXOP_AOAOA(fused_divide_add_round_up, 6)
SPEC_FMXOP_AOAOA(fused_divide_add_round_down, 7)
SPEC_FMXOP_AOAOA(fused_divide_add_approximate, 8)
SPEC_FMXOP_AOAOA(fused_divide_sub_round_near_even, 9)
SPEC_FMXOP_AOAOA(fused_divide_sub_round_near_zero, 10)
SPEC_FMXOP_AOAOA(fused_divide_sub_round_up, 11)
SPEC_FMXOP_AOAOA(fused_divide_sub_round_down, 12)
SPEC_FMXOP_AOAOA(fused_divide_sub_approximate, 13)
SPEC_FMXOP_AOAOA(negative_fused_divide_add_round_near_even, 14)
SPEC_FMXOP_AOAOA(negative_fused_divide_add_round_near_zero, 15)
SPEC_FMXOP_AOAOA(negative_fused_divide_add_round_up, 16)
SPEC_FMXOP_AOAOA(negative_fused_divide_add_round_down, 17)
SPEC_FMXOP_AOAOA(negative_fused_divide_add_approximate, 18)
SPEC_FMXOP_AOAOA(negative_fused_divide_sub_round_near_even, 19)
SPEC_FMXOP_AOAOA(negative_fused_divide_sub_round_near_zero, 20)
SPEC_FMXOP_AOAOA(negative_fused_divide_sub_round_up, 21)
SPEC_FMXOP_AOAOA(negative_fused_divide_sub_round_down, 22)
SPEC_FMXOP_AOAOA(negative_fused_divide_sub_approximate, 23)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_add_round_near_even, 24)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_add_round_near_zero, 25)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_add_round_up, 26)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_add_round_down, 27)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_add_approximate, 28)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_sub_round_near_even, 29)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_sub_round_near_zero, 30)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_sub_round_up, 31)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_sub_round_down, 32)
SPEC_FMXOP_AOAOA(fused_reciprocal_square_sub_approximate, 33)
SPEC_FMXOP_AOAOA(negative_negative_fused_reciprocal_square_add_round_near_even, 34)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_add_round_near_zero, 35)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_add_round_up, 36)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_add_round_down, 37)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_add_approximate, 38)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_sub_round_near_even, 39)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_sub_round_near_zero, 40)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_sub_round_up, 41)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_sub_round_down, 42)
SPEC_FMXOP_AOAOA(negative_fused_reciprocal_square_sub_approximate, 43)

#undef SPEC_FMXOP_AOAOA

//////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType1, typename SrcType2, typename SrcType3, typename DstType, int what>
void FMXImpl_AOAOA(const GpuMat& src1, const GpuMat& src2, const GpuMat& src3, GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    static_assert ((static_cast<int>(VecTraits<SrcType1>::cn) == static_cast<int>(VecTraits<SrcType2>::cn)) && (static_cast<int>(VecTraits<SrcType1>::cn) == static_cast<int>(VecTraits<SrcType3>::cn)) && (static_cast<int>(VecTraits<SrcType1>::cn) == static_cast<int>(VecTraits<DstType>::cn)), "ERROR IN FUNCTION FMXImpl_AOAOA: inconsistent number of channels");

    typedef typename TypeVec<typename std::conditional<std::is_integral<typename VecTraits<SrcType1>::elem_type>::value && std::is_integral<typename VecTraits<SrcType2>::elem_type>::value && std::is_integral<typename VecTraits<SrcType3>::elem_type>::value, float, double>::type, VecTraits<SrcType1>::cn>::vec_type working_type;

    FMXOP_aoaoa<SrcType1, SrcType2, SrcType3, working_type, DstType, what> op;

    if(mask.empty())
        gridTransformTernary(cv::cudev::globPtr<SrcType1>(src1), cv::cudev::globPtr<SrcType2>(src2),cv::cudev::globPtr<SrcType3>(src3), cv::cudev::globPtr<DstType>(dst), op, stream);
    else
        gridTransformTernary(cv::cudev::globPtr<SrcType1>(src1), cv::cudev::globPtr<SrcType2>(src2),cv::cudev::globPtr<SrcType3>(src3), cv::cudev::globPtr<DstType>(dst), op, mask, stream);
}


template <typename SrcType, typename ScalarDepth, typename DstType, int what>
void FMXImpl_SOSOA(const Scalar& value1, const Scalar& value2, const GpuMat& src,  GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    static_assert (static_cast<int>(VecTraits<SrcType>::cn) == static_cast<int>(VecTraits<DstType>::cn), "ERROR IN FUNCTION FMXImpl_SOSOA: inconsistent number of channels");

    typedef typename TypeVec<typename std::conditional<std::is_integral<typename VecTraits<SrcType>::elem_type>::value, float, double>::type, VecTraits<SrcType>::cn>::vec_type working_type;
    typedef typename VecTraits<working_type>::elem_type element_type;

    Scalar_<element_type> value1_ = value1;
    Scalar_<element_type> value2_ = value2;

    FMXOP_sosoa<SrcType,working_type, DstType, what> op;

    op._s1 = VecTraits<working_type>::make(value1_.val);
    op._s2 = VecTraits<working_type>::make(value2_.val);

    if(mask.empty())
        gridTransformUnary(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
    else
        gridTransformUnary(globPtr<SrcType>(src), globPtr<DstType>(dst), op, mask, stream);
}


template <typename SrcType, typename DstType, int what>
void FMXImpl_SOAOS(const Scalar& value1, const GpuMat& src, const Scalar& value2, GpuMat& dst, const GpuMat& mask, Stream& stream)
{

    static_assert (static_cast<int>(VecTraits<SrcType>::cn) == static_cast<int>(VecTraits<DstType>::cn), "ERROR IN FUNCTION FMXImpl_SOAOS: inconsistent number of channels");

    typedef typename TypeVec<typename std::conditional<std::is_integral<typename VecTraits<SrcType>::elem_type>::value, float, double>::type, VecTraits<SrcType>::cn>::vec_type working_type;
    typedef typename VecTraits<working_type>::elem_type element_type;

    Scalar_<element_type> value1_ = value1;
    Scalar_<element_type> value2_ = value2;

    FMXOP_soaos<SrcType, working_type, DstType, what> op;

    op._s1 = VecTraits<working_type>::make(value1_.val);
    op._s3 = VecTraits<working_type>::make(value2_.val);

    if(mask.empty())
        gridTransformUnary(globPtr<SrcType>(src), globPtr<DstType>(dst), op, stream);
    else
        gridTransformUnary(globPtr<SrcType>(src), globPtr<DstType>(dst), op, mask, stream);
}



template <typename SrcType1, typename SrcType2, typename DstType,int what>
void FMXImpl_AOAOS(const GpuMat& src1,const GpuMat& src2, const Scalar& value, cudev::GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    static_assert ((static_cast<int>(VecTraits<SrcType1>::cn) == static_cast<int>(VecTraits<SrcType2>::cn)) && (static_cast<int>(VecTraits<SrcType1>::cn) == static_cast<int>(VecTraits<DstType>::cn)), "ERROR IN FUNCTION FMXImpl_AOAOS: inconsistent number of channels");

    typedef typename TypeVec<typename std::conditional<std::is_integral<typename VecTraits<SrcType1>::elem_type>::value && std::is_integral<typename VecTraits<SrcType2>::elem_type>::value, float, double>::type, VecTraits<SrcType1>::cn>::vec_type working_type;
    typedef typename VecTraits<working_type>::elem_type element_type;

    Scalar_<element_type> value_ = value;

    FMXOP_aoaos<SrcType1, SrcType2, working_type, DstType, what> op;

    op._s3 = VecTraits<working_type>::make(value_.val);

    if(mask.empty())
        gridTransformBinary(globPtr<SrcType1>(src1), globPtr<SrcType2>(src2), globPtr<DstType>(dst), op, stream);
    else
        gridTransformBinary(globPtr<SrcType1>(src1), globPtr<SrcType2>(src2), globPtr<DstType>(dst), op, mask, stream);
}

template <typename SrcType1, typename SrcType2, typename DstType, int what>
void FMXImpl_SOAOA(const Scalar& value, const GpuMat& src1,const GpuMat& src2, GpuMat& dst, const GpuMat& mask, Stream& stream)
{
    static_assert ((static_cast<int>(VecTraits<SrcType1>::cn) == static_cast<int>(VecTraits<SrcType2>::cn)) && (static_cast<int>(VecTraits<SrcType1>::cn) == static_cast<int>(VecTraits<DstType>::cn)), "ERROR IN FUNCTION FMXImpl_SOAOA: inconsistent number of channels");

    typedef typename TypeVec<typename std::conditional<std::is_integral<typename VecTraits<SrcType1>::elem_type>::value && std::is_integral<typename VecTraits<SrcType2>::elem_type>::value, float, double>::type, VecTraits<SrcType1>::cn>::vec_type working_type;
    typedef typename VecTraits<working_type>::elem_type element_type;


    cv::Scalar_<element_type> value_ = value;

    FMXOP_soaoa<SrcType1, SrcType2, working_type, DstType, what> op;

    op._s1 = VecTraits<working_type>::make(value_.val);

    if(mask.empty())
        gridTransformBinary(globPtr<SrcType1>(src1), globPtr<SrcType2>(src2), globPtr<DstType>(dst), op, stream);
    else
        gridTransformBinary(globPtr<SrcType1>(src1), globPtr<SrcType2>(src2), globPtr<DstType>(dst), op, mask, stream);
}


} // anonymous

template<int op>
void fmxImpl(const GpuMat& _src1, const GpuMat& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{

    CV_Assert_5(_src1.size() == _src2.size(),
                _src1.size() == _src3.size(),
                _src1.channels() == _src2.channels(),
                _src1.channels() == _src3.channels(),
                _mask.empty() || ((_mask.size() == _src1.size()) && (_mask.type() == CV_8UC1))
                );

    typedef void(*function_type)(const GpuMat&, const GpuMat&, const GpuMat&, GpuMat&, const GpuMat&, Stream&);

    static const function_type functions[7][7][7][7][4] = {
        {
            {
                {
                    { FMXImpl_AOAOA<uchar, uchar, uchar, uchar, op>, FMXImpl_AOAOA<uchar2, uchar2, uchar2, uchar2, op>, FMXImpl_AOAOA<uchar3, uchar3, uchar3, uchar3, op>, FMXImpl_AOAOA<uchar4, uchar4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, uchar, schar, op>, FMXImpl_AOAOA<uchar2, uchar2, uchar2, char2, op>, FMXImpl_AOAOA<uchar3, uchar3, uchar3, char3, op>, FMXImpl_AOAOA<uchar4, uchar4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, uchar, ushort, op>, FMXImpl_AOAOA<uchar2, uchar2, uchar2, ushort2, op>, FMXImpl_AOAOA<uchar3, uchar3, uchar3, ushort3, op>, FMXImpl_AOAOA<uchar4, uchar4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, uchar, short, op>, FMXImpl_AOAOA<uchar2, uchar2, uchar2, short2, op>, FMXImpl_AOAOA<uchar3, uchar3, uchar3, short3, op>, FMXImpl_AOAOA<uchar4, uchar4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, uchar, int, op>, FMXImpl_AOAOA<uchar2, uchar2, uchar2, int2, op>, FMXImpl_AOAOA<uchar3, uchar3, uchar3, int3, op>, FMXImpl_AOAOA<uchar4, uchar4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, uchar, float, op>, FMXImpl_AOAOA<uchar2, uchar2, uchar2, float2, op>, FMXImpl_AOAOA<uchar3, uchar3, uchar3, float3, op>, FMXImpl_AOAOA<uchar4, uchar4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, uchar, double, op>, FMXImpl_AOAOA<uchar2, uchar2, uchar2, double2, op>, FMXImpl_AOAOA<uchar3, uchar3, uchar3, double3, op>, FMXImpl_AOAOA<uchar4, uchar4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, uchar, schar, uchar, op>, FMXImpl_AOAOA<uchar2, uchar2, char2, uchar2, op>, FMXImpl_AOAOA<uchar3, uchar3, char3, uchar3, op>, FMXImpl_AOAOA<uchar4, uchar4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, schar, schar, op>, FMXImpl_AOAOA<uchar2, uchar2, char2, char2, op>, FMXImpl_AOAOA<uchar3, uchar3, char3, char3, op>, FMXImpl_AOAOA<uchar4, uchar4, char4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, schar, ushort, op>, FMXImpl_AOAOA<uchar2, uchar2, char2, ushort2, op>, FMXImpl_AOAOA<uchar3, uchar3, char3, ushort3, op>, FMXImpl_AOAOA<uchar4, uchar4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, schar, short, op>, FMXImpl_AOAOA<uchar2, uchar2, char2, short2, op>, FMXImpl_AOAOA<uchar3, uchar3, char3, short3, op>, FMXImpl_AOAOA<uchar4, uchar4, char4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, schar, int, op>, FMXImpl_AOAOA<uchar2, uchar2, char2, int2, op>, FMXImpl_AOAOA<uchar3, uchar3, char3, int3, op>, FMXImpl_AOAOA<uchar4, uchar4, char4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, schar, float, op>, FMXImpl_AOAOA<uchar2, uchar2, char2, float2, op>, FMXImpl_AOAOA<uchar3, uchar3, char3, float3, op>, FMXImpl_AOAOA<uchar4, uchar4, char4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, schar, double, op>, FMXImpl_AOAOA<uchar2, uchar2, char2, double2, op>, FMXImpl_AOAOA<uchar3, uchar3, char3, double3, op>, FMXImpl_AOAOA<uchar4, uchar4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, uchar, ushort, uchar, op>, FMXImpl_AOAOA<uchar2, uchar2, ushort2, uchar2, op>, FMXImpl_AOAOA<uchar3, uchar3, ushort3, uchar3, op>, FMXImpl_AOAOA<uchar4, uchar4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, ushort, schar, op>, FMXImpl_AOAOA<uchar2, uchar2, ushort2, char2, op>, FMXImpl_AOAOA<uchar3, uchar3, ushort3, char3, op>, FMXImpl_AOAOA<uchar4, uchar4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, ushort, ushort, op>, FMXImpl_AOAOA<uchar2, uchar2, ushort2, ushort2, op>, FMXImpl_AOAOA<uchar3, uchar3, ushort3, ushort3, op>, FMXImpl_AOAOA<uchar4, uchar4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, ushort, short, op>, FMXImpl_AOAOA<uchar2, uchar2, ushort2, short2, op>, FMXImpl_AOAOA<uchar3, uchar3, ushort3, short3, op>, FMXImpl_AOAOA<uchar4, uchar4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, ushort, int, op>, FMXImpl_AOAOA<uchar2, uchar2, ushort2, int2, op>, FMXImpl_AOAOA<uchar3, uchar3, ushort3, int3, op>, FMXImpl_AOAOA<uchar4, uchar4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, ushort, float, op>, FMXImpl_AOAOA<uchar2, uchar2, ushort2, float2, op>, FMXImpl_AOAOA<uchar3, uchar3, ushort3, float3, op>, FMXImpl_AOAOA<uchar4, uchar4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, ushort, double, op>, FMXImpl_AOAOA<uchar2, uchar2, ushort2, double2, op>, FMXImpl_AOAOA<uchar3, uchar3, ushort3, double3, op>, FMXImpl_AOAOA<uchar4, uchar4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, uchar, short, uchar, op>, FMXImpl_AOAOA<uchar2, uchar2, short2, uchar2, op>, FMXImpl_AOAOA<uchar3, uchar3, short3, uchar3, op>, FMXImpl_AOAOA<uchar4, uchar4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, short, schar, op>, FMXImpl_AOAOA<uchar2, uchar2, short2, char2, op>, FMXImpl_AOAOA<uchar3, uchar3, short3, char3, op>, FMXImpl_AOAOA<uchar4, uchar4, short4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, short, ushort, op>, FMXImpl_AOAOA<uchar2, uchar2, short2, ushort2, op>, FMXImpl_AOAOA<uchar3, uchar3, short3, ushort3, op>, FMXImpl_AOAOA<uchar4, uchar4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, short, short, op>, FMXImpl_AOAOA<uchar2, uchar2, short2, short2, op>, FMXImpl_AOAOA<uchar3, uchar3, short3, short3, op>, FMXImpl_AOAOA<uchar4, uchar4, short4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, short, int, op>, FMXImpl_AOAOA<uchar2, uchar2, short2, int2, op>, FMXImpl_AOAOA<uchar3, uchar3, short3, int3, op>, FMXImpl_AOAOA<uchar4, uchar4, short4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, short, float, op>, FMXImpl_AOAOA<uchar2, uchar2, short2, float2, op>, FMXImpl_AOAOA<uchar3, uchar3, short3, float3, op>, FMXImpl_AOAOA<uchar4, uchar4, short4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, short, double, op>, FMXImpl_AOAOA<uchar2, uchar2, short2, double2, op>, FMXImpl_AOAOA<uchar3, uchar3, short3, double3, op>, FMXImpl_AOAOA<uchar4, uchar4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, uchar, int, uchar, op>, FMXImpl_AOAOA<uchar2, uchar2, int2, uchar2, op>, FMXImpl_AOAOA<uchar3, uchar3, int3, uchar3, op>, FMXImpl_AOAOA<uchar4, uchar4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, int, schar, op>, FMXImpl_AOAOA<uchar2, uchar2, int2, char2, op>, FMXImpl_AOAOA<uchar3, uchar3, int3, char3, op>, FMXImpl_AOAOA<uchar4, uchar4, int4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, int, ushort, op>, FMXImpl_AOAOA<uchar2, uchar2, int2, ushort2, op>, FMXImpl_AOAOA<uchar3, uchar3, int3, ushort3, op>, FMXImpl_AOAOA<uchar4, uchar4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, int, short, op>, FMXImpl_AOAOA<uchar2, uchar2, int2, short2, op>, FMXImpl_AOAOA<uchar3, uchar3, int3, short3, op>, FMXImpl_AOAOA<uchar4, uchar4, int4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, int, int, op>, FMXImpl_AOAOA<uchar2, uchar2, int2, int2, op>, FMXImpl_AOAOA<uchar3, uchar3, int3, int3, op>, FMXImpl_AOAOA<uchar4, uchar4, int4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, int, float, op>, FMXImpl_AOAOA<uchar2, uchar2, int2, float2, op>, FMXImpl_AOAOA<uchar3, uchar3, int3, float3, op>, FMXImpl_AOAOA<uchar4, uchar4, int4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, int, double, op>, FMXImpl_AOAOA<uchar2, uchar2, int2, double2, op>, FMXImpl_AOAOA<uchar3, uchar3, int3, double3, op>, FMXImpl_AOAOA<uchar4, uchar4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, uchar, float, uchar, op>, FMXImpl_AOAOA<uchar2, uchar2, float2, uchar2, op>, FMXImpl_AOAOA<uchar3, uchar3, float3, uchar3, op>, FMXImpl_AOAOA<uchar4, uchar4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, float, schar, op>, FMXImpl_AOAOA<uchar2, uchar2, float2, char2, op>, FMXImpl_AOAOA<uchar3, uchar3, float3, char3, op>, FMXImpl_AOAOA<uchar4, uchar4, float4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, float, ushort, op>, FMXImpl_AOAOA<uchar2, uchar2, float2, ushort2, op>, FMXImpl_AOAOA<uchar3, uchar3, float3, ushort3, op>, FMXImpl_AOAOA<uchar4, uchar4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, float, short, op>, FMXImpl_AOAOA<uchar2, uchar2, float2, short2, op>, FMXImpl_AOAOA<uchar3, uchar3, float3, short3, op>, FMXImpl_AOAOA<uchar4, uchar4, float4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, float, int, op>, FMXImpl_AOAOA<uchar2, uchar2, float2, int2, op>, FMXImpl_AOAOA<uchar3, uchar3, float3, int3, op>, FMXImpl_AOAOA<uchar4, uchar4, float4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, float, float, op>, FMXImpl_AOAOA<uchar2, uchar2, float2, float2, op>, FMXImpl_AOAOA<uchar3, uchar3, float3, float3, op>, FMXImpl_AOAOA<uchar4, uchar4, float4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, float, double, op>, FMXImpl_AOAOA<uchar2, uchar2, float2, double2, op>, FMXImpl_AOAOA<uchar3, uchar3, float3, double3, op>, FMXImpl_AOAOA<uchar4, uchar4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, uchar, double, uchar, op>, FMXImpl_AOAOA<uchar2, uchar2, double2, uchar2, op>, FMXImpl_AOAOA<uchar3, uchar3, double3, uchar3, op>, FMXImpl_AOAOA<uchar4, uchar4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, double, schar, op>, FMXImpl_AOAOA<uchar2, uchar2, double2, char2, op>, FMXImpl_AOAOA<uchar3, uchar3, double3, char3, op>, FMXImpl_AOAOA<uchar4, uchar4, double4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, double, ushort, op>, FMXImpl_AOAOA<uchar2, uchar2, double2, ushort2, op>, FMXImpl_AOAOA<uchar3, uchar3, double3, ushort3, op>, FMXImpl_AOAOA<uchar4, uchar4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, double, short, op>, FMXImpl_AOAOA<uchar2, uchar2, double2, short2, op>, FMXImpl_AOAOA<uchar3, uchar3, double3, short3, op>, FMXImpl_AOAOA<uchar4, uchar4, double4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, double, int, op>, FMXImpl_AOAOA<uchar2, uchar2, double2, int2, op>, FMXImpl_AOAOA<uchar3, uchar3, double3, int3, op>, FMXImpl_AOAOA<uchar4, uchar4, double4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, double, float, op>, FMXImpl_AOAOA<uchar2, uchar2, double2, float2, op>, FMXImpl_AOAOA<uchar3, uchar3, double3, float3, op>, FMXImpl_AOAOA<uchar4, uchar4, double4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, uchar, double, double, op>, FMXImpl_AOAOA<uchar2, uchar2, double2, double2, op>, FMXImpl_AOAOA<uchar3, uchar3, double3, double3, op>, FMXImpl_AOAOA<uchar4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<uchar, schar, uchar, uchar, op>, FMXImpl_AOAOA<uchar2, char2, uchar2, uchar2, op>, FMXImpl_AOAOA<uchar3, char3, uchar3, uchar3, op>, FMXImpl_AOAOA<uchar4, char4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, uchar, schar, op>, FMXImpl_AOAOA<uchar2, char2, uchar2, char2, op>, FMXImpl_AOAOA<uchar3, char3, uchar3, char3, op>, FMXImpl_AOAOA<uchar4, char4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, uchar, ushort, op>, FMXImpl_AOAOA<uchar2, char2, uchar2, ushort2, op>, FMXImpl_AOAOA<uchar3, char3, uchar3, ushort3, op>, FMXImpl_AOAOA<uchar4, char4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, uchar, short, op>, FMXImpl_AOAOA<uchar2, char2, uchar2, short2, op>, FMXImpl_AOAOA<uchar3, char3, uchar3, short3, op>, FMXImpl_AOAOA<uchar4, char4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, uchar, int, op>, FMXImpl_AOAOA<uchar2, char2, uchar2, int2, op>, FMXImpl_AOAOA<uchar3, char3, uchar3, int3, op>, FMXImpl_AOAOA<uchar4, char4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, uchar, float, op>, FMXImpl_AOAOA<uchar2, char2, uchar2, float2, op>, FMXImpl_AOAOA<uchar3, char3, uchar3, float3, op>, FMXImpl_AOAOA<uchar4, char4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, uchar, double, op>, FMXImpl_AOAOA<uchar2, char2, uchar2, double2, op>, FMXImpl_AOAOA<uchar3, char3, uchar3, double3, op>, FMXImpl_AOAOA<uchar4, char4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, schar, schar, uchar, op>, FMXImpl_AOAOA<uchar2, char2, char2, uchar2, op>, FMXImpl_AOAOA<uchar3, char3, char3, uchar3, op>, FMXImpl_AOAOA<uchar4, char4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, schar, schar, op>, FMXImpl_AOAOA<uchar2, char2, char2, char2, op>, FMXImpl_AOAOA<uchar3, char3, char3, char3, op>, FMXImpl_AOAOA<uchar4, char4, char4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, schar, ushort, op>, FMXImpl_AOAOA<uchar2, char2, char2, ushort2, op>, FMXImpl_AOAOA<uchar3, char3, char3, ushort3, op>, FMXImpl_AOAOA<uchar4, char4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, schar, short, op>, FMXImpl_AOAOA<uchar2, char2, char2, short2, op>, FMXImpl_AOAOA<uchar3, char3, char3, short3, op>, FMXImpl_AOAOA<uchar4, char4, char4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, schar, int, op>, FMXImpl_AOAOA<uchar2, char2, char2, int2, op>, FMXImpl_AOAOA<uchar3, char3, char3, int3, op>, FMXImpl_AOAOA<uchar4, char4, char4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, schar, float, op>, FMXImpl_AOAOA<uchar2, char2, char2, float2, op>, FMXImpl_AOAOA<uchar3, char3, char3, float3, op>, FMXImpl_AOAOA<uchar4, char4, char4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, schar, double, op>, FMXImpl_AOAOA<uchar2, char2, char2, double2, op>, FMXImpl_AOAOA<uchar3, char3, char3, double3, op>, FMXImpl_AOAOA<uchar4, char4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, schar, ushort, uchar, op>, FMXImpl_AOAOA<uchar2, char2, ushort2, uchar2, op>, FMXImpl_AOAOA<uchar3, char3, ushort3, uchar3, op>, FMXImpl_AOAOA<uchar4, char4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, ushort, schar, op>, FMXImpl_AOAOA<uchar2, char2, ushort2, char2, op>, FMXImpl_AOAOA<uchar3, char3, ushort3, char3, op>, FMXImpl_AOAOA<uchar4, char4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, ushort, ushort, op>, FMXImpl_AOAOA<uchar2, char2, ushort2, ushort2, op>, FMXImpl_AOAOA<uchar3, char3, ushort3, ushort3, op>, FMXImpl_AOAOA<uchar4, char4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, ushort, short, op>, FMXImpl_AOAOA<uchar2, char2, ushort2, short2, op>, FMXImpl_AOAOA<uchar3, char3, ushort3, short3, op>, FMXImpl_AOAOA<uchar4, char4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, ushort, int, op>, FMXImpl_AOAOA<uchar2, char2, ushort2, int2, op>, FMXImpl_AOAOA<uchar3, char3, ushort3, int3, op>, FMXImpl_AOAOA<uchar4, char4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, ushort, float, op>, FMXImpl_AOAOA<uchar2, char2, ushort2, float2, op>, FMXImpl_AOAOA<uchar3, char3, ushort3, float3, op>, FMXImpl_AOAOA<uchar4, char4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, ushort, double, op>, FMXImpl_AOAOA<uchar2, char2, ushort2, double2, op>, FMXImpl_AOAOA<uchar3, char3, ushort3, double3, op>, FMXImpl_AOAOA<uchar4, char4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, schar, short, uchar, op>, FMXImpl_AOAOA<uchar2, char2, short2, uchar2, op>, FMXImpl_AOAOA<uchar3, char3, short3, uchar3, op>, FMXImpl_AOAOA<uchar4, char4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, short, schar, op>, FMXImpl_AOAOA<uchar2, char2, short2, char2, op>, FMXImpl_AOAOA<uchar3, char3, short3, char3, op>, FMXImpl_AOAOA<uchar4, char4, short4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, short, ushort, op>, FMXImpl_AOAOA<uchar2, char2, short2, ushort2, op>, FMXImpl_AOAOA<uchar3, char3, short3, ushort3, op>, FMXImpl_AOAOA<uchar4, char4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, short, short, op>, FMXImpl_AOAOA<uchar2, char2, short2, short2, op>, FMXImpl_AOAOA<uchar3, char3, short3, short3, op>, FMXImpl_AOAOA<uchar4, char4, short4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, short, int, op>, FMXImpl_AOAOA<uchar2, char2, short2, int2, op>, FMXImpl_AOAOA<uchar3, char3, short3, int3, op>, FMXImpl_AOAOA<uchar4, char4, short4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, short, float, op>, FMXImpl_AOAOA<uchar2, char2, short2, float2, op>, FMXImpl_AOAOA<uchar3, char3, short3, float3, op>, FMXImpl_AOAOA<uchar4, char4, short4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, short, double, op>, FMXImpl_AOAOA<uchar2, char2, short2, double2, op>, FMXImpl_AOAOA<uchar3, char3, short3, double3, op>, FMXImpl_AOAOA<uchar4, char4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, schar, int, uchar, op>, FMXImpl_AOAOA<uchar2, char2, int2, uchar2, op>, FMXImpl_AOAOA<uchar3, char3, int3, uchar3, op>, FMXImpl_AOAOA<uchar4, char4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, int, schar, op>, FMXImpl_AOAOA<uchar2, char2, int2, char2, op>, FMXImpl_AOAOA<uchar3, char3, int3, char3, op>, FMXImpl_AOAOA<uchar4, char4, int4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, int, ushort, op>, FMXImpl_AOAOA<uchar2, char2, int2, ushort2, op>, FMXImpl_AOAOA<uchar3, char3, int3, ushort3, op>, FMXImpl_AOAOA<uchar4, char4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, int, short, op>, FMXImpl_AOAOA<uchar2, char2, int2, short2, op>, FMXImpl_AOAOA<uchar3, char3, int3, short3, op>, FMXImpl_AOAOA<uchar4, char4, int4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, int, int, op>, FMXImpl_AOAOA<uchar2, char2, int2, int2, op>, FMXImpl_AOAOA<uchar3, char3, int3, int3, op>, FMXImpl_AOAOA<uchar4, char4, int4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, int, float, op>, FMXImpl_AOAOA<uchar2, char2, int2, float2, op>, FMXImpl_AOAOA<uchar3, char3, int3, float3, op>, FMXImpl_AOAOA<uchar4, char4, int4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, int, double, op>, FMXImpl_AOAOA<uchar2, char2, int2, double2, op>, FMXImpl_AOAOA<uchar3, char3, int3, double3, op>, FMXImpl_AOAOA<uchar4, char4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, schar, float, uchar, op>, FMXImpl_AOAOA<uchar2, char2, float2, uchar2, op>, FMXImpl_AOAOA<uchar3, char3, float3, uchar3, op>, FMXImpl_AOAOA<uchar4, char4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, float, schar, op>, FMXImpl_AOAOA<uchar2, char2, float2, char2, op>, FMXImpl_AOAOA<uchar3, char3, float3, char3, op>, FMXImpl_AOAOA<uchar4, char4, float4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, float, ushort, op>, FMXImpl_AOAOA<uchar2, char2, float2, ushort2, op>, FMXImpl_AOAOA<uchar3, char3, float3, ushort3, op>, FMXImpl_AOAOA<uchar4, char4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, float, short, op>, FMXImpl_AOAOA<uchar2, char2, float2, short2, op>, FMXImpl_AOAOA<uchar3, char3, float3, short3, op>, FMXImpl_AOAOA<uchar4, char4, float4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, float, int, op>, FMXImpl_AOAOA<uchar2, char2, float2, int2, op>, FMXImpl_AOAOA<uchar3, char3, float3, int3, op>, FMXImpl_AOAOA<uchar4, char4, float4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, float, float, op>, FMXImpl_AOAOA<uchar2, char2, float2, float2, op>, FMXImpl_AOAOA<uchar3, char3, float3, float3, op>, FMXImpl_AOAOA<uchar4, char4, float4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, float, double, op>, FMXImpl_AOAOA<uchar2, char2, float2, double2, op>, FMXImpl_AOAOA<uchar3, char3, float3, double3, op>, FMXImpl_AOAOA<uchar4, char4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, schar, double, uchar, op>, FMXImpl_AOAOA<uchar2, char2, double2, uchar2, op>, FMXImpl_AOAOA<uchar3, char3, double3, uchar3, op>, FMXImpl_AOAOA<uchar4, char4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, double, schar, op>, FMXImpl_AOAOA<uchar2, char2, double2, char2, op>, FMXImpl_AOAOA<uchar3, char3, double3, char3, op>, FMXImpl_AOAOA<uchar4, char4, double4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, double, ushort, op>, FMXImpl_AOAOA<uchar2, char2, double2, ushort2, op>, FMXImpl_AOAOA<uchar3, char3, double3, ushort3, op>, FMXImpl_AOAOA<uchar4, char4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, double, short, op>, FMXImpl_AOAOA<uchar2, char2, double2, short2, op>, FMXImpl_AOAOA<uchar3, char3, double3, short3, op>, FMXImpl_AOAOA<uchar4, char4, double4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, double, int, op>, FMXImpl_AOAOA<uchar2, char2, double2, int2, op>, FMXImpl_AOAOA<uchar3, char3, double3, int3, op>, FMXImpl_AOAOA<uchar4, char4, double4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, double, float, op>, FMXImpl_AOAOA<uchar2, char2, double2, float2, op>, FMXImpl_AOAOA<uchar3, char3, double3, float3, op>, FMXImpl_AOAOA<uchar4, char4, double4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, schar, double, double, op>, FMXImpl_AOAOA<uchar2, char2, double2, double2, op>, FMXImpl_AOAOA<uchar3, char3, double3, double3, op>, FMXImpl_AOAOA<uchar4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<uchar, ushort, uchar, uchar, op>, FMXImpl_AOAOA<uchar2, ushort2, uchar2, uchar2, op>, FMXImpl_AOAOA<uchar3, ushort3, uchar3, uchar3, op>, FMXImpl_AOAOA<uchar4, ushort4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, uchar, schar, op>, FMXImpl_AOAOA<uchar2, ushort2, uchar2, char2, op>, FMXImpl_AOAOA<uchar3, ushort3, uchar3, char3, op>, FMXImpl_AOAOA<uchar4, ushort4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, uchar, ushort, op>, FMXImpl_AOAOA<uchar2, ushort2, uchar2, ushort2, op>, FMXImpl_AOAOA<uchar3, ushort3, uchar3, ushort3, op>, FMXImpl_AOAOA<uchar4, ushort4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, uchar, short, op>, FMXImpl_AOAOA<uchar2, ushort2, uchar2, short2, op>, FMXImpl_AOAOA<uchar3, ushort3, uchar3, short3, op>, FMXImpl_AOAOA<uchar4, ushort4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, uchar, int, op>, FMXImpl_AOAOA<uchar2, ushort2, uchar2, int2, op>, FMXImpl_AOAOA<uchar3, ushort3, uchar3, int3, op>, FMXImpl_AOAOA<uchar4, ushort4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, uchar, float, op>, FMXImpl_AOAOA<uchar2, ushort2, uchar2, float2, op>, FMXImpl_AOAOA<uchar3, ushort3, uchar3, float3, op>, FMXImpl_AOAOA<uchar4, ushort4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, uchar, double, op>, FMXImpl_AOAOA<uchar2, ushort2, uchar2, double2, op>, FMXImpl_AOAOA<uchar3, ushort3, uchar3, double3, op>, FMXImpl_AOAOA<uchar4, ushort4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, ushort, schar, uchar, op>, FMXImpl_AOAOA<uchar2, ushort2, char2, uchar2, op>, FMXImpl_AOAOA<uchar3, ushort3, char3, uchar3, op>, FMXImpl_AOAOA<uchar4, ushort4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, schar, schar, op>, FMXImpl_AOAOA<uchar2, ushort2, char2, char2, op>, FMXImpl_AOAOA<uchar3, ushort3, char3, char3, op>, FMXImpl_AOAOA<uchar4, ushort4, char4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, schar, ushort, op>, FMXImpl_AOAOA<uchar2, ushort2, char2, ushort2, op>, FMXImpl_AOAOA<uchar3, ushort3, char3, ushort3, op>, FMXImpl_AOAOA<uchar4, ushort4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, schar, short, op>, FMXImpl_AOAOA<uchar2, ushort2, char2, short2, op>, FMXImpl_AOAOA<uchar3, ushort3, char3, short3, op>, FMXImpl_AOAOA<uchar4, ushort4, char4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, schar, int, op>, FMXImpl_AOAOA<uchar2, ushort2, char2, int2, op>, FMXImpl_AOAOA<uchar3, ushort3, char3, int3, op>, FMXImpl_AOAOA<uchar4, ushort4, char4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, schar, float, op>, FMXImpl_AOAOA<uchar2, ushort2, char2, float2, op>, FMXImpl_AOAOA<uchar3, ushort3, char3, float3, op>, FMXImpl_AOAOA<uchar4, ushort4, char4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, schar, double, op>, FMXImpl_AOAOA<uchar2, ushort2, char2, double2, op>, FMXImpl_AOAOA<uchar3, ushort3, char3, double3, op>, FMXImpl_AOAOA<uchar4, ushort4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, ushort, ushort, uchar, op>, FMXImpl_AOAOA<uchar2, ushort2, ushort2, uchar2, op>, FMXImpl_AOAOA<uchar3, ushort3, ushort3, uchar3, op>, FMXImpl_AOAOA<uchar4, ushort4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, ushort, schar, op>, FMXImpl_AOAOA<uchar2, ushort2, ushort2, char2, op>, FMXImpl_AOAOA<uchar3, ushort3, ushort3, char3, op>, FMXImpl_AOAOA<uchar4, ushort4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, ushort, ushort, op>, FMXImpl_AOAOA<uchar2, ushort2, ushort2, ushort2, op>, FMXImpl_AOAOA<uchar3, ushort3, ushort3, ushort3, op>, FMXImpl_AOAOA<uchar4, ushort4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, ushort, short, op>, FMXImpl_AOAOA<uchar2, ushort2, ushort2, short2, op>, FMXImpl_AOAOA<uchar3, ushort3, ushort3, short3, op>, FMXImpl_AOAOA<uchar4, ushort4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, ushort, int, op>, FMXImpl_AOAOA<uchar2, ushort2, ushort2, int2, op>, FMXImpl_AOAOA<uchar3, ushort3, ushort3, int3, op>, FMXImpl_AOAOA<uchar4, ushort4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, ushort, float, op>, FMXImpl_AOAOA<uchar2, ushort2, ushort2, float2, op>, FMXImpl_AOAOA<uchar3, ushort3, ushort3, float3, op>, FMXImpl_AOAOA<uchar4, ushort4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, ushort, double, op>, FMXImpl_AOAOA<uchar2, ushort2, ushort2, double2, op>, FMXImpl_AOAOA<uchar3, ushort3, ushort3, double3, op>, FMXImpl_AOAOA<uchar4, ushort4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, ushort, short, uchar, op>, FMXImpl_AOAOA<uchar2, ushort2, short2, uchar2, op>, FMXImpl_AOAOA<uchar3, ushort3, short3, uchar3, op>, FMXImpl_AOAOA<uchar4, ushort4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, short, schar, op>, FMXImpl_AOAOA<uchar2, ushort2, short2, char2, op>, FMXImpl_AOAOA<uchar3, ushort3, short3, char3, op>, FMXImpl_AOAOA<uchar4, ushort4, short4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, short, ushort, op>, FMXImpl_AOAOA<uchar2, ushort2, short2, ushort2, op>, FMXImpl_AOAOA<uchar3, ushort3, short3, ushort3, op>, FMXImpl_AOAOA<uchar4, ushort4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, short, short, op>, FMXImpl_AOAOA<uchar2, ushort2, short2, short2, op>, FMXImpl_AOAOA<uchar3, ushort3, short3, short3, op>, FMXImpl_AOAOA<uchar4, ushort4, short4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, short, int, op>, FMXImpl_AOAOA<uchar2, ushort2, short2, int2, op>, FMXImpl_AOAOA<uchar3, ushort3, short3, int3, op>, FMXImpl_AOAOA<uchar4, ushort4, short4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, short, float, op>, FMXImpl_AOAOA<uchar2, ushort2, short2, float2, op>, FMXImpl_AOAOA<uchar3, ushort3, short3, float3, op>, FMXImpl_AOAOA<uchar4, ushort4, short4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, short, double, op>, FMXImpl_AOAOA<uchar2, ushort2, short2, double2, op>, FMXImpl_AOAOA<uchar3, ushort3, short3, double3, op>, FMXImpl_AOAOA<uchar4, ushort4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, ushort, int, uchar, op>, FMXImpl_AOAOA<uchar2, ushort2, int2, uchar2, op>, FMXImpl_AOAOA<uchar3, ushort3, int3, uchar3, op>, FMXImpl_AOAOA<uchar4, ushort4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, int, schar, op>, FMXImpl_AOAOA<uchar2, ushort2, int2, char2, op>, FMXImpl_AOAOA<uchar3, ushort3, int3, char3, op>, FMXImpl_AOAOA<uchar4, ushort4, int4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, int, ushort, op>, FMXImpl_AOAOA<uchar2, ushort2, int2, ushort2, op>, FMXImpl_AOAOA<uchar3, ushort3, int3, ushort3, op>, FMXImpl_AOAOA<uchar4, ushort4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, int, short, op>, FMXImpl_AOAOA<uchar2, ushort2, int2, short2, op>, FMXImpl_AOAOA<uchar3, ushort3, int3, short3, op>, FMXImpl_AOAOA<uchar4, ushort4, int4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, int, int, op>, FMXImpl_AOAOA<uchar2, ushort2, int2, int2, op>, FMXImpl_AOAOA<uchar3, ushort3, int3, int3, op>, FMXImpl_AOAOA<uchar4, ushort4, int4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, int, float, op>, FMXImpl_AOAOA<uchar2, ushort2, int2, float2, op>, FMXImpl_AOAOA<uchar3, ushort3, int3, float3, op>, FMXImpl_AOAOA<uchar4, ushort4, int4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, int, double, op>, FMXImpl_AOAOA<uchar2, ushort2, int2, double2, op>, FMXImpl_AOAOA<uchar3, ushort3, int3, double3, op>, FMXImpl_AOAOA<uchar4, ushort4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, ushort, float, uchar, op>, FMXImpl_AOAOA<uchar2, ushort2, float2, uchar2, op>, FMXImpl_AOAOA<uchar3, ushort3, float3, uchar3, op>, FMXImpl_AOAOA<uchar4, ushort4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, float, schar, op>, FMXImpl_AOAOA<uchar2, ushort2, float2, char2, op>, FMXImpl_AOAOA<uchar3, ushort3, float3, char3, op>, FMXImpl_AOAOA<uchar4, ushort4, float4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, float, ushort, op>, FMXImpl_AOAOA<uchar2, ushort2, float2, ushort2, op>, FMXImpl_AOAOA<uchar3, ushort3, float3, ushort3, op>, FMXImpl_AOAOA<uchar4, ushort4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, float, short, op>, FMXImpl_AOAOA<uchar2, ushort2, float2, short2, op>, FMXImpl_AOAOA<uchar3, ushort3, float3, short3, op>, FMXImpl_AOAOA<uchar4, ushort4, float4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, float, int, op>, FMXImpl_AOAOA<uchar2, ushort2, float2, int2, op>, FMXImpl_AOAOA<uchar3, ushort3, float3, int3, op>, FMXImpl_AOAOA<uchar4, ushort4, float4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, float, float, op>, FMXImpl_AOAOA<uchar2, ushort2, float2, float2, op>, FMXImpl_AOAOA<uchar3, ushort3, float3, float3, op>, FMXImpl_AOAOA<uchar4, ushort4, float4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, float, double, op>, FMXImpl_AOAOA<uchar2, ushort2, float2, double2, op>, FMXImpl_AOAOA<uchar3, ushort3, float3, double3, op>, FMXImpl_AOAOA<uchar4, ushort4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, ushort, double, uchar, op>, FMXImpl_AOAOA<uchar2, ushort2, double2, uchar2, op>, FMXImpl_AOAOA<uchar3, ushort3, double3, uchar3, op>, FMXImpl_AOAOA<uchar4, ushort4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, double, schar, op>, FMXImpl_AOAOA<uchar2, ushort2, double2, char2, op>, FMXImpl_AOAOA<uchar3, ushort3, double3, char3, op>, FMXImpl_AOAOA<uchar4, ushort4, double4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, double, ushort, op>, FMXImpl_AOAOA<uchar2, ushort2, double2, ushort2, op>, FMXImpl_AOAOA<uchar3, ushort3, double3, ushort3, op>, FMXImpl_AOAOA<uchar4, ushort4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, double, short, op>, FMXImpl_AOAOA<uchar2, ushort2, double2, short2, op>, FMXImpl_AOAOA<uchar3, ushort3, double3, short3, op>, FMXImpl_AOAOA<uchar4, ushort4, double4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, double, int, op>, FMXImpl_AOAOA<uchar2, ushort2, double2, int2, op>, FMXImpl_AOAOA<uchar3, ushort3, double3, int3, op>, FMXImpl_AOAOA<uchar4, ushort4, double4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, double, float, op>, FMXImpl_AOAOA<uchar2, ushort2, double2, float2, op>, FMXImpl_AOAOA<uchar3, ushort3, double3, float3, op>, FMXImpl_AOAOA<uchar4, ushort4, double4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, ushort, double, double, op>, FMXImpl_AOAOA<uchar2, ushort2, double2, double2, op>, FMXImpl_AOAOA<uchar3, ushort3, double3, double3, op>, FMXImpl_AOAOA<uchar4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<uchar, short, uchar, uchar, op>, FMXImpl_AOAOA<uchar2, short2, uchar2, uchar2, op>, FMXImpl_AOAOA<uchar3, short3, uchar3, uchar3, op>, FMXImpl_AOAOA<uchar4, short4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, short, uchar, schar, op>, FMXImpl_AOAOA<uchar2, short2, uchar2, char2, op>, FMXImpl_AOAOA<uchar3, short3, uchar3, char3, op>, FMXImpl_AOAOA<uchar4, short4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, short, uchar, ushort, op>, FMXImpl_AOAOA<uchar2, short2, uchar2, ushort2, op>, FMXImpl_AOAOA<uchar3, short3, uchar3, ushort3, op>, FMXImpl_AOAOA<uchar4, short4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, short, uchar, short, op>, FMXImpl_AOAOA<uchar2, short2, uchar2, short2, op>, FMXImpl_AOAOA<uchar3, short3, uchar3, short3, op>, FMXImpl_AOAOA<uchar4, short4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, short, uchar, int, op>, FMXImpl_AOAOA<uchar2, short2, uchar2, int2, op>, FMXImpl_AOAOA<uchar3, short3, uchar3, int3, op>, FMXImpl_AOAOA<uchar4, short4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, short, uchar, float, op>, FMXImpl_AOAOA<uchar2, short2, uchar2, float2, op>, FMXImpl_AOAOA<uchar3, short3, uchar3, float3, op>, FMXImpl_AOAOA<uchar4, short4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, short, uchar, double, op>, FMXImpl_AOAOA<uchar2, short2, uchar2, double2, op>, FMXImpl_AOAOA<uchar3, short3, uchar3, double3, op>, FMXImpl_AOAOA<uchar4, short4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, short, schar, uchar, op>, FMXImpl_AOAOA<uchar2, short2, char2, uchar2, op>, FMXImpl_AOAOA<uchar3, short3, char3, uchar3, op>, FMXImpl_AOAOA<uchar4, short4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, short, schar, schar, op>, FMXImpl_AOAOA<uchar2, short2, char2, char2, op>, FMXImpl_AOAOA<uchar3, short3, char3, char3, op>, FMXImpl_AOAOA<uchar4, short4, char4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, short, schar, ushort, op>, FMXImpl_AOAOA<uchar2, short2, char2, ushort2, op>, FMXImpl_AOAOA<uchar3, short3, char3, ushort3, op>, FMXImpl_AOAOA<uchar4, short4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, short, schar, short, op>, FMXImpl_AOAOA<uchar2, short2, char2, short2, op>, FMXImpl_AOAOA<uchar3, short3, char3, short3, op>, FMXImpl_AOAOA<uchar4, short4, char4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, short, schar, int, op>, FMXImpl_AOAOA<uchar2, short2, char2, int2, op>, FMXImpl_AOAOA<uchar3, short3, char3, int3, op>, FMXImpl_AOAOA<uchar4, short4, char4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, short, schar, float, op>, FMXImpl_AOAOA<uchar2, short2, char2, float2, op>, FMXImpl_AOAOA<uchar3, short3, char3, float3, op>, FMXImpl_AOAOA<uchar4, short4, char4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, short, schar, double, op>, FMXImpl_AOAOA<uchar2, short2, char2, double2, op>, FMXImpl_AOAOA<uchar3, short3, char3, double3, op>, FMXImpl_AOAOA<uchar4, short4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, short, ushort, uchar, op>, FMXImpl_AOAOA<uchar2, short2, ushort2, uchar2, op>, FMXImpl_AOAOA<uchar3, short3, ushort3, uchar3, op>, FMXImpl_AOAOA<uchar4, short4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, short, ushort, schar, op>, FMXImpl_AOAOA<uchar2, short2, ushort2, char2, op>, FMXImpl_AOAOA<uchar3, short3, ushort3, char3, op>, FMXImpl_AOAOA<uchar4, short4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, short, ushort, ushort, op>, FMXImpl_AOAOA<uchar2, short2, ushort2, ushort2, op>, FMXImpl_AOAOA<uchar3, short3, ushort3, ushort3, op>, FMXImpl_AOAOA<uchar4, short4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, short, ushort, short, op>, FMXImpl_AOAOA<uchar2, short2, ushort2, short2, op>, FMXImpl_AOAOA<uchar3, short3, ushort3, short3, op>, FMXImpl_AOAOA<uchar4, short4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, short, ushort, int, op>, FMXImpl_AOAOA<uchar2, short2, ushort2, int2, op>, FMXImpl_AOAOA<uchar3, short3, ushort3, int3, op>, FMXImpl_AOAOA<uchar4, short4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, short, ushort, float, op>, FMXImpl_AOAOA<uchar2, short2, ushort2, float2, op>, FMXImpl_AOAOA<uchar3, short3, ushort3, float3, op>, FMXImpl_AOAOA<uchar4, short4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, short, ushort, double, op>, FMXImpl_AOAOA<uchar2, short2, ushort2, double2, op>, FMXImpl_AOAOA<uchar3, short3, ushort3, double3, op>, FMXImpl_AOAOA<uchar4, short4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, short, short, uchar, op>, FMXImpl_AOAOA<uchar2, short2, short2, uchar2, op>, FMXImpl_AOAOA<uchar3, short3, short3, uchar3, op>, FMXImpl_AOAOA<uchar4, short4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, short, short, schar, op>, FMXImpl_AOAOA<uchar2, short2, short2, char2, op>, FMXImpl_AOAOA<uchar3, short3, short3, char3, op>, FMXImpl_AOAOA<uchar4, short4, short4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, short, short, ushort, op>, FMXImpl_AOAOA<uchar2, short2, short2, ushort2, op>, FMXImpl_AOAOA<uchar3, short3, short3, ushort3, op>, FMXImpl_AOAOA<uchar4, short4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, short, short, short, op>, FMXImpl_AOAOA<uchar2, short2, short2, short2, op>, FMXImpl_AOAOA<uchar3, short3, short3, short3, op>, FMXImpl_AOAOA<uchar4, short4, short4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, short, short, int, op>, FMXImpl_AOAOA<uchar2, short2, short2, int2, op>, FMXImpl_AOAOA<uchar3, short3, short3, int3, op>, FMXImpl_AOAOA<uchar4, short4, short4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, short, short, float, op>, FMXImpl_AOAOA<uchar2, short2, short2, float2, op>, FMXImpl_AOAOA<uchar3, short3, short3, float3, op>, FMXImpl_AOAOA<uchar4, short4, short4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, short, short, double, op>, FMXImpl_AOAOA<uchar2, short2, short2, double2, op>, FMXImpl_AOAOA<uchar3, short3, short3, double3, op>, FMXImpl_AOAOA<uchar4, short4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, short, int, uchar, op>, FMXImpl_AOAOA<uchar2, short2, int2, uchar2, op>, FMXImpl_AOAOA<uchar3, short3, int3, uchar3, op>, FMXImpl_AOAOA<uchar4, short4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, short, int, schar, op>, FMXImpl_AOAOA<uchar2, short2, int2, char2, op>, FMXImpl_AOAOA<uchar3, short3, int3, char3, op>, FMXImpl_AOAOA<uchar4, short4, int4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, short, int, ushort, op>, FMXImpl_AOAOA<uchar2, short2, int2, ushort2, op>, FMXImpl_AOAOA<uchar3, short3, int3, ushort3, op>, FMXImpl_AOAOA<uchar4, short4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, short, int, short, op>, FMXImpl_AOAOA<uchar2, short2, int2, short2, op>, FMXImpl_AOAOA<uchar3, short3, int3, short3, op>, FMXImpl_AOAOA<uchar4, short4, int4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, short, int, int, op>, FMXImpl_AOAOA<uchar2, short2, int2, int2, op>, FMXImpl_AOAOA<uchar3, short3, int3, int3, op>, FMXImpl_AOAOA<uchar4, short4, int4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, short, int, float, op>, FMXImpl_AOAOA<uchar2, short2, int2, float2, op>, FMXImpl_AOAOA<uchar3, short3, int3, float3, op>, FMXImpl_AOAOA<uchar4, short4, int4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, short, int, double, op>, FMXImpl_AOAOA<uchar2, short2, int2, double2, op>, FMXImpl_AOAOA<uchar3, short3, int3, double3, op>, FMXImpl_AOAOA<uchar4, short4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, short, float, uchar, op>, FMXImpl_AOAOA<uchar2, short2, float2, uchar2, op>, FMXImpl_AOAOA<uchar3, short3, float3, uchar3, op>, FMXImpl_AOAOA<uchar4, short4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, short, float, schar, op>, FMXImpl_AOAOA<uchar2, short2, float2, char2, op>, FMXImpl_AOAOA<uchar3, short3, float3, char3, op>, FMXImpl_AOAOA<uchar4, short4, float4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, short, float, ushort, op>, FMXImpl_AOAOA<uchar2, short2, float2, ushort2, op>, FMXImpl_AOAOA<uchar3, short3, float3, ushort3, op>, FMXImpl_AOAOA<uchar4, short4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, short, float, short, op>, FMXImpl_AOAOA<uchar2, short2, float2, short2, op>, FMXImpl_AOAOA<uchar3, short3, float3, short3, op>, FMXImpl_AOAOA<uchar4, short4, float4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, short, float, int, op>, FMXImpl_AOAOA<uchar2, short2, float2, int2, op>, FMXImpl_AOAOA<uchar3, short3, float3, int3, op>, FMXImpl_AOAOA<uchar4, short4, float4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, short, float, float, op>, FMXImpl_AOAOA<uchar2, short2, float2, float2, op>, FMXImpl_AOAOA<uchar3, short3, float3, float3, op>, FMXImpl_AOAOA<uchar4, short4, float4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, short, float, double, op>, FMXImpl_AOAOA<uchar2, short2, float2, double2, op>, FMXImpl_AOAOA<uchar3, short3, float3, double3, op>, FMXImpl_AOAOA<uchar4, short4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, short, double, uchar, op>, FMXImpl_AOAOA<uchar2, short2, double2, uchar2, op>, FMXImpl_AOAOA<uchar3, short3, double3, uchar3, op>, FMXImpl_AOAOA<uchar4, short4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, short, double, schar, op>, FMXImpl_AOAOA<uchar2, short2, double2, char2, op>, FMXImpl_AOAOA<uchar3, short3, double3, char3, op>, FMXImpl_AOAOA<uchar4, short4, double4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, short, double, ushort, op>, FMXImpl_AOAOA<uchar2, short2, double2, ushort2, op>, FMXImpl_AOAOA<uchar3, short3, double3, ushort3, op>, FMXImpl_AOAOA<uchar4, short4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, short, double, short, op>, FMXImpl_AOAOA<uchar2, short2, double2, short2, op>, FMXImpl_AOAOA<uchar3, short3, double3, short3, op>, FMXImpl_AOAOA<uchar4, short4, double4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, short, double, int, op>, FMXImpl_AOAOA<uchar2, short2, double2, int2, op>, FMXImpl_AOAOA<uchar3, short3, double3, int3, op>, FMXImpl_AOAOA<uchar4, short4, double4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, short, double, float, op>, FMXImpl_AOAOA<uchar2, short2, double2, float2, op>, FMXImpl_AOAOA<uchar3, short3, double3, float3, op>, FMXImpl_AOAOA<uchar4, short4, double4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, short, double, double, op>, FMXImpl_AOAOA<uchar2, short2, double2, double2, op>, FMXImpl_AOAOA<uchar3, short3, double3, double3, op>, FMXImpl_AOAOA<uchar4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<uchar, int, uchar, uchar, op>, FMXImpl_AOAOA<uchar2, int2, uchar2, uchar2, op>, FMXImpl_AOAOA<uchar3, int3, uchar3, uchar3, op>, FMXImpl_AOAOA<uchar4, int4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, int, uchar, schar, op>, FMXImpl_AOAOA<uchar2, int2, uchar2, char2, op>, FMXImpl_AOAOA<uchar3, int3, uchar3, char3, op>, FMXImpl_AOAOA<uchar4, int4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, int, uchar, ushort, op>, FMXImpl_AOAOA<uchar2, int2, uchar2, ushort2, op>, FMXImpl_AOAOA<uchar3, int3, uchar3, ushort3, op>, FMXImpl_AOAOA<uchar4, int4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, int, uchar, short, op>, FMXImpl_AOAOA<uchar2, int2, uchar2, short2, op>, FMXImpl_AOAOA<uchar3, int3, uchar3, short3, op>, FMXImpl_AOAOA<uchar4, int4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, int, uchar, int, op>, FMXImpl_AOAOA<uchar2, int2, uchar2, int2, op>, FMXImpl_AOAOA<uchar3, int3, uchar3, int3, op>, FMXImpl_AOAOA<uchar4, int4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, int, uchar, float, op>, FMXImpl_AOAOA<uchar2, int2, uchar2, float2, op>, FMXImpl_AOAOA<uchar3, int3, uchar3, float3, op>, FMXImpl_AOAOA<uchar4, int4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, int, uchar, double, op>, FMXImpl_AOAOA<uchar2, int2, uchar2, double2, op>, FMXImpl_AOAOA<uchar3, int3, uchar3, double3, op>, FMXImpl_AOAOA<uchar4, int4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, int, schar, uchar, op>, FMXImpl_AOAOA<uchar2, int2, char2, uchar2, op>, FMXImpl_AOAOA<uchar3, int3, char3, uchar3, op>, FMXImpl_AOAOA<uchar4, int4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, int, schar, schar, op>, FMXImpl_AOAOA<uchar2, int2, char2, char2, op>, FMXImpl_AOAOA<uchar3, int3, char3, char3, op>, FMXImpl_AOAOA<uchar4, int4, char4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, int, schar, ushort, op>, FMXImpl_AOAOA<uchar2, int2, char2, ushort2, op>, FMXImpl_AOAOA<uchar3, int3, char3, ushort3, op>, FMXImpl_AOAOA<uchar4, int4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, int, schar, short, op>, FMXImpl_AOAOA<uchar2, int2, char2, short2, op>, FMXImpl_AOAOA<uchar3, int3, char3, short3, op>, FMXImpl_AOAOA<uchar4, int4, char4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, int, schar, int, op>, FMXImpl_AOAOA<uchar2, int2, char2, int2, op>, FMXImpl_AOAOA<uchar3, int3, char3, int3, op>, FMXImpl_AOAOA<uchar4, int4, char4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, int, schar, float, op>, FMXImpl_AOAOA<uchar2, int2, char2, float2, op>, FMXImpl_AOAOA<uchar3, int3, char3, float3, op>, FMXImpl_AOAOA<uchar4, int4, char4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, int, schar, double, op>, FMXImpl_AOAOA<uchar2, int2, char2, double2, op>, FMXImpl_AOAOA<uchar3, int3, char3, double3, op>, FMXImpl_AOAOA<uchar4, int4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, int, ushort, uchar, op>, FMXImpl_AOAOA<uchar2, int2, ushort2, uchar2, op>, FMXImpl_AOAOA<uchar3, int3, ushort3, uchar3, op>, FMXImpl_AOAOA<uchar4, int4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, int, ushort, schar, op>, FMXImpl_AOAOA<uchar2, int2, ushort2, char2, op>, FMXImpl_AOAOA<uchar3, int3, ushort3, char3, op>, FMXImpl_AOAOA<uchar4, int4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, int, ushort, ushort, op>, FMXImpl_AOAOA<uchar2, int2, ushort2, ushort2, op>, FMXImpl_AOAOA<uchar3, int3, ushort3, ushort3, op>, FMXImpl_AOAOA<uchar4, int4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, int, ushort, short, op>, FMXImpl_AOAOA<uchar2, int2, ushort2, short2, op>, FMXImpl_AOAOA<uchar3, int3, ushort3, short3, op>, FMXImpl_AOAOA<uchar4, int4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, int, ushort, int, op>, FMXImpl_AOAOA<uchar2, int2, ushort2, int2, op>, FMXImpl_AOAOA<uchar3, int3, ushort3, int3, op>, FMXImpl_AOAOA<uchar4, int4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, int, ushort, float, op>, FMXImpl_AOAOA<uchar2, int2, ushort2, float2, op>, FMXImpl_AOAOA<uchar3, int3, ushort3, float3, op>, FMXImpl_AOAOA<uchar4, int4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, int, ushort, double, op>, FMXImpl_AOAOA<uchar2, int2, ushort2, double2, op>, FMXImpl_AOAOA<uchar3, int3, ushort3, double3, op>, FMXImpl_AOAOA<uchar4, int4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, int, short, uchar, op>, FMXImpl_AOAOA<uchar2, int2, short2, uchar2, op>, FMXImpl_AOAOA<uchar3, int3, short3, uchar3, op>, FMXImpl_AOAOA<uchar4, int4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, int, short, schar, op>, FMXImpl_AOAOA<uchar2, int2, short2, char2, op>, FMXImpl_AOAOA<uchar3, int3, short3, char3, op>, FMXImpl_AOAOA<uchar4, int4, short4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, int, short, ushort, op>, FMXImpl_AOAOA<uchar2, int2, short2, ushort2, op>, FMXImpl_AOAOA<uchar3, int3, short3, ushort3, op>, FMXImpl_AOAOA<uchar4, int4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, int, short, short, op>, FMXImpl_AOAOA<uchar2, int2, short2, short2, op>, FMXImpl_AOAOA<uchar3, int3, short3, short3, op>, FMXImpl_AOAOA<uchar4, int4, short4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, int, short, int, op>, FMXImpl_AOAOA<uchar2, int2, short2, int2, op>, FMXImpl_AOAOA<uchar3, int3, short3, int3, op>, FMXImpl_AOAOA<uchar4, int4, short4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, int, short, float, op>, FMXImpl_AOAOA<uchar2, int2, short2, float2, op>, FMXImpl_AOAOA<uchar3, int3, short3, float3, op>, FMXImpl_AOAOA<uchar4, int4, short4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, int, short, double, op>, FMXImpl_AOAOA<uchar2, int2, short2, double2, op>, FMXImpl_AOAOA<uchar3, int3, short3, double3, op>, FMXImpl_AOAOA<uchar4, int4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, int, int, uchar, op>, FMXImpl_AOAOA<uchar2, int2, int2, uchar2, op>, FMXImpl_AOAOA<uchar3, int3, int3, uchar3, op>, FMXImpl_AOAOA<uchar4, int4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, int, int, schar, op>, FMXImpl_AOAOA<uchar2, int2, int2, char2, op>, FMXImpl_AOAOA<uchar3, int3, int3, char3, op>, FMXImpl_AOAOA<uchar4, int4, int4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, int, int, ushort, op>, FMXImpl_AOAOA<uchar2, int2, int2, ushort2, op>, FMXImpl_AOAOA<uchar3, int3, int3, ushort3, op>, FMXImpl_AOAOA<uchar4, int4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, int, int, short, op>, FMXImpl_AOAOA<uchar2, int2, int2, short2, op>, FMXImpl_AOAOA<uchar3, int3, int3, short3, op>, FMXImpl_AOAOA<uchar4, int4, int4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, int, int, int, op>, FMXImpl_AOAOA<uchar2, int2, int2, int2, op>, FMXImpl_AOAOA<uchar3, int3, int3, int3, op>, FMXImpl_AOAOA<uchar4, int4, int4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, int, int, float, op>, FMXImpl_AOAOA<uchar2, int2, int2, float2, op>, FMXImpl_AOAOA<uchar3, int3, int3, float3, op>, FMXImpl_AOAOA<uchar4, int4, int4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, int, int, double, op>, FMXImpl_AOAOA<uchar2, int2, int2, double2, op>, FMXImpl_AOAOA<uchar3, int3, int3, double3, op>, FMXImpl_AOAOA<uchar4, int4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, int, float, uchar, op>, FMXImpl_AOAOA<uchar2, int2, float2, uchar2, op>, FMXImpl_AOAOA<uchar3, int3, float3, uchar3, op>, FMXImpl_AOAOA<uchar4, int4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, int, float, schar, op>, FMXImpl_AOAOA<uchar2, int2, float2, char2, op>, FMXImpl_AOAOA<uchar3, int3, float3, char3, op>, FMXImpl_AOAOA<uchar4, int4, float4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, int, float, ushort, op>, FMXImpl_AOAOA<uchar2, int2, float2, ushort2, op>, FMXImpl_AOAOA<uchar3, int3, float3, ushort3, op>, FMXImpl_AOAOA<uchar4, int4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, int, float, short, op>, FMXImpl_AOAOA<uchar2, int2, float2, short2, op>, FMXImpl_AOAOA<uchar3, int3, float3, short3, op>, FMXImpl_AOAOA<uchar4, int4, float4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, int, float, int, op>, FMXImpl_AOAOA<uchar2, int2, float2, int2, op>, FMXImpl_AOAOA<uchar3, int3, float3, int3, op>, FMXImpl_AOAOA<uchar4, int4, float4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, int, float, float, op>, FMXImpl_AOAOA<uchar2, int2, float2, float2, op>, FMXImpl_AOAOA<uchar3, int3, float3, float3, op>, FMXImpl_AOAOA<uchar4, int4, float4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, int, float, double, op>, FMXImpl_AOAOA<uchar2, int2, float2, double2, op>, FMXImpl_AOAOA<uchar3, int3, float3, double3, op>, FMXImpl_AOAOA<uchar4, int4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, int, double, uchar, op>, FMXImpl_AOAOA<uchar2, int2, double2, uchar2, op>, FMXImpl_AOAOA<uchar3, int3, double3, uchar3, op>, FMXImpl_AOAOA<uchar4, int4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, int, double, schar, op>, FMXImpl_AOAOA<uchar2, int2, double2, char2, op>, FMXImpl_AOAOA<uchar3, int3, double3, char3, op>, FMXImpl_AOAOA<uchar4, int4, double4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, int, double, ushort, op>, FMXImpl_AOAOA<uchar2, int2, double2, ushort2, op>, FMXImpl_AOAOA<uchar3, int3, double3, ushort3, op>, FMXImpl_AOAOA<uchar4, int4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, int, double, short, op>, FMXImpl_AOAOA<uchar2, int2, double2, short2, op>, FMXImpl_AOAOA<uchar3, int3, double3, short3, op>, FMXImpl_AOAOA<uchar4, int4, double4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, int, double, int, op>, FMXImpl_AOAOA<uchar2, int2, double2, int2, op>, FMXImpl_AOAOA<uchar3, int3, double3, int3, op>, FMXImpl_AOAOA<uchar4, int4, double4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, int, double, float, op>, FMXImpl_AOAOA<uchar2, int2, double2, float2, op>, FMXImpl_AOAOA<uchar3, int3, double3, float3, op>, FMXImpl_AOAOA<uchar4, int4, double4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, int, double, double, op>, FMXImpl_AOAOA<uchar2, int2, double2, double2, op>, FMXImpl_AOAOA<uchar3, int3, double3, double3, op>, FMXImpl_AOAOA<uchar4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<uchar, float, uchar, uchar, op>, FMXImpl_AOAOA<uchar2, float2, uchar2, uchar2, op>, FMXImpl_AOAOA<uchar3, float3, uchar3, uchar3, op>, FMXImpl_AOAOA<uchar4, float4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, float, uchar, schar, op>, FMXImpl_AOAOA<uchar2, float2, uchar2, char2, op>, FMXImpl_AOAOA<uchar3, float3, uchar3, char3, op>, FMXImpl_AOAOA<uchar4, float4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, float, uchar, ushort, op>, FMXImpl_AOAOA<uchar2, float2, uchar2, ushort2, op>, FMXImpl_AOAOA<uchar3, float3, uchar3, ushort3, op>, FMXImpl_AOAOA<uchar4, float4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, float, uchar, short, op>, FMXImpl_AOAOA<uchar2, float2, uchar2, short2, op>, FMXImpl_AOAOA<uchar3, float3, uchar3, short3, op>, FMXImpl_AOAOA<uchar4, float4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, float, uchar, int, op>, FMXImpl_AOAOA<uchar2, float2, uchar2, int2, op>, FMXImpl_AOAOA<uchar3, float3, uchar3, int3, op>, FMXImpl_AOAOA<uchar4, float4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, float, uchar, float, op>, FMXImpl_AOAOA<uchar2, float2, uchar2, float2, op>, FMXImpl_AOAOA<uchar3, float3, uchar3, float3, op>, FMXImpl_AOAOA<uchar4, float4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, float, uchar, double, op>, FMXImpl_AOAOA<uchar2, float2, uchar2, double2, op>, FMXImpl_AOAOA<uchar3, float3, uchar3, double3, op>, FMXImpl_AOAOA<uchar4, float4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, float, schar, uchar, op>, FMXImpl_AOAOA<uchar2, float2, char2, uchar2, op>, FMXImpl_AOAOA<uchar3, float3, char3, uchar3, op>, FMXImpl_AOAOA<uchar4, float4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, float, schar, schar, op>, FMXImpl_AOAOA<uchar2, float2, char2, char2, op>, FMXImpl_AOAOA<uchar3, float3, char3, char3, op>, FMXImpl_AOAOA<uchar4, float4, char4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, float, schar, ushort, op>, FMXImpl_AOAOA<uchar2, float2, char2, ushort2, op>, FMXImpl_AOAOA<uchar3, float3, char3, ushort3, op>, FMXImpl_AOAOA<uchar4, float4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, float, schar, short, op>, FMXImpl_AOAOA<uchar2, float2, char2, short2, op>, FMXImpl_AOAOA<uchar3, float3, char3, short3, op>, FMXImpl_AOAOA<uchar4, float4, char4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, float, schar, int, op>, FMXImpl_AOAOA<uchar2, float2, char2, int2, op>, FMXImpl_AOAOA<uchar3, float3, char3, int3, op>, FMXImpl_AOAOA<uchar4, float4, char4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, float, schar, float, op>, FMXImpl_AOAOA<uchar2, float2, char2, float2, op>, FMXImpl_AOAOA<uchar3, float3, char3, float3, op>, FMXImpl_AOAOA<uchar4, float4, char4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, float, schar, double, op>, FMXImpl_AOAOA<uchar2, float2, char2, double2, op>, FMXImpl_AOAOA<uchar3, float3, char3, double3, op>, FMXImpl_AOAOA<uchar4, float4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, float, ushort, uchar, op>, FMXImpl_AOAOA<uchar2, float2, ushort2, uchar2, op>, FMXImpl_AOAOA<uchar3, float3, ushort3, uchar3, op>, FMXImpl_AOAOA<uchar4, float4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, float, ushort, schar, op>, FMXImpl_AOAOA<uchar2, float2, ushort2, char2, op>, FMXImpl_AOAOA<uchar3, float3, ushort3, char3, op>, FMXImpl_AOAOA<uchar4, float4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, float, ushort, ushort, op>, FMXImpl_AOAOA<uchar2, float2, ushort2, ushort2, op>, FMXImpl_AOAOA<uchar3, float3, ushort3, ushort3, op>, FMXImpl_AOAOA<uchar4, float4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, float, ushort, short, op>, FMXImpl_AOAOA<uchar2, float2, ushort2, short2, op>, FMXImpl_AOAOA<uchar3, float3, ushort3, short3, op>, FMXImpl_AOAOA<uchar4, float4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, float, ushort, int, op>, FMXImpl_AOAOA<uchar2, float2, ushort2, int2, op>, FMXImpl_AOAOA<uchar3, float3, ushort3, int3, op>, FMXImpl_AOAOA<uchar4, float4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, float, ushort, float, op>, FMXImpl_AOAOA<uchar2, float2, ushort2, float2, op>, FMXImpl_AOAOA<uchar3, float3, ushort3, float3, op>, FMXImpl_AOAOA<uchar4, float4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, float, ushort, double, op>, FMXImpl_AOAOA<uchar2, float2, ushort2, double2, op>, FMXImpl_AOAOA<uchar3, float3, ushort3, double3, op>, FMXImpl_AOAOA<uchar4, float4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, float, short, uchar, op>, FMXImpl_AOAOA<uchar2, float2, short2, uchar2, op>, FMXImpl_AOAOA<uchar3, float3, short3, uchar3, op>, FMXImpl_AOAOA<uchar4, float4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, float, short, schar, op>, FMXImpl_AOAOA<uchar2, float2, short2, char2, op>, FMXImpl_AOAOA<uchar3, float3, short3, char3, op>, FMXImpl_AOAOA<uchar4, float4, short4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, float, short, ushort, op>, FMXImpl_AOAOA<uchar2, float2, short2, ushort2, op>, FMXImpl_AOAOA<uchar3, float3, short3, ushort3, op>, FMXImpl_AOAOA<uchar4, float4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, float, short, short, op>, FMXImpl_AOAOA<uchar2, float2, short2, short2, op>, FMXImpl_AOAOA<uchar3, float3, short3, short3, op>, FMXImpl_AOAOA<uchar4, float4, short4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, float, short, int, op>, FMXImpl_AOAOA<uchar2, float2, short2, int2, op>, FMXImpl_AOAOA<uchar3, float3, short3, int3, op>, FMXImpl_AOAOA<uchar4, float4, short4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, float, short, float, op>, FMXImpl_AOAOA<uchar2, float2, short2, float2, op>, FMXImpl_AOAOA<uchar3, float3, short3, float3, op>, FMXImpl_AOAOA<uchar4, float4, short4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, float, short, double, op>, FMXImpl_AOAOA<uchar2, float2, short2, double2, op>, FMXImpl_AOAOA<uchar3, float3, short3, double3, op>, FMXImpl_AOAOA<uchar4, float4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, float, int, uchar, op>, FMXImpl_AOAOA<uchar2, float2, int2, uchar2, op>, FMXImpl_AOAOA<uchar3, float3, int3, uchar3, op>, FMXImpl_AOAOA<uchar4, float4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, float, int, schar, op>, FMXImpl_AOAOA<uchar2, float2, int2, char2, op>, FMXImpl_AOAOA<uchar3, float3, int3, char3, op>, FMXImpl_AOAOA<uchar4, float4, int4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, float, int, ushort, op>, FMXImpl_AOAOA<uchar2, float2, int2, ushort2, op>, FMXImpl_AOAOA<uchar3, float3, int3, ushort3, op>, FMXImpl_AOAOA<uchar4, float4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, float, int, short, op>, FMXImpl_AOAOA<uchar2, float2, int2, short2, op>, FMXImpl_AOAOA<uchar3, float3, int3, short3, op>, FMXImpl_AOAOA<uchar4, float4, int4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, float, int, int, op>, FMXImpl_AOAOA<uchar2, float2, int2, int2, op>, FMXImpl_AOAOA<uchar3, float3, int3, int3, op>, FMXImpl_AOAOA<uchar4, float4, int4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, float, int, float, op>, FMXImpl_AOAOA<uchar2, float2, int2, float2, op>, FMXImpl_AOAOA<uchar3, float3, int3, float3, op>, FMXImpl_AOAOA<uchar4, float4, int4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, float, int, double, op>, FMXImpl_AOAOA<uchar2, float2, int2, double2, op>, FMXImpl_AOAOA<uchar3, float3, int3, double3, op>, FMXImpl_AOAOA<uchar4, float4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, float, float, uchar, op>, FMXImpl_AOAOA<uchar2, float2, float2, uchar2, op>, FMXImpl_AOAOA<uchar3, float3, float3, uchar3, op>, FMXImpl_AOAOA<uchar4, float4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, float, float, schar, op>, FMXImpl_AOAOA<uchar2, float2, float2, char2, op>, FMXImpl_AOAOA<uchar3, float3, float3, char3, op>, FMXImpl_AOAOA<uchar4, float4, float4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, float, float, ushort, op>, FMXImpl_AOAOA<uchar2, float2, float2, ushort2, op>, FMXImpl_AOAOA<uchar3, float3, float3, ushort3, op>, FMXImpl_AOAOA<uchar4, float4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, float, float, short, op>, FMXImpl_AOAOA<uchar2, float2, float2, short2, op>, FMXImpl_AOAOA<uchar3, float3, float3, short3, op>, FMXImpl_AOAOA<uchar4, float4, float4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, float, float, int, op>, FMXImpl_AOAOA<uchar2, float2, float2, int2, op>, FMXImpl_AOAOA<uchar3, float3, float3, int3, op>, FMXImpl_AOAOA<uchar4, float4, float4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, float, float, float, op>, FMXImpl_AOAOA<uchar2, float2, float2, float2, op>, FMXImpl_AOAOA<uchar3, float3, float3, float3, op>, FMXImpl_AOAOA<uchar4, float4, float4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, float, float, double, op>, FMXImpl_AOAOA<uchar2, float2, float2, double2, op>, FMXImpl_AOAOA<uchar3, float3, float3, double3, op>, FMXImpl_AOAOA<uchar4, float4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, float, double, uchar, op>, FMXImpl_AOAOA<uchar2, float2, double2, uchar2, op>, FMXImpl_AOAOA<uchar3, float3, double3, uchar3, op>, FMXImpl_AOAOA<uchar4, float4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, float, double, schar, op>, FMXImpl_AOAOA<uchar2, float2, double2, char2, op>, FMXImpl_AOAOA<uchar3, float3, double3, char3, op>, FMXImpl_AOAOA<uchar4, float4, double4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, float, double, ushort, op>, FMXImpl_AOAOA<uchar2, float2, double2, ushort2, op>, FMXImpl_AOAOA<uchar3, float3, double3, ushort3, op>, FMXImpl_AOAOA<uchar4, float4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, float, double, short, op>, FMXImpl_AOAOA<uchar2, float2, double2, short2, op>, FMXImpl_AOAOA<uchar3, float3, double3, short3, op>, FMXImpl_AOAOA<uchar4, float4, double4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, float, double, int, op>, FMXImpl_AOAOA<uchar2, float2, double2, int2, op>, FMXImpl_AOAOA<uchar3, float3, double3, int3, op>, FMXImpl_AOAOA<uchar4, float4, double4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, float, double, float, op>, FMXImpl_AOAOA<uchar2, float2, double2, float2, op>, FMXImpl_AOAOA<uchar3, float3, double3, float3, op>, FMXImpl_AOAOA<uchar4, float4, double4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, float, double, double, op>, FMXImpl_AOAOA<uchar2, float2, double2, double2, op>, FMXImpl_AOAOA<uchar3, float3, double3, double3, op>, FMXImpl_AOAOA<uchar4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<uchar, double, uchar, uchar, op>, FMXImpl_AOAOA<uchar2, double2, uchar2, uchar2, op>, FMXImpl_AOAOA<uchar3, double3, uchar3, uchar3, op>, FMXImpl_AOAOA<uchar4, double4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, double, uchar, schar, op>, FMXImpl_AOAOA<uchar2, double2, uchar2, char2, op>, FMXImpl_AOAOA<uchar3, double3, uchar3, char3, op>, FMXImpl_AOAOA<uchar4, double4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, double, uchar, ushort, op>, FMXImpl_AOAOA<uchar2, double2, uchar2, ushort2, op>, FMXImpl_AOAOA<uchar3, double3, uchar3, ushort3, op>, FMXImpl_AOAOA<uchar4, double4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, double, uchar, short, op>, FMXImpl_AOAOA<uchar2, double2, uchar2, short2, op>, FMXImpl_AOAOA<uchar3, double3, uchar3, short3, op>, FMXImpl_AOAOA<uchar4, double4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, double, uchar, int, op>, FMXImpl_AOAOA<uchar2, double2, uchar2, int2, op>, FMXImpl_AOAOA<uchar3, double3, uchar3, int3, op>, FMXImpl_AOAOA<uchar4, double4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, double, uchar, float, op>, FMXImpl_AOAOA<uchar2, double2, uchar2, float2, op>, FMXImpl_AOAOA<uchar3, double3, uchar3, float3, op>, FMXImpl_AOAOA<uchar4, double4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, double, uchar, double, op>, FMXImpl_AOAOA<uchar2, double2, uchar2, double2, op>, FMXImpl_AOAOA<uchar3, double3, uchar3, double3, op>, FMXImpl_AOAOA<uchar4, double4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, double, schar, uchar, op>, FMXImpl_AOAOA<uchar2, double2, char2, uchar2, op>, FMXImpl_AOAOA<uchar3, double3, char3, uchar3, op>, FMXImpl_AOAOA<uchar4, double4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, double, schar, schar, op>, FMXImpl_AOAOA<uchar2, double2, char2, char2, op>, FMXImpl_AOAOA<uchar3, double3, char3, char3, op>, FMXImpl_AOAOA<uchar4, double4, char4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, double, schar, ushort, op>, FMXImpl_AOAOA<uchar2, double2, char2, ushort2, op>, FMXImpl_AOAOA<uchar3, double3, char3, ushort3, op>, FMXImpl_AOAOA<uchar4, double4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, double, schar, short, op>, FMXImpl_AOAOA<uchar2, double2, char2, short2, op>, FMXImpl_AOAOA<uchar3, double3, char3, short3, op>, FMXImpl_AOAOA<uchar4, double4, char4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, double, schar, int, op>, FMXImpl_AOAOA<uchar2, double2, char2, int2, op>, FMXImpl_AOAOA<uchar3, double3, char3, int3, op>, FMXImpl_AOAOA<uchar4, double4, char4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, double, schar, float, op>, FMXImpl_AOAOA<uchar2, double2, char2, float2, op>, FMXImpl_AOAOA<uchar3, double3, char3, float3, op>, FMXImpl_AOAOA<uchar4, double4, char4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, double, schar, double, op>, FMXImpl_AOAOA<uchar2, double2, char2, double2, op>, FMXImpl_AOAOA<uchar3, double3, char3, double3, op>, FMXImpl_AOAOA<uchar4, double4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, double, ushort, uchar, op>, FMXImpl_AOAOA<uchar2, double2, ushort2, uchar2, op>, FMXImpl_AOAOA<uchar3, double3, ushort3, uchar3, op>, FMXImpl_AOAOA<uchar4, double4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, double, ushort, schar, op>, FMXImpl_AOAOA<uchar2, double2, ushort2, char2, op>, FMXImpl_AOAOA<uchar3, double3, ushort3, char3, op>, FMXImpl_AOAOA<uchar4, double4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, double, ushort, ushort, op>, FMXImpl_AOAOA<uchar2, double2, ushort2, ushort2, op>, FMXImpl_AOAOA<uchar3, double3, ushort3, ushort3, op>, FMXImpl_AOAOA<uchar4, double4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, double, ushort, short, op>, FMXImpl_AOAOA<uchar2, double2, ushort2, short2, op>, FMXImpl_AOAOA<uchar3, double3, ushort3, short3, op>, FMXImpl_AOAOA<uchar4, double4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, double, ushort, int, op>, FMXImpl_AOAOA<uchar2, double2, ushort2, int2, op>, FMXImpl_AOAOA<uchar3, double3, ushort3, int3, op>, FMXImpl_AOAOA<uchar4, double4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, double, ushort, float, op>, FMXImpl_AOAOA<uchar2, double2, ushort2, float2, op>, FMXImpl_AOAOA<uchar3, double3, ushort3, float3, op>, FMXImpl_AOAOA<uchar4, double4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, double, ushort, double, op>, FMXImpl_AOAOA<uchar2, double2, ushort2, double2, op>, FMXImpl_AOAOA<uchar3, double3, ushort3, double3, op>, FMXImpl_AOAOA<uchar4, double4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, double, short, uchar, op>, FMXImpl_AOAOA<uchar2, double2, short2, uchar2, op>, FMXImpl_AOAOA<uchar3, double3, short3, uchar3, op>, FMXImpl_AOAOA<uchar4, double4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, double, short, schar, op>, FMXImpl_AOAOA<uchar2, double2, short2, char2, op>, FMXImpl_AOAOA<uchar3, double3, short3, char3, op>, FMXImpl_AOAOA<uchar4, double4, short4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, double, short, ushort, op>, FMXImpl_AOAOA<uchar2, double2, short2, ushort2, op>, FMXImpl_AOAOA<uchar3, double3, short3, ushort3, op>, FMXImpl_AOAOA<uchar4, double4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, double, short, short, op>, FMXImpl_AOAOA<uchar2, double2, short2, short2, op>, FMXImpl_AOAOA<uchar3, double3, short3, short3, op>, FMXImpl_AOAOA<uchar4, double4, short4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, double, short, int, op>, FMXImpl_AOAOA<uchar2, double2, short2, int2, op>, FMXImpl_AOAOA<uchar3, double3, short3, int3, op>, FMXImpl_AOAOA<uchar4, double4, short4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, double, short, float, op>, FMXImpl_AOAOA<uchar2, double2, short2, float2, op>, FMXImpl_AOAOA<uchar3, double3, short3, float3, op>, FMXImpl_AOAOA<uchar4, double4, short4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, double, short, double, op>, FMXImpl_AOAOA<uchar2, double2, short2, double2, op>, FMXImpl_AOAOA<uchar3, double3, short3, double3, op>, FMXImpl_AOAOA<uchar4, double4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, double, int, uchar, op>, FMXImpl_AOAOA<uchar2, double2, int2, uchar2, op>, FMXImpl_AOAOA<uchar3, double3, int3, uchar3, op>, FMXImpl_AOAOA<uchar4, double4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, double, int, schar, op>, FMXImpl_AOAOA<uchar2, double2, int2, char2, op>, FMXImpl_AOAOA<uchar3, double3, int3, char3, op>, FMXImpl_AOAOA<uchar4, double4, int4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, double, int, ushort, op>, FMXImpl_AOAOA<uchar2, double2, int2, ushort2, op>, FMXImpl_AOAOA<uchar3, double3, int3, ushort3, op>, FMXImpl_AOAOA<uchar4, double4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, double, int, short, op>, FMXImpl_AOAOA<uchar2, double2, int2, short2, op>, FMXImpl_AOAOA<uchar3, double3, int3, short3, op>, FMXImpl_AOAOA<uchar4, double4, int4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, double, int, int, op>, FMXImpl_AOAOA<uchar2, double2, int2, int2, op>, FMXImpl_AOAOA<uchar3, double3, int3, int3, op>, FMXImpl_AOAOA<uchar4, double4, int4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, double, int, float, op>, FMXImpl_AOAOA<uchar2, double2, int2, float2, op>, FMXImpl_AOAOA<uchar3, double3, int3, float3, op>, FMXImpl_AOAOA<uchar4, double4, int4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, double, int, double, op>, FMXImpl_AOAOA<uchar2, double2, int2, double2, op>, FMXImpl_AOAOA<uchar3, double3, int3, double3, op>, FMXImpl_AOAOA<uchar4, double4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, double, float, uchar, op>, FMXImpl_AOAOA<uchar2, double2, float2, uchar2, op>, FMXImpl_AOAOA<uchar3, double3, float3, uchar3, op>, FMXImpl_AOAOA<uchar4, double4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, double, float, schar, op>, FMXImpl_AOAOA<uchar2, double2, float2, char2, op>, FMXImpl_AOAOA<uchar3, double3, float3, char3, op>, FMXImpl_AOAOA<uchar4, double4, float4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, double, float, ushort, op>, FMXImpl_AOAOA<uchar2, double2, float2, ushort2, op>, FMXImpl_AOAOA<uchar3, double3, float3, ushort3, op>, FMXImpl_AOAOA<uchar4, double4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, double, float, short, op>, FMXImpl_AOAOA<uchar2, double2, float2, short2, op>, FMXImpl_AOAOA<uchar3, double3, float3, short3, op>, FMXImpl_AOAOA<uchar4, double4, float4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, double, float, int, op>, FMXImpl_AOAOA<uchar2, double2, float2, int2, op>, FMXImpl_AOAOA<uchar3, double3, float3, int3, op>, FMXImpl_AOAOA<uchar4, double4, float4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, double, float, float, op>, FMXImpl_AOAOA<uchar2, double2, float2, float2, op>, FMXImpl_AOAOA<uchar3, double3, float3, float3, op>, FMXImpl_AOAOA<uchar4, double4, float4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, double, float, double, op>, FMXImpl_AOAOA<uchar2, double2, float2, double2, op>, FMXImpl_AOAOA<uchar3, double3, float3, double3, op>, FMXImpl_AOAOA<uchar4, double4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<uchar, double, double, uchar, op>, FMXImpl_AOAOA<uchar2, double2, double2, uchar2, op>, FMXImpl_AOAOA<uchar3, double3, double3, uchar3, op>, FMXImpl_AOAOA<uchar4, double4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<uchar, double, double, schar, op>, FMXImpl_AOAOA<uchar2, double2, double2, char2, op>, FMXImpl_AOAOA<uchar3, double3, double3, char3, op>, FMXImpl_AOAOA<uchar4, double4, double4, char4, op>  },
                    { FMXImpl_AOAOA<uchar, double, double, ushort, op>, FMXImpl_AOAOA<uchar2, double2, double2, ushort2, op>, FMXImpl_AOAOA<uchar3, double3, double3, ushort3, op>, FMXImpl_AOAOA<uchar4, double4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<uchar, double, double, short, op>, FMXImpl_AOAOA<uchar2, double2, double2, short2, op>, FMXImpl_AOAOA<uchar3, double3, double3, short3, op>, FMXImpl_AOAOA<uchar4, double4, double4, short4, op>  },
                    { FMXImpl_AOAOA<uchar, double, double, int, op>, FMXImpl_AOAOA<uchar2, double2, double2, int2, op>, FMXImpl_AOAOA<uchar3, double3, double3, int3, op>, FMXImpl_AOAOA<uchar4, double4, double4, int4, op>  },
                    { FMXImpl_AOAOA<uchar, double, double, float, op>, FMXImpl_AOAOA<uchar2, double2, double2, float2, op>, FMXImpl_AOAOA<uchar3, double3, double3, float3, op>, FMXImpl_AOAOA<uchar4, double4, double4, float4, op>  },
                    { FMXImpl_AOAOA<uchar, double, double, double, op>, FMXImpl_AOAOA<uchar2, double2, double2, double2, op>, FMXImpl_AOAOA<uchar3, double3, double3, double3, op>, FMXImpl_AOAOA<uchar4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { FMXImpl_AOAOA<schar, uchar, uchar, uchar, op>, FMXImpl_AOAOA<char2, uchar2, uchar2, uchar2, op>, FMXImpl_AOAOA<char3, uchar3, uchar3, uchar3, op>, FMXImpl_AOAOA<char4, uchar4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, uchar, schar, op>, FMXImpl_AOAOA<char2, uchar2, uchar2, char2, op>, FMXImpl_AOAOA<char3, uchar3, uchar3, char3, op>, FMXImpl_AOAOA<char4, uchar4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, uchar, ushort, op>, FMXImpl_AOAOA<char2, uchar2, uchar2, ushort2, op>, FMXImpl_AOAOA<char3, uchar3, uchar3, ushort3, op>, FMXImpl_AOAOA<char4, uchar4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, uchar, short, op>, FMXImpl_AOAOA<char2, uchar2, uchar2, short2, op>, FMXImpl_AOAOA<char3, uchar3, uchar3, short3, op>, FMXImpl_AOAOA<char4, uchar4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, uchar, int, op>, FMXImpl_AOAOA<char2, uchar2, uchar2, int2, op>, FMXImpl_AOAOA<char3, uchar3, uchar3, int3, op>, FMXImpl_AOAOA<char4, uchar4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, uchar, float, op>, FMXImpl_AOAOA<char2, uchar2, uchar2, float2, op>, FMXImpl_AOAOA<char3, uchar3, uchar3, float3, op>, FMXImpl_AOAOA<char4, uchar4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, uchar, double, op>, FMXImpl_AOAOA<char2, uchar2, uchar2, double2, op>, FMXImpl_AOAOA<char3, uchar3, uchar3, double3, op>, FMXImpl_AOAOA<char4, uchar4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, uchar, schar, uchar, op>, FMXImpl_AOAOA<char2, uchar2, char2, uchar2, op>, FMXImpl_AOAOA<char3, uchar3, char3, uchar3, op>, FMXImpl_AOAOA<char4, uchar4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, schar, schar, op>, FMXImpl_AOAOA<char2, uchar2, char2, char2, op>, FMXImpl_AOAOA<char3, uchar3, char3, char3, op>, FMXImpl_AOAOA<char4, uchar4, char4, char4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, schar, ushort, op>, FMXImpl_AOAOA<char2, uchar2, char2, ushort2, op>, FMXImpl_AOAOA<char3, uchar3, char3, ushort3, op>, FMXImpl_AOAOA<char4, uchar4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, schar, short, op>, FMXImpl_AOAOA<char2, uchar2, char2, short2, op>, FMXImpl_AOAOA<char3, uchar3, char3, short3, op>, FMXImpl_AOAOA<char4, uchar4, char4, short4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, schar, int, op>, FMXImpl_AOAOA<char2, uchar2, char2, int2, op>, FMXImpl_AOAOA<char3, uchar3, char3, int3, op>, FMXImpl_AOAOA<char4, uchar4, char4, int4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, schar, float, op>, FMXImpl_AOAOA<char2, uchar2, char2, float2, op>, FMXImpl_AOAOA<char3, uchar3, char3, float3, op>, FMXImpl_AOAOA<char4, uchar4, char4, float4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, schar, double, op>, FMXImpl_AOAOA<char2, uchar2, char2, double2, op>, FMXImpl_AOAOA<char3, uchar3, char3, double3, op>, FMXImpl_AOAOA<char4, uchar4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, uchar, ushort, uchar, op>, FMXImpl_AOAOA<char2, uchar2, ushort2, uchar2, op>, FMXImpl_AOAOA<char3, uchar3, ushort3, uchar3, op>, FMXImpl_AOAOA<char4, uchar4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, ushort, schar, op>, FMXImpl_AOAOA<char2, uchar2, ushort2, char2, op>, FMXImpl_AOAOA<char3, uchar3, ushort3, char3, op>, FMXImpl_AOAOA<char4, uchar4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, ushort, ushort, op>, FMXImpl_AOAOA<char2, uchar2, ushort2, ushort2, op>, FMXImpl_AOAOA<char3, uchar3, ushort3, ushort3, op>, FMXImpl_AOAOA<char4, uchar4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, ushort, short, op>, FMXImpl_AOAOA<char2, uchar2, ushort2, short2, op>, FMXImpl_AOAOA<char3, uchar3, ushort3, short3, op>, FMXImpl_AOAOA<char4, uchar4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, ushort, int, op>, FMXImpl_AOAOA<char2, uchar2, ushort2, int2, op>, FMXImpl_AOAOA<char3, uchar3, ushort3, int3, op>, FMXImpl_AOAOA<char4, uchar4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, ushort, float, op>, FMXImpl_AOAOA<char2, uchar2, ushort2, float2, op>, FMXImpl_AOAOA<char3, uchar3, ushort3, float3, op>, FMXImpl_AOAOA<char4, uchar4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, ushort, double, op>, FMXImpl_AOAOA<char2, uchar2, ushort2, double2, op>, FMXImpl_AOAOA<char3, uchar3, ushort3, double3, op>, FMXImpl_AOAOA<char4, uchar4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, uchar, short, uchar, op>, FMXImpl_AOAOA<char2, uchar2, short2, uchar2, op>, FMXImpl_AOAOA<char3, uchar3, short3, uchar3, op>, FMXImpl_AOAOA<char4, uchar4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, short, schar, op>, FMXImpl_AOAOA<char2, uchar2, short2, char2, op>, FMXImpl_AOAOA<char3, uchar3, short3, char3, op>, FMXImpl_AOAOA<char4, uchar4, short4, char4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, short, ushort, op>, FMXImpl_AOAOA<char2, uchar2, short2, ushort2, op>, FMXImpl_AOAOA<char3, uchar3, short3, ushort3, op>, FMXImpl_AOAOA<char4, uchar4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, short, short, op>, FMXImpl_AOAOA<char2, uchar2, short2, short2, op>, FMXImpl_AOAOA<char3, uchar3, short3, short3, op>, FMXImpl_AOAOA<char4, uchar4, short4, short4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, short, int, op>, FMXImpl_AOAOA<char2, uchar2, short2, int2, op>, FMXImpl_AOAOA<char3, uchar3, short3, int3, op>, FMXImpl_AOAOA<char4, uchar4, short4, int4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, short, float, op>, FMXImpl_AOAOA<char2, uchar2, short2, float2, op>, FMXImpl_AOAOA<char3, uchar3, short3, float3, op>, FMXImpl_AOAOA<char4, uchar4, short4, float4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, short, double, op>, FMXImpl_AOAOA<char2, uchar2, short2, double2, op>, FMXImpl_AOAOA<char3, uchar3, short3, double3, op>, FMXImpl_AOAOA<char4, uchar4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, uchar, int, uchar, op>, FMXImpl_AOAOA<char2, uchar2, int2, uchar2, op>, FMXImpl_AOAOA<char3, uchar3, int3, uchar3, op>, FMXImpl_AOAOA<char4, uchar4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, int, schar, op>, FMXImpl_AOAOA<char2, uchar2, int2, char2, op>, FMXImpl_AOAOA<char3, uchar3, int3, char3, op>, FMXImpl_AOAOA<char4, uchar4, int4, char4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, int, ushort, op>, FMXImpl_AOAOA<char2, uchar2, int2, ushort2, op>, FMXImpl_AOAOA<char3, uchar3, int3, ushort3, op>, FMXImpl_AOAOA<char4, uchar4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, int, short, op>, FMXImpl_AOAOA<char2, uchar2, int2, short2, op>, FMXImpl_AOAOA<char3, uchar3, int3, short3, op>, FMXImpl_AOAOA<char4, uchar4, int4, short4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, int, int, op>, FMXImpl_AOAOA<char2, uchar2, int2, int2, op>, FMXImpl_AOAOA<char3, uchar3, int3, int3, op>, FMXImpl_AOAOA<char4, uchar4, int4, int4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, int, float, op>, FMXImpl_AOAOA<char2, uchar2, int2, float2, op>, FMXImpl_AOAOA<char3, uchar3, int3, float3, op>, FMXImpl_AOAOA<char4, uchar4, int4, float4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, int, double, op>, FMXImpl_AOAOA<char2, uchar2, int2, double2, op>, FMXImpl_AOAOA<char3, uchar3, int3, double3, op>, FMXImpl_AOAOA<char4, uchar4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, uchar, float, uchar, op>, FMXImpl_AOAOA<char2, uchar2, float2, uchar2, op>, FMXImpl_AOAOA<char3, uchar3, float3, uchar3, op>, FMXImpl_AOAOA<char4, uchar4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, float, schar, op>, FMXImpl_AOAOA<char2, uchar2, float2, char2, op>, FMXImpl_AOAOA<char3, uchar3, float3, char3, op>, FMXImpl_AOAOA<char4, uchar4, float4, char4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, float, ushort, op>, FMXImpl_AOAOA<char2, uchar2, float2, ushort2, op>, FMXImpl_AOAOA<char3, uchar3, float3, ushort3, op>, FMXImpl_AOAOA<char4, uchar4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, float, short, op>, FMXImpl_AOAOA<char2, uchar2, float2, short2, op>, FMXImpl_AOAOA<char3, uchar3, float3, short3, op>, FMXImpl_AOAOA<char4, uchar4, float4, short4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, float, int, op>, FMXImpl_AOAOA<char2, uchar2, float2, int2, op>, FMXImpl_AOAOA<char3, uchar3, float3, int3, op>, FMXImpl_AOAOA<char4, uchar4, float4, int4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, float, float, op>, FMXImpl_AOAOA<char2, uchar2, float2, float2, op>, FMXImpl_AOAOA<char3, uchar3, float3, float3, op>, FMXImpl_AOAOA<char4, uchar4, float4, float4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, float, double, op>, FMXImpl_AOAOA<char2, uchar2, float2, double2, op>, FMXImpl_AOAOA<char3, uchar3, float3, double3, op>, FMXImpl_AOAOA<char4, uchar4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, uchar, double, uchar, op>, FMXImpl_AOAOA<char2, uchar2, double2, uchar2, op>, FMXImpl_AOAOA<char3, uchar3, double3, uchar3, op>, FMXImpl_AOAOA<char4, uchar4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, double, schar, op>, FMXImpl_AOAOA<char2, uchar2, double2, char2, op>, FMXImpl_AOAOA<char3, uchar3, double3, char3, op>, FMXImpl_AOAOA<char4, uchar4, double4, char4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, double, ushort, op>, FMXImpl_AOAOA<char2, uchar2, double2, ushort2, op>, FMXImpl_AOAOA<char3, uchar3, double3, ushort3, op>, FMXImpl_AOAOA<char4, uchar4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, double, short, op>, FMXImpl_AOAOA<char2, uchar2, double2, short2, op>, FMXImpl_AOAOA<char3, uchar3, double3, short3, op>, FMXImpl_AOAOA<char4, uchar4, double4, short4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, double, int, op>, FMXImpl_AOAOA<char2, uchar2, double2, int2, op>, FMXImpl_AOAOA<char3, uchar3, double3, int3, op>, FMXImpl_AOAOA<char4, uchar4, double4, int4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, double, float, op>, FMXImpl_AOAOA<char2, uchar2, double2, float2, op>, FMXImpl_AOAOA<char3, uchar3, double3, float3, op>, FMXImpl_AOAOA<char4, uchar4, double4, float4, op>  },
                    { FMXImpl_AOAOA<schar, uchar, double, double, op>, FMXImpl_AOAOA<char2, uchar2, double2, double2, op>, FMXImpl_AOAOA<char3, uchar3, double3, double3, op>, FMXImpl_AOAOA<char4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<schar, schar, uchar, uchar, op>, FMXImpl_AOAOA<char2, char2, uchar2, uchar2, op>, FMXImpl_AOAOA<char3, char3, uchar3, uchar3, op>, FMXImpl_AOAOA<char4, char4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, schar, uchar, schar, op>, FMXImpl_AOAOA<char2, char2, uchar2, char2, op>, FMXImpl_AOAOA<char3, char3, uchar3, char3, op>, FMXImpl_AOAOA<char4, char4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<schar, schar, uchar, ushort, op>, FMXImpl_AOAOA<char2, char2, uchar2, ushort2, op>, FMXImpl_AOAOA<char3, char3, uchar3, ushort3, op>, FMXImpl_AOAOA<char4, char4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, schar, uchar, short, op>, FMXImpl_AOAOA<char2, char2, uchar2, short2, op>, FMXImpl_AOAOA<char3, char3, uchar3, short3, op>, FMXImpl_AOAOA<char4, char4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<schar, schar, uchar, int, op>, FMXImpl_AOAOA<char2, char2, uchar2, int2, op>, FMXImpl_AOAOA<char3, char3, uchar3, int3, op>, FMXImpl_AOAOA<char4, char4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<schar, schar, uchar, float, op>, FMXImpl_AOAOA<char2, char2, uchar2, float2, op>, FMXImpl_AOAOA<char3, char3, uchar3, float3, op>, FMXImpl_AOAOA<char4, char4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<schar, schar, uchar, double, op>, FMXImpl_AOAOA<char2, char2, uchar2, double2, op>, FMXImpl_AOAOA<char3, char3, uchar3, double3, op>, FMXImpl_AOAOA<char4, char4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, schar, schar, uchar, op>, FMXImpl_AOAOA<char2, char2, char2, uchar2, op>, FMXImpl_AOAOA<char3, char3, char3, uchar3, op>, FMXImpl_AOAOA<char4, char4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, schar, schar, schar, op>, FMXImpl_AOAOA<char2, char2, char2, char2, op>, FMXImpl_AOAOA<char3, char3, char3, char3, op>, FMXImpl_AOAOA<char4, char4, char4, char4, op>  },
                    { FMXImpl_AOAOA<schar, schar, schar, ushort, op>, FMXImpl_AOAOA<char2, char2, char2, ushort2, op>, FMXImpl_AOAOA<char3, char3, char3, ushort3, op>, FMXImpl_AOAOA<char4, char4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, schar, schar, short, op>, FMXImpl_AOAOA<char2, char2, char2, short2, op>, FMXImpl_AOAOA<char3, char3, char3, short3, op>, FMXImpl_AOAOA<char4, char4, char4, short4, op>  },
                    { FMXImpl_AOAOA<schar, schar, schar, int, op>, FMXImpl_AOAOA<char2, char2, char2, int2, op>, FMXImpl_AOAOA<char3, char3, char3, int3, op>, FMXImpl_AOAOA<char4, char4, char4, int4, op>  },
                    { FMXImpl_AOAOA<schar, schar, schar, float, op>, FMXImpl_AOAOA<char2, char2, char2, float2, op>, FMXImpl_AOAOA<char3, char3, char3, float3, op>, FMXImpl_AOAOA<char4, char4, char4, float4, op>  },
                    { FMXImpl_AOAOA<schar, schar, schar, double, op>, FMXImpl_AOAOA<char2, char2, char2, double2, op>, FMXImpl_AOAOA<char3, char3, char3, double3, op>, FMXImpl_AOAOA<char4, char4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, schar, ushort, uchar, op>, FMXImpl_AOAOA<char2, char2, ushort2, uchar2, op>, FMXImpl_AOAOA<char3, char3, ushort3, uchar3, op>, FMXImpl_AOAOA<char4, char4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, schar, ushort, schar, op>, FMXImpl_AOAOA<char2, char2, ushort2, char2, op>, FMXImpl_AOAOA<char3, char3, ushort3, char3, op>, FMXImpl_AOAOA<char4, char4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<schar, schar, ushort, ushort, op>, FMXImpl_AOAOA<char2, char2, ushort2, ushort2, op>, FMXImpl_AOAOA<char3, char3, ushort3, ushort3, op>, FMXImpl_AOAOA<char4, char4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, schar, ushort, short, op>, FMXImpl_AOAOA<char2, char2, ushort2, short2, op>, FMXImpl_AOAOA<char3, char3, ushort3, short3, op>, FMXImpl_AOAOA<char4, char4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<schar, schar, ushort, int, op>, FMXImpl_AOAOA<char2, char2, ushort2, int2, op>, FMXImpl_AOAOA<char3, char3, ushort3, int3, op>, FMXImpl_AOAOA<char4, char4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<schar, schar, ushort, float, op>, FMXImpl_AOAOA<char2, char2, ushort2, float2, op>, FMXImpl_AOAOA<char3, char3, ushort3, float3, op>, FMXImpl_AOAOA<char4, char4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<schar, schar, ushort, double, op>, FMXImpl_AOAOA<char2, char2, ushort2, double2, op>, FMXImpl_AOAOA<char3, char3, ushort3, double3, op>, FMXImpl_AOAOA<char4, char4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, schar, short, uchar, op>, FMXImpl_AOAOA<char2, char2, short2, uchar2, op>, FMXImpl_AOAOA<char3, char3, short3, uchar3, op>, FMXImpl_AOAOA<char4, char4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, schar, short, schar, op>, FMXImpl_AOAOA<char2, char2, short2, char2, op>, FMXImpl_AOAOA<char3, char3, short3, char3, op>, FMXImpl_AOAOA<char4, char4, short4, char4, op>  },
                    { FMXImpl_AOAOA<schar, schar, short, ushort, op>, FMXImpl_AOAOA<char2, char2, short2, ushort2, op>, FMXImpl_AOAOA<char3, char3, short3, ushort3, op>, FMXImpl_AOAOA<char4, char4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, schar, short, short, op>, FMXImpl_AOAOA<char2, char2, short2, short2, op>, FMXImpl_AOAOA<char3, char3, short3, short3, op>, FMXImpl_AOAOA<char4, char4, short4, short4, op>  },
                    { FMXImpl_AOAOA<schar, schar, short, int, op>, FMXImpl_AOAOA<char2, char2, short2, int2, op>, FMXImpl_AOAOA<char3, char3, short3, int3, op>, FMXImpl_AOAOA<char4, char4, short4, int4, op>  },
                    { FMXImpl_AOAOA<schar, schar, short, float, op>, FMXImpl_AOAOA<char2, char2, short2, float2, op>, FMXImpl_AOAOA<char3, char3, short3, float3, op>, FMXImpl_AOAOA<char4, char4, short4, float4, op>  },
                    { FMXImpl_AOAOA<schar, schar, short, double, op>, FMXImpl_AOAOA<char2, char2, short2, double2, op>, FMXImpl_AOAOA<char3, char3, short3, double3, op>, FMXImpl_AOAOA<char4, char4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, schar, int, uchar, op>, FMXImpl_AOAOA<char2, char2, int2, uchar2, op>, FMXImpl_AOAOA<char3, char3, int3, uchar3, op>, FMXImpl_AOAOA<char4, char4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, schar, int, schar, op>, FMXImpl_AOAOA<char2, char2, int2, char2, op>, FMXImpl_AOAOA<char3, char3, int3, char3, op>, FMXImpl_AOAOA<char4, char4, int4, char4, op>  },
                    { FMXImpl_AOAOA<schar, schar, int, ushort, op>, FMXImpl_AOAOA<char2, char2, int2, ushort2, op>, FMXImpl_AOAOA<char3, char3, int3, ushort3, op>, FMXImpl_AOAOA<char4, char4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, schar, int, short, op>, FMXImpl_AOAOA<char2, char2, int2, short2, op>, FMXImpl_AOAOA<char3, char3, int3, short3, op>, FMXImpl_AOAOA<char4, char4, int4, short4, op>  },
                    { FMXImpl_AOAOA<schar, schar, int, int, op>, FMXImpl_AOAOA<char2, char2, int2, int2, op>, FMXImpl_AOAOA<char3, char3, int3, int3, op>, FMXImpl_AOAOA<char4, char4, int4, int4, op>  },
                    { FMXImpl_AOAOA<schar, schar, int, float, op>, FMXImpl_AOAOA<char2, char2, int2, float2, op>, FMXImpl_AOAOA<char3, char3, int3, float3, op>, FMXImpl_AOAOA<char4, char4, int4, float4, op>  },
                    { FMXImpl_AOAOA<schar, schar, int, double, op>, FMXImpl_AOAOA<char2, char2, int2, double2, op>, FMXImpl_AOAOA<char3, char3, int3, double3, op>, FMXImpl_AOAOA<char4, char4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, schar, float, uchar, op>, FMXImpl_AOAOA<char2, char2, float2, uchar2, op>, FMXImpl_AOAOA<char3, char3, float3, uchar3, op>, FMXImpl_AOAOA<char4, char4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, schar, float, schar, op>, FMXImpl_AOAOA<char2, char2, float2, char2, op>, FMXImpl_AOAOA<char3, char3, float3, char3, op>, FMXImpl_AOAOA<char4, char4, float4, char4, op>  },
                    { FMXImpl_AOAOA<schar, schar, float, ushort, op>, FMXImpl_AOAOA<char2, char2, float2, ushort2, op>, FMXImpl_AOAOA<char3, char3, float3, ushort3, op>, FMXImpl_AOAOA<char4, char4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, schar, float, short, op>, FMXImpl_AOAOA<char2, char2, float2, short2, op>, FMXImpl_AOAOA<char3, char3, float3, short3, op>, FMXImpl_AOAOA<char4, char4, float4, short4, op>  },
                    { FMXImpl_AOAOA<schar, schar, float, int, op>, FMXImpl_AOAOA<char2, char2, float2, int2, op>, FMXImpl_AOAOA<char3, char3, float3, int3, op>, FMXImpl_AOAOA<char4, char4, float4, int4, op>  },
                    { FMXImpl_AOAOA<schar, schar, float, float, op>, FMXImpl_AOAOA<char2, char2, float2, float2, op>, FMXImpl_AOAOA<char3, char3, float3, float3, op>, FMXImpl_AOAOA<char4, char4, float4, float4, op>  },
                    { FMXImpl_AOAOA<schar, schar, float, double, op>, FMXImpl_AOAOA<char2, char2, float2, double2, op>, FMXImpl_AOAOA<char3, char3, float3, double3, op>, FMXImpl_AOAOA<char4, char4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, schar, double, uchar, op>, FMXImpl_AOAOA<char2, char2, double2, uchar2, op>, FMXImpl_AOAOA<char3, char3, double3, uchar3, op>, FMXImpl_AOAOA<char4, char4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, schar, double, schar, op>, FMXImpl_AOAOA<char2, char2, double2, char2, op>, FMXImpl_AOAOA<char3, char3, double3, char3, op>, FMXImpl_AOAOA<char4, char4, double4, char4, op>  },
                    { FMXImpl_AOAOA<schar, schar, double, ushort, op>, FMXImpl_AOAOA<char2, char2, double2, ushort2, op>, FMXImpl_AOAOA<char3, char3, double3, ushort3, op>, FMXImpl_AOAOA<char4, char4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, schar, double, short, op>, FMXImpl_AOAOA<char2, char2, double2, short2, op>, FMXImpl_AOAOA<char3, char3, double3, short3, op>, FMXImpl_AOAOA<char4, char4, double4, short4, op>  },
                    { FMXImpl_AOAOA<schar, schar, double, int, op>, FMXImpl_AOAOA<char2, char2, double2, int2, op>, FMXImpl_AOAOA<char3, char3, double3, int3, op>, FMXImpl_AOAOA<char4, char4, double4, int4, op>  },
                    { FMXImpl_AOAOA<schar, schar, double, float, op>, FMXImpl_AOAOA<char2, char2, double2, float2, op>, FMXImpl_AOAOA<char3, char3, double3, float3, op>, FMXImpl_AOAOA<char4, char4, double4, float4, op>  },
                    { FMXImpl_AOAOA<schar, schar, double, double, op>, FMXImpl_AOAOA<char2, char2, double2, double2, op>, FMXImpl_AOAOA<char3, char3, double3, double3, op>, FMXImpl_AOAOA<char4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<schar, ushort, uchar, uchar, op>, FMXImpl_AOAOA<char2, ushort2, uchar2, uchar2, op>, FMXImpl_AOAOA<char3, ushort3, uchar3, uchar3, op>, FMXImpl_AOAOA<char4, ushort4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, uchar, schar, op>, FMXImpl_AOAOA<char2, ushort2, uchar2, char2, op>, FMXImpl_AOAOA<char3, ushort3, uchar3, char3, op>, FMXImpl_AOAOA<char4, ushort4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, uchar, ushort, op>, FMXImpl_AOAOA<char2, ushort2, uchar2, ushort2, op>, FMXImpl_AOAOA<char3, ushort3, uchar3, ushort3, op>, FMXImpl_AOAOA<char4, ushort4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, uchar, short, op>, FMXImpl_AOAOA<char2, ushort2, uchar2, short2, op>, FMXImpl_AOAOA<char3, ushort3, uchar3, short3, op>, FMXImpl_AOAOA<char4, ushort4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, uchar, int, op>, FMXImpl_AOAOA<char2, ushort2, uchar2, int2, op>, FMXImpl_AOAOA<char3, ushort3, uchar3, int3, op>, FMXImpl_AOAOA<char4, ushort4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, uchar, float, op>, FMXImpl_AOAOA<char2, ushort2, uchar2, float2, op>, FMXImpl_AOAOA<char3, ushort3, uchar3, float3, op>, FMXImpl_AOAOA<char4, ushort4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, uchar, double, op>, FMXImpl_AOAOA<char2, ushort2, uchar2, double2, op>, FMXImpl_AOAOA<char3, ushort3, uchar3, double3, op>, FMXImpl_AOAOA<char4, ushort4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, ushort, schar, uchar, op>, FMXImpl_AOAOA<char2, ushort2, char2, uchar2, op>, FMXImpl_AOAOA<char3, ushort3, char3, uchar3, op>, FMXImpl_AOAOA<char4, ushort4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, schar, schar, op>, FMXImpl_AOAOA<char2, ushort2, char2, char2, op>, FMXImpl_AOAOA<char3, ushort3, char3, char3, op>, FMXImpl_AOAOA<char4, ushort4, char4, char4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, schar, ushort, op>, FMXImpl_AOAOA<char2, ushort2, char2, ushort2, op>, FMXImpl_AOAOA<char3, ushort3, char3, ushort3, op>, FMXImpl_AOAOA<char4, ushort4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, schar, short, op>, FMXImpl_AOAOA<char2, ushort2, char2, short2, op>, FMXImpl_AOAOA<char3, ushort3, char3, short3, op>, FMXImpl_AOAOA<char4, ushort4, char4, short4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, schar, int, op>, FMXImpl_AOAOA<char2, ushort2, char2, int2, op>, FMXImpl_AOAOA<char3, ushort3, char3, int3, op>, FMXImpl_AOAOA<char4, ushort4, char4, int4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, schar, float, op>, FMXImpl_AOAOA<char2, ushort2, char2, float2, op>, FMXImpl_AOAOA<char3, ushort3, char3, float3, op>, FMXImpl_AOAOA<char4, ushort4, char4, float4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, schar, double, op>, FMXImpl_AOAOA<char2, ushort2, char2, double2, op>, FMXImpl_AOAOA<char3, ushort3, char3, double3, op>, FMXImpl_AOAOA<char4, ushort4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, ushort, ushort, uchar, op>, FMXImpl_AOAOA<char2, ushort2, ushort2, uchar2, op>, FMXImpl_AOAOA<char3, ushort3, ushort3, uchar3, op>, FMXImpl_AOAOA<char4, ushort4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, ushort, schar, op>, FMXImpl_AOAOA<char2, ushort2, ushort2, char2, op>, FMXImpl_AOAOA<char3, ushort3, ushort3, char3, op>, FMXImpl_AOAOA<char4, ushort4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, ushort, ushort, op>, FMXImpl_AOAOA<char2, ushort2, ushort2, ushort2, op>, FMXImpl_AOAOA<char3, ushort3, ushort3, ushort3, op>, FMXImpl_AOAOA<char4, ushort4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, ushort, short, op>, FMXImpl_AOAOA<char2, ushort2, ushort2, short2, op>, FMXImpl_AOAOA<char3, ushort3, ushort3, short3, op>, FMXImpl_AOAOA<char4, ushort4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, ushort, int, op>, FMXImpl_AOAOA<char2, ushort2, ushort2, int2, op>, FMXImpl_AOAOA<char3, ushort3, ushort3, int3, op>, FMXImpl_AOAOA<char4, ushort4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, ushort, float, op>, FMXImpl_AOAOA<char2, ushort2, ushort2, float2, op>, FMXImpl_AOAOA<char3, ushort3, ushort3, float3, op>, FMXImpl_AOAOA<char4, ushort4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, ushort, double, op>, FMXImpl_AOAOA<char2, ushort2, ushort2, double2, op>, FMXImpl_AOAOA<char3, ushort3, ushort3, double3, op>, FMXImpl_AOAOA<char4, ushort4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, ushort, short, uchar, op>, FMXImpl_AOAOA<char2, ushort2, short2, uchar2, op>, FMXImpl_AOAOA<char3, ushort3, short3, uchar3, op>, FMXImpl_AOAOA<char4, ushort4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, short, schar, op>, FMXImpl_AOAOA<char2, ushort2, short2, char2, op>, FMXImpl_AOAOA<char3, ushort3, short3, char3, op>, FMXImpl_AOAOA<char4, ushort4, short4, char4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, short, ushort, op>, FMXImpl_AOAOA<char2, ushort2, short2, ushort2, op>, FMXImpl_AOAOA<char3, ushort3, short3, ushort3, op>, FMXImpl_AOAOA<char4, ushort4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, short, short, op>, FMXImpl_AOAOA<char2, ushort2, short2, short2, op>, FMXImpl_AOAOA<char3, ushort3, short3, short3, op>, FMXImpl_AOAOA<char4, ushort4, short4, short4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, short, int, op>, FMXImpl_AOAOA<char2, ushort2, short2, int2, op>, FMXImpl_AOAOA<char3, ushort3, short3, int3, op>, FMXImpl_AOAOA<char4, ushort4, short4, int4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, short, float, op>, FMXImpl_AOAOA<char2, ushort2, short2, float2, op>, FMXImpl_AOAOA<char3, ushort3, short3, float3, op>, FMXImpl_AOAOA<char4, ushort4, short4, float4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, short, double, op>, FMXImpl_AOAOA<char2, ushort2, short2, double2, op>, FMXImpl_AOAOA<char3, ushort3, short3, double3, op>, FMXImpl_AOAOA<char4, ushort4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, ushort, int, uchar, op>, FMXImpl_AOAOA<char2, ushort2, int2, uchar2, op>, FMXImpl_AOAOA<char3, ushort3, int3, uchar3, op>, FMXImpl_AOAOA<char4, ushort4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, int, schar, op>, FMXImpl_AOAOA<char2, ushort2, int2, char2, op>, FMXImpl_AOAOA<char3, ushort3, int3, char3, op>, FMXImpl_AOAOA<char4, ushort4, int4, char4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, int, ushort, op>, FMXImpl_AOAOA<char2, ushort2, int2, ushort2, op>, FMXImpl_AOAOA<char3, ushort3, int3, ushort3, op>, FMXImpl_AOAOA<char4, ushort4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, int, short, op>, FMXImpl_AOAOA<char2, ushort2, int2, short2, op>, FMXImpl_AOAOA<char3, ushort3, int3, short3, op>, FMXImpl_AOAOA<char4, ushort4, int4, short4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, int, int, op>, FMXImpl_AOAOA<char2, ushort2, int2, int2, op>, FMXImpl_AOAOA<char3, ushort3, int3, int3, op>, FMXImpl_AOAOA<char4, ushort4, int4, int4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, int, float, op>, FMXImpl_AOAOA<char2, ushort2, int2, float2, op>, FMXImpl_AOAOA<char3, ushort3, int3, float3, op>, FMXImpl_AOAOA<char4, ushort4, int4, float4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, int, double, op>, FMXImpl_AOAOA<char2, ushort2, int2, double2, op>, FMXImpl_AOAOA<char3, ushort3, int3, double3, op>, FMXImpl_AOAOA<char4, ushort4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, ushort, float, uchar, op>, FMXImpl_AOAOA<char2, ushort2, float2, uchar2, op>, FMXImpl_AOAOA<char3, ushort3, float3, uchar3, op>, FMXImpl_AOAOA<char4, ushort4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, float, schar, op>, FMXImpl_AOAOA<char2, ushort2, float2, char2, op>, FMXImpl_AOAOA<char3, ushort3, float3, char3, op>, FMXImpl_AOAOA<char4, ushort4, float4, char4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, float, ushort, op>, FMXImpl_AOAOA<char2, ushort2, float2, ushort2, op>, FMXImpl_AOAOA<char3, ushort3, float3, ushort3, op>, FMXImpl_AOAOA<char4, ushort4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, float, short, op>, FMXImpl_AOAOA<char2, ushort2, float2, short2, op>, FMXImpl_AOAOA<char3, ushort3, float3, short3, op>, FMXImpl_AOAOA<char4, ushort4, float4, short4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, float, int, op>, FMXImpl_AOAOA<char2, ushort2, float2, int2, op>, FMXImpl_AOAOA<char3, ushort3, float3, int3, op>, FMXImpl_AOAOA<char4, ushort4, float4, int4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, float, float, op>, FMXImpl_AOAOA<char2, ushort2, float2, float2, op>, FMXImpl_AOAOA<char3, ushort3, float3, float3, op>, FMXImpl_AOAOA<char4, ushort4, float4, float4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, float, double, op>, FMXImpl_AOAOA<char2, ushort2, float2, double2, op>, FMXImpl_AOAOA<char3, ushort3, float3, double3, op>, FMXImpl_AOAOA<char4, ushort4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, ushort, double, uchar, op>, FMXImpl_AOAOA<char2, ushort2, double2, uchar2, op>, FMXImpl_AOAOA<char3, ushort3, double3, uchar3, op>, FMXImpl_AOAOA<char4, ushort4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, double, schar, op>, FMXImpl_AOAOA<char2, ushort2, double2, char2, op>, FMXImpl_AOAOA<char3, ushort3, double3, char3, op>, FMXImpl_AOAOA<char4, ushort4, double4, char4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, double, ushort, op>, FMXImpl_AOAOA<char2, ushort2, double2, ushort2, op>, FMXImpl_AOAOA<char3, ushort3, double3, ushort3, op>, FMXImpl_AOAOA<char4, ushort4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, double, short, op>, FMXImpl_AOAOA<char2, ushort2, double2, short2, op>, FMXImpl_AOAOA<char3, ushort3, double3, short3, op>, FMXImpl_AOAOA<char4, ushort4, double4, short4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, double, int, op>, FMXImpl_AOAOA<char2, ushort2, double2, int2, op>, FMXImpl_AOAOA<char3, ushort3, double3, int3, op>, FMXImpl_AOAOA<char4, ushort4, double4, int4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, double, float, op>, FMXImpl_AOAOA<char2, ushort2, double2, float2, op>, FMXImpl_AOAOA<char3, ushort3, double3, float3, op>, FMXImpl_AOAOA<char4, ushort4, double4, float4, op>  },
                    { FMXImpl_AOAOA<schar, ushort, double, double, op>, FMXImpl_AOAOA<char2, ushort2, double2, double2, op>, FMXImpl_AOAOA<char3, ushort3, double3, double3, op>, FMXImpl_AOAOA<char4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<schar, short, uchar, uchar, op>, FMXImpl_AOAOA<char2, short2, uchar2, uchar2, op>, FMXImpl_AOAOA<char3, short3, uchar3, uchar3, op>, FMXImpl_AOAOA<char4, short4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, short, uchar, schar, op>, FMXImpl_AOAOA<char2, short2, uchar2, char2, op>, FMXImpl_AOAOA<char3, short3, uchar3, char3, op>, FMXImpl_AOAOA<char4, short4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<schar, short, uchar, ushort, op>, FMXImpl_AOAOA<char2, short2, uchar2, ushort2, op>, FMXImpl_AOAOA<char3, short3, uchar3, ushort3, op>, FMXImpl_AOAOA<char4, short4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, short, uchar, short, op>, FMXImpl_AOAOA<char2, short2, uchar2, short2, op>, FMXImpl_AOAOA<char3, short3, uchar3, short3, op>, FMXImpl_AOAOA<char4, short4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<schar, short, uchar, int, op>, FMXImpl_AOAOA<char2, short2, uchar2, int2, op>, FMXImpl_AOAOA<char3, short3, uchar3, int3, op>, FMXImpl_AOAOA<char4, short4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<schar, short, uchar, float, op>, FMXImpl_AOAOA<char2, short2, uchar2, float2, op>, FMXImpl_AOAOA<char3, short3, uchar3, float3, op>, FMXImpl_AOAOA<char4, short4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<schar, short, uchar, double, op>, FMXImpl_AOAOA<char2, short2, uchar2, double2, op>, FMXImpl_AOAOA<char3, short3, uchar3, double3, op>, FMXImpl_AOAOA<char4, short4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, short, schar, uchar, op>, FMXImpl_AOAOA<char2, short2, char2, uchar2, op>, FMXImpl_AOAOA<char3, short3, char3, uchar3, op>, FMXImpl_AOAOA<char4, short4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, short, schar, schar, op>, FMXImpl_AOAOA<char2, short2, char2, char2, op>, FMXImpl_AOAOA<char3, short3, char3, char3, op>, FMXImpl_AOAOA<char4, short4, char4, char4, op>  },
                    { FMXImpl_AOAOA<schar, short, schar, ushort, op>, FMXImpl_AOAOA<char2, short2, char2, ushort2, op>, FMXImpl_AOAOA<char3, short3, char3, ushort3, op>, FMXImpl_AOAOA<char4, short4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, short, schar, short, op>, FMXImpl_AOAOA<char2, short2, char2, short2, op>, FMXImpl_AOAOA<char3, short3, char3, short3, op>, FMXImpl_AOAOA<char4, short4, char4, short4, op>  },
                    { FMXImpl_AOAOA<schar, short, schar, int, op>, FMXImpl_AOAOA<char2, short2, char2, int2, op>, FMXImpl_AOAOA<char3, short3, char3, int3, op>, FMXImpl_AOAOA<char4, short4, char4, int4, op>  },
                    { FMXImpl_AOAOA<schar, short, schar, float, op>, FMXImpl_AOAOA<char2, short2, char2, float2, op>, FMXImpl_AOAOA<char3, short3, char3, float3, op>, FMXImpl_AOAOA<char4, short4, char4, float4, op>  },
                    { FMXImpl_AOAOA<schar, short, schar, double, op>, FMXImpl_AOAOA<char2, short2, char2, double2, op>, FMXImpl_AOAOA<char3, short3, char3, double3, op>, FMXImpl_AOAOA<char4, short4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, short, ushort, uchar, op>, FMXImpl_AOAOA<char2, short2, ushort2, uchar2, op>, FMXImpl_AOAOA<char3, short3, ushort3, uchar3, op>, FMXImpl_AOAOA<char4, short4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, short, ushort, schar, op>, FMXImpl_AOAOA<char2, short2, ushort2, char2, op>, FMXImpl_AOAOA<char3, short3, ushort3, char3, op>, FMXImpl_AOAOA<char4, short4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<schar, short, ushort, ushort, op>, FMXImpl_AOAOA<char2, short2, ushort2, ushort2, op>, FMXImpl_AOAOA<char3, short3, ushort3, ushort3, op>, FMXImpl_AOAOA<char4, short4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, short, ushort, short, op>, FMXImpl_AOAOA<char2, short2, ushort2, short2, op>, FMXImpl_AOAOA<char3, short3, ushort3, short3, op>, FMXImpl_AOAOA<char4, short4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<schar, short, ushort, int, op>, FMXImpl_AOAOA<char2, short2, ushort2, int2, op>, FMXImpl_AOAOA<char3, short3, ushort3, int3, op>, FMXImpl_AOAOA<char4, short4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<schar, short, ushort, float, op>, FMXImpl_AOAOA<char2, short2, ushort2, float2, op>, FMXImpl_AOAOA<char3, short3, ushort3, float3, op>, FMXImpl_AOAOA<char4, short4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<schar, short, ushort, double, op>, FMXImpl_AOAOA<char2, short2, ushort2, double2, op>, FMXImpl_AOAOA<char3, short3, ushort3, double3, op>, FMXImpl_AOAOA<char4, short4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, short, short, uchar, op>, FMXImpl_AOAOA<char2, short2, short2, uchar2, op>, FMXImpl_AOAOA<char3, short3, short3, uchar3, op>, FMXImpl_AOAOA<char4, short4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, short, short, schar, op>, FMXImpl_AOAOA<char2, short2, short2, char2, op>, FMXImpl_AOAOA<char3, short3, short3, char3, op>, FMXImpl_AOAOA<char4, short4, short4, char4, op>  },
                    { FMXImpl_AOAOA<schar, short, short, ushort, op>, FMXImpl_AOAOA<char2, short2, short2, ushort2, op>, FMXImpl_AOAOA<char3, short3, short3, ushort3, op>, FMXImpl_AOAOA<char4, short4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, short, short, short, op>, FMXImpl_AOAOA<char2, short2, short2, short2, op>, FMXImpl_AOAOA<char3, short3, short3, short3, op>, FMXImpl_AOAOA<char4, short4, short4, short4, op>  },
                    { FMXImpl_AOAOA<schar, short, short, int, op>, FMXImpl_AOAOA<char2, short2, short2, int2, op>, FMXImpl_AOAOA<char3, short3, short3, int3, op>, FMXImpl_AOAOA<char4, short4, short4, int4, op>  },
                    { FMXImpl_AOAOA<schar, short, short, float, op>, FMXImpl_AOAOA<char2, short2, short2, float2, op>, FMXImpl_AOAOA<char3, short3, short3, float3, op>, FMXImpl_AOAOA<char4, short4, short4, float4, op>  },
                    { FMXImpl_AOAOA<schar, short, short, double, op>, FMXImpl_AOAOA<char2, short2, short2, double2, op>, FMXImpl_AOAOA<char3, short3, short3, double3, op>, FMXImpl_AOAOA<char4, short4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, short, int, uchar, op>, FMXImpl_AOAOA<char2, short2, int2, uchar2, op>, FMXImpl_AOAOA<char3, short3, int3, uchar3, op>, FMXImpl_AOAOA<char4, short4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, short, int, schar, op>, FMXImpl_AOAOA<char2, short2, int2, char2, op>, FMXImpl_AOAOA<char3, short3, int3, char3, op>, FMXImpl_AOAOA<char4, short4, int4, char4, op>  },
                    { FMXImpl_AOAOA<schar, short, int, ushort, op>, FMXImpl_AOAOA<char2, short2, int2, ushort2, op>, FMXImpl_AOAOA<char3, short3, int3, ushort3, op>, FMXImpl_AOAOA<char4, short4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, short, int, short, op>, FMXImpl_AOAOA<char2, short2, int2, short2, op>, FMXImpl_AOAOA<char3, short3, int3, short3, op>, FMXImpl_AOAOA<char4, short4, int4, short4, op>  },
                    { FMXImpl_AOAOA<schar, short, int, int, op>, FMXImpl_AOAOA<char2, short2, int2, int2, op>, FMXImpl_AOAOA<char3, short3, int3, int3, op>, FMXImpl_AOAOA<char4, short4, int4, int4, op>  },
                    { FMXImpl_AOAOA<schar, short, int, float, op>, FMXImpl_AOAOA<char2, short2, int2, float2, op>, FMXImpl_AOAOA<char3, short3, int3, float3, op>, FMXImpl_AOAOA<char4, short4, int4, float4, op>  },
                    { FMXImpl_AOAOA<schar, short, int, double, op>, FMXImpl_AOAOA<char2, short2, int2, double2, op>, FMXImpl_AOAOA<char3, short3, int3, double3, op>, FMXImpl_AOAOA<char4, short4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, short, float, uchar, op>, FMXImpl_AOAOA<char2, short2, float2, uchar2, op>, FMXImpl_AOAOA<char3, short3, float3, uchar3, op>, FMXImpl_AOAOA<char4, short4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, short, float, schar, op>, FMXImpl_AOAOA<char2, short2, float2, char2, op>, FMXImpl_AOAOA<char3, short3, float3, char3, op>, FMXImpl_AOAOA<char4, short4, float4, char4, op>  },
                    { FMXImpl_AOAOA<schar, short, float, ushort, op>, FMXImpl_AOAOA<char2, short2, float2, ushort2, op>, FMXImpl_AOAOA<char3, short3, float3, ushort3, op>, FMXImpl_AOAOA<char4, short4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, short, float, short, op>, FMXImpl_AOAOA<char2, short2, float2, short2, op>, FMXImpl_AOAOA<char3, short3, float3, short3, op>, FMXImpl_AOAOA<char4, short4, float4, short4, op>  },
                    { FMXImpl_AOAOA<schar, short, float, int, op>, FMXImpl_AOAOA<char2, short2, float2, int2, op>, FMXImpl_AOAOA<char3, short3, float3, int3, op>, FMXImpl_AOAOA<char4, short4, float4, int4, op>  },
                    { FMXImpl_AOAOA<schar, short, float, float, op>, FMXImpl_AOAOA<char2, short2, float2, float2, op>, FMXImpl_AOAOA<char3, short3, float3, float3, op>, FMXImpl_AOAOA<char4, short4, float4, float4, op>  },
                    { FMXImpl_AOAOA<schar, short, float, double, op>, FMXImpl_AOAOA<char2, short2, float2, double2, op>, FMXImpl_AOAOA<char3, short3, float3, double3, op>, FMXImpl_AOAOA<char4, short4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, short, double, uchar, op>, FMXImpl_AOAOA<char2, short2, double2, uchar2, op>, FMXImpl_AOAOA<char3, short3, double3, uchar3, op>, FMXImpl_AOAOA<char4, short4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, short, double, schar, op>, FMXImpl_AOAOA<char2, short2, double2, char2, op>, FMXImpl_AOAOA<char3, short3, double3, char3, op>, FMXImpl_AOAOA<char4, short4, double4, char4, op>  },
                    { FMXImpl_AOAOA<schar, short, double, ushort, op>, FMXImpl_AOAOA<char2, short2, double2, ushort2, op>, FMXImpl_AOAOA<char3, short3, double3, ushort3, op>, FMXImpl_AOAOA<char4, short4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, short, double, short, op>, FMXImpl_AOAOA<char2, short2, double2, short2, op>, FMXImpl_AOAOA<char3, short3, double3, short3, op>, FMXImpl_AOAOA<char4, short4, double4, short4, op>  },
                    { FMXImpl_AOAOA<schar, short, double, int, op>, FMXImpl_AOAOA<char2, short2, double2, int2, op>, FMXImpl_AOAOA<char3, short3, double3, int3, op>, FMXImpl_AOAOA<char4, short4, double4, int4, op>  },
                    { FMXImpl_AOAOA<schar, short, double, float, op>, FMXImpl_AOAOA<char2, short2, double2, float2, op>, FMXImpl_AOAOA<char3, short3, double3, float3, op>, FMXImpl_AOAOA<char4, short4, double4, float4, op>  },
                    { FMXImpl_AOAOA<schar, short, double, double, op>, FMXImpl_AOAOA<char2, short2, double2, double2, op>, FMXImpl_AOAOA<char3, short3, double3, double3, op>, FMXImpl_AOAOA<char4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<schar, int, uchar, uchar, op>, FMXImpl_AOAOA<char2, int2, uchar2, uchar2, op>, FMXImpl_AOAOA<char3, int3, uchar3, uchar3, op>, FMXImpl_AOAOA<char4, int4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, int, uchar, schar, op>, FMXImpl_AOAOA<char2, int2, uchar2, char2, op>, FMXImpl_AOAOA<char3, int3, uchar3, char3, op>, FMXImpl_AOAOA<char4, int4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<schar, int, uchar, ushort, op>, FMXImpl_AOAOA<char2, int2, uchar2, ushort2, op>, FMXImpl_AOAOA<char3, int3, uchar3, ushort3, op>, FMXImpl_AOAOA<char4, int4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, int, uchar, short, op>, FMXImpl_AOAOA<char2, int2, uchar2, short2, op>, FMXImpl_AOAOA<char3, int3, uchar3, short3, op>, FMXImpl_AOAOA<char4, int4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<schar, int, uchar, int, op>, FMXImpl_AOAOA<char2, int2, uchar2, int2, op>, FMXImpl_AOAOA<char3, int3, uchar3, int3, op>, FMXImpl_AOAOA<char4, int4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<schar, int, uchar, float, op>, FMXImpl_AOAOA<char2, int2, uchar2, float2, op>, FMXImpl_AOAOA<char3, int3, uchar3, float3, op>, FMXImpl_AOAOA<char4, int4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<schar, int, uchar, double, op>, FMXImpl_AOAOA<char2, int2, uchar2, double2, op>, FMXImpl_AOAOA<char3, int3, uchar3, double3, op>, FMXImpl_AOAOA<char4, int4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, int, schar, uchar, op>, FMXImpl_AOAOA<char2, int2, char2, uchar2, op>, FMXImpl_AOAOA<char3, int3, char3, uchar3, op>, FMXImpl_AOAOA<char4, int4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, int, schar, schar, op>, FMXImpl_AOAOA<char2, int2, char2, char2, op>, FMXImpl_AOAOA<char3, int3, char3, char3, op>, FMXImpl_AOAOA<char4, int4, char4, char4, op>  },
                    { FMXImpl_AOAOA<schar, int, schar, ushort, op>, FMXImpl_AOAOA<char2, int2, char2, ushort2, op>, FMXImpl_AOAOA<char3, int3, char3, ushort3, op>, FMXImpl_AOAOA<char4, int4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, int, schar, short, op>, FMXImpl_AOAOA<char2, int2, char2, short2, op>, FMXImpl_AOAOA<char3, int3, char3, short3, op>, FMXImpl_AOAOA<char4, int4, char4, short4, op>  },
                    { FMXImpl_AOAOA<schar, int, schar, int, op>, FMXImpl_AOAOA<char2, int2, char2, int2, op>, FMXImpl_AOAOA<char3, int3, char3, int3, op>, FMXImpl_AOAOA<char4, int4, char4, int4, op>  },
                    { FMXImpl_AOAOA<schar, int, schar, float, op>, FMXImpl_AOAOA<char2, int2, char2, float2, op>, FMXImpl_AOAOA<char3, int3, char3, float3, op>, FMXImpl_AOAOA<char4, int4, char4, float4, op>  },
                    { FMXImpl_AOAOA<schar, int, schar, double, op>, FMXImpl_AOAOA<char2, int2, char2, double2, op>, FMXImpl_AOAOA<char3, int3, char3, double3, op>, FMXImpl_AOAOA<char4, int4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, int, ushort, uchar, op>, FMXImpl_AOAOA<char2, int2, ushort2, uchar2, op>, FMXImpl_AOAOA<char3, int3, ushort3, uchar3, op>, FMXImpl_AOAOA<char4, int4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, int, ushort, schar, op>, FMXImpl_AOAOA<char2, int2, ushort2, char2, op>, FMXImpl_AOAOA<char3, int3, ushort3, char3, op>, FMXImpl_AOAOA<char4, int4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<schar, int, ushort, ushort, op>, FMXImpl_AOAOA<char2, int2, ushort2, ushort2, op>, FMXImpl_AOAOA<char3, int3, ushort3, ushort3, op>, FMXImpl_AOAOA<char4, int4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, int, ushort, short, op>, FMXImpl_AOAOA<char2, int2, ushort2, short2, op>, FMXImpl_AOAOA<char3, int3, ushort3, short3, op>, FMXImpl_AOAOA<char4, int4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<schar, int, ushort, int, op>, FMXImpl_AOAOA<char2, int2, ushort2, int2, op>, FMXImpl_AOAOA<char3, int3, ushort3, int3, op>, FMXImpl_AOAOA<char4, int4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<schar, int, ushort, float, op>, FMXImpl_AOAOA<char2, int2, ushort2, float2, op>, FMXImpl_AOAOA<char3, int3, ushort3, float3, op>, FMXImpl_AOAOA<char4, int4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<schar, int, ushort, double, op>, FMXImpl_AOAOA<char2, int2, ushort2, double2, op>, FMXImpl_AOAOA<char3, int3, ushort3, double3, op>, FMXImpl_AOAOA<char4, int4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, int, short, uchar, op>, FMXImpl_AOAOA<char2, int2, short2, uchar2, op>, FMXImpl_AOAOA<char3, int3, short3, uchar3, op>, FMXImpl_AOAOA<char4, int4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, int, short, schar, op>, FMXImpl_AOAOA<char2, int2, short2, char2, op>, FMXImpl_AOAOA<char3, int3, short3, char3, op>, FMXImpl_AOAOA<char4, int4, short4, char4, op>  },
                    { FMXImpl_AOAOA<schar, int, short, ushort, op>, FMXImpl_AOAOA<char2, int2, short2, ushort2, op>, FMXImpl_AOAOA<char3, int3, short3, ushort3, op>, FMXImpl_AOAOA<char4, int4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, int, short, short, op>, FMXImpl_AOAOA<char2, int2, short2, short2, op>, FMXImpl_AOAOA<char3, int3, short3, short3, op>, FMXImpl_AOAOA<char4, int4, short4, short4, op>  },
                    { FMXImpl_AOAOA<schar, int, short, int, op>, FMXImpl_AOAOA<char2, int2, short2, int2, op>, FMXImpl_AOAOA<char3, int3, short3, int3, op>, FMXImpl_AOAOA<char4, int4, short4, int4, op>  },
                    { FMXImpl_AOAOA<schar, int, short, float, op>, FMXImpl_AOAOA<char2, int2, short2, float2, op>, FMXImpl_AOAOA<char3, int3, short3, float3, op>, FMXImpl_AOAOA<char4, int4, short4, float4, op>  },
                    { FMXImpl_AOAOA<schar, int, short, double, op>, FMXImpl_AOAOA<char2, int2, short2, double2, op>, FMXImpl_AOAOA<char3, int3, short3, double3, op>, FMXImpl_AOAOA<char4, int4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, int, int, uchar, op>, FMXImpl_AOAOA<char2, int2, int2, uchar2, op>, FMXImpl_AOAOA<char3, int3, int3, uchar3, op>, FMXImpl_AOAOA<char4, int4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, int, int, schar, op>, FMXImpl_AOAOA<char2, int2, int2, char2, op>, FMXImpl_AOAOA<char3, int3, int3, char3, op>, FMXImpl_AOAOA<char4, int4, int4, char4, op>  },
                    { FMXImpl_AOAOA<schar, int, int, ushort, op>, FMXImpl_AOAOA<char2, int2, int2, ushort2, op>, FMXImpl_AOAOA<char3, int3, int3, ushort3, op>, FMXImpl_AOAOA<char4, int4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, int, int, short, op>, FMXImpl_AOAOA<char2, int2, int2, short2, op>, FMXImpl_AOAOA<char3, int3, int3, short3, op>, FMXImpl_AOAOA<char4, int4, int4, short4, op>  },
                    { FMXImpl_AOAOA<schar, int, int, int, op>, FMXImpl_AOAOA<char2, int2, int2, int2, op>, FMXImpl_AOAOA<char3, int3, int3, int3, op>, FMXImpl_AOAOA<char4, int4, int4, int4, op>  },
                    { FMXImpl_AOAOA<schar, int, int, float, op>, FMXImpl_AOAOA<char2, int2, int2, float2, op>, FMXImpl_AOAOA<char3, int3, int3, float3, op>, FMXImpl_AOAOA<char4, int4, int4, float4, op>  },
                    { FMXImpl_AOAOA<schar, int, int, double, op>, FMXImpl_AOAOA<char2, int2, int2, double2, op>, FMXImpl_AOAOA<char3, int3, int3, double3, op>, FMXImpl_AOAOA<char4, int4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, int, float, uchar, op>, FMXImpl_AOAOA<char2, int2, float2, uchar2, op>, FMXImpl_AOAOA<char3, int3, float3, uchar3, op>, FMXImpl_AOAOA<char4, int4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, int, float, schar, op>, FMXImpl_AOAOA<char2, int2, float2, char2, op>, FMXImpl_AOAOA<char3, int3, float3, char3, op>, FMXImpl_AOAOA<char4, int4, float4, char4, op>  },
                    { FMXImpl_AOAOA<schar, int, float, ushort, op>, FMXImpl_AOAOA<char2, int2, float2, ushort2, op>, FMXImpl_AOAOA<char3, int3, float3, ushort3, op>, FMXImpl_AOAOA<char4, int4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, int, float, short, op>, FMXImpl_AOAOA<char2, int2, float2, short2, op>, FMXImpl_AOAOA<char3, int3, float3, short3, op>, FMXImpl_AOAOA<char4, int4, float4, short4, op>  },
                    { FMXImpl_AOAOA<schar, int, float, int, op>, FMXImpl_AOAOA<char2, int2, float2, int2, op>, FMXImpl_AOAOA<char3, int3, float3, int3, op>, FMXImpl_AOAOA<char4, int4, float4, int4, op>  },
                    { FMXImpl_AOAOA<schar, int, float, float, op>, FMXImpl_AOAOA<char2, int2, float2, float2, op>, FMXImpl_AOAOA<char3, int3, float3, float3, op>, FMXImpl_AOAOA<char4, int4, float4, float4, op>  },
                    { FMXImpl_AOAOA<schar, int, float, double, op>, FMXImpl_AOAOA<char2, int2, float2, double2, op>, FMXImpl_AOAOA<char3, int3, float3, double3, op>, FMXImpl_AOAOA<char4, int4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, int, double, uchar, op>, FMXImpl_AOAOA<char2, int2, double2, uchar2, op>, FMXImpl_AOAOA<char3, int3, double3, uchar3, op>, FMXImpl_AOAOA<char4, int4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, int, double, schar, op>, FMXImpl_AOAOA<char2, int2, double2, char2, op>, FMXImpl_AOAOA<char3, int3, double3, char3, op>, FMXImpl_AOAOA<char4, int4, double4, char4, op>  },
                    { FMXImpl_AOAOA<schar, int, double, ushort, op>, FMXImpl_AOAOA<char2, int2, double2, ushort2, op>, FMXImpl_AOAOA<char3, int3, double3, ushort3, op>, FMXImpl_AOAOA<char4, int4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, int, double, short, op>, FMXImpl_AOAOA<char2, int2, double2, short2, op>, FMXImpl_AOAOA<char3, int3, double3, short3, op>, FMXImpl_AOAOA<char4, int4, double4, short4, op>  },
                    { FMXImpl_AOAOA<schar, int, double, int, op>, FMXImpl_AOAOA<char2, int2, double2, int2, op>, FMXImpl_AOAOA<char3, int3, double3, int3, op>, FMXImpl_AOAOA<char4, int4, double4, int4, op>  },
                    { FMXImpl_AOAOA<schar, int, double, float, op>, FMXImpl_AOAOA<char2, int2, double2, float2, op>, FMXImpl_AOAOA<char3, int3, double3, float3, op>, FMXImpl_AOAOA<char4, int4, double4, float4, op>  },
                    { FMXImpl_AOAOA<schar, int, double, double, op>, FMXImpl_AOAOA<char2, int2, double2, double2, op>, FMXImpl_AOAOA<char3, int3, double3, double3, op>, FMXImpl_AOAOA<char4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<schar, float, uchar, uchar, op>, FMXImpl_AOAOA<char2, float2, uchar2, uchar2, op>, FMXImpl_AOAOA<char3, float3, uchar3, uchar3, op>, FMXImpl_AOAOA<char4, float4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, float, uchar, schar, op>, FMXImpl_AOAOA<char2, float2, uchar2, char2, op>, FMXImpl_AOAOA<char3, float3, uchar3, char3, op>, FMXImpl_AOAOA<char4, float4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<schar, float, uchar, ushort, op>, FMXImpl_AOAOA<char2, float2, uchar2, ushort2, op>, FMXImpl_AOAOA<char3, float3, uchar3, ushort3, op>, FMXImpl_AOAOA<char4, float4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, float, uchar, short, op>, FMXImpl_AOAOA<char2, float2, uchar2, short2, op>, FMXImpl_AOAOA<char3, float3, uchar3, short3, op>, FMXImpl_AOAOA<char4, float4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<schar, float, uchar, int, op>, FMXImpl_AOAOA<char2, float2, uchar2, int2, op>, FMXImpl_AOAOA<char3, float3, uchar3, int3, op>, FMXImpl_AOAOA<char4, float4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<schar, float, uchar, float, op>, FMXImpl_AOAOA<char2, float2, uchar2, float2, op>, FMXImpl_AOAOA<char3, float3, uchar3, float3, op>, FMXImpl_AOAOA<char4, float4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<schar, float, uchar, double, op>, FMXImpl_AOAOA<char2, float2, uchar2, double2, op>, FMXImpl_AOAOA<char3, float3, uchar3, double3, op>, FMXImpl_AOAOA<char4, float4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, float, schar, uchar, op>, FMXImpl_AOAOA<char2, float2, char2, uchar2, op>, FMXImpl_AOAOA<char3, float3, char3, uchar3, op>, FMXImpl_AOAOA<char4, float4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, float, schar, schar, op>, FMXImpl_AOAOA<char2, float2, char2, char2, op>, FMXImpl_AOAOA<char3, float3, char3, char3, op>, FMXImpl_AOAOA<char4, float4, char4, char4, op>  },
                    { FMXImpl_AOAOA<schar, float, schar, ushort, op>, FMXImpl_AOAOA<char2, float2, char2, ushort2, op>, FMXImpl_AOAOA<char3, float3, char3, ushort3, op>, FMXImpl_AOAOA<char4, float4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, float, schar, short, op>, FMXImpl_AOAOA<char2, float2, char2, short2, op>, FMXImpl_AOAOA<char3, float3, char3, short3, op>, FMXImpl_AOAOA<char4, float4, char4, short4, op>  },
                    { FMXImpl_AOAOA<schar, float, schar, int, op>, FMXImpl_AOAOA<char2, float2, char2, int2, op>, FMXImpl_AOAOA<char3, float3, char3, int3, op>, FMXImpl_AOAOA<char4, float4, char4, int4, op>  },
                    { FMXImpl_AOAOA<schar, float, schar, float, op>, FMXImpl_AOAOA<char2, float2, char2, float2, op>, FMXImpl_AOAOA<char3, float3, char3, float3, op>, FMXImpl_AOAOA<char4, float4, char4, float4, op>  },
                    { FMXImpl_AOAOA<schar, float, schar, double, op>, FMXImpl_AOAOA<char2, float2, char2, double2, op>, FMXImpl_AOAOA<char3, float3, char3, double3, op>, FMXImpl_AOAOA<char4, float4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, float, ushort, uchar, op>, FMXImpl_AOAOA<char2, float2, ushort2, uchar2, op>, FMXImpl_AOAOA<char3, float3, ushort3, uchar3, op>, FMXImpl_AOAOA<char4, float4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, float, ushort, schar, op>, FMXImpl_AOAOA<char2, float2, ushort2, char2, op>, FMXImpl_AOAOA<char3, float3, ushort3, char3, op>, FMXImpl_AOAOA<char4, float4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<schar, float, ushort, ushort, op>, FMXImpl_AOAOA<char2, float2, ushort2, ushort2, op>, FMXImpl_AOAOA<char3, float3, ushort3, ushort3, op>, FMXImpl_AOAOA<char4, float4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, float, ushort, short, op>, FMXImpl_AOAOA<char2, float2, ushort2, short2, op>, FMXImpl_AOAOA<char3, float3, ushort3, short3, op>, FMXImpl_AOAOA<char4, float4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<schar, float, ushort, int, op>, FMXImpl_AOAOA<char2, float2, ushort2, int2, op>, FMXImpl_AOAOA<char3, float3, ushort3, int3, op>, FMXImpl_AOAOA<char4, float4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<schar, float, ushort, float, op>, FMXImpl_AOAOA<char2, float2, ushort2, float2, op>, FMXImpl_AOAOA<char3, float3, ushort3, float3, op>, FMXImpl_AOAOA<char4, float4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<schar, float, ushort, double, op>, FMXImpl_AOAOA<char2, float2, ushort2, double2, op>, FMXImpl_AOAOA<char3, float3, ushort3, double3, op>, FMXImpl_AOAOA<char4, float4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, float, short, uchar, op>, FMXImpl_AOAOA<char2, float2, short2, uchar2, op>, FMXImpl_AOAOA<char3, float3, short3, uchar3, op>, FMXImpl_AOAOA<char4, float4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, float, short, schar, op>, FMXImpl_AOAOA<char2, float2, short2, char2, op>, FMXImpl_AOAOA<char3, float3, short3, char3, op>, FMXImpl_AOAOA<char4, float4, short4, char4, op>  },
                    { FMXImpl_AOAOA<schar, float, short, ushort, op>, FMXImpl_AOAOA<char2, float2, short2, ushort2, op>, FMXImpl_AOAOA<char3, float3, short3, ushort3, op>, FMXImpl_AOAOA<char4, float4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, float, short, short, op>, FMXImpl_AOAOA<char2, float2, short2, short2, op>, FMXImpl_AOAOA<char3, float3, short3, short3, op>, FMXImpl_AOAOA<char4, float4, short4, short4, op>  },
                    { FMXImpl_AOAOA<schar, float, short, int, op>, FMXImpl_AOAOA<char2, float2, short2, int2, op>, FMXImpl_AOAOA<char3, float3, short3, int3, op>, FMXImpl_AOAOA<char4, float4, short4, int4, op>  },
                    { FMXImpl_AOAOA<schar, float, short, float, op>, FMXImpl_AOAOA<char2, float2, short2, float2, op>, FMXImpl_AOAOA<char3, float3, short3, float3, op>, FMXImpl_AOAOA<char4, float4, short4, float4, op>  },
                    { FMXImpl_AOAOA<schar, float, short, double, op>, FMXImpl_AOAOA<char2, float2, short2, double2, op>, FMXImpl_AOAOA<char3, float3, short3, double3, op>, FMXImpl_AOAOA<char4, float4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, float, int, uchar, op>, FMXImpl_AOAOA<char2, float2, int2, uchar2, op>, FMXImpl_AOAOA<char3, float3, int3, uchar3, op>, FMXImpl_AOAOA<char4, float4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, float, int, schar, op>, FMXImpl_AOAOA<char2, float2, int2, char2, op>, FMXImpl_AOAOA<char3, float3, int3, char3, op>, FMXImpl_AOAOA<char4, float4, int4, char4, op>  },
                    { FMXImpl_AOAOA<schar, float, int, ushort, op>, FMXImpl_AOAOA<char2, float2, int2, ushort2, op>, FMXImpl_AOAOA<char3, float3, int3, ushort3, op>, FMXImpl_AOAOA<char4, float4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, float, int, short, op>, FMXImpl_AOAOA<char2, float2, int2, short2, op>, FMXImpl_AOAOA<char3, float3, int3, short3, op>, FMXImpl_AOAOA<char4, float4, int4, short4, op>  },
                    { FMXImpl_AOAOA<schar, float, int, int, op>, FMXImpl_AOAOA<char2, float2, int2, int2, op>, FMXImpl_AOAOA<char3, float3, int3, int3, op>, FMXImpl_AOAOA<char4, float4, int4, int4, op>  },
                    { FMXImpl_AOAOA<schar, float, int, float, op>, FMXImpl_AOAOA<char2, float2, int2, float2, op>, FMXImpl_AOAOA<char3, float3, int3, float3, op>, FMXImpl_AOAOA<char4, float4, int4, float4, op>  },
                    { FMXImpl_AOAOA<schar, float, int, double, op>, FMXImpl_AOAOA<char2, float2, int2, double2, op>, FMXImpl_AOAOA<char3, float3, int3, double3, op>, FMXImpl_AOAOA<char4, float4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, float, float, uchar, op>, FMXImpl_AOAOA<char2, float2, float2, uchar2, op>, FMXImpl_AOAOA<char3, float3, float3, uchar3, op>, FMXImpl_AOAOA<char4, float4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, float, float, schar, op>, FMXImpl_AOAOA<char2, float2, float2, char2, op>, FMXImpl_AOAOA<char3, float3, float3, char3, op>, FMXImpl_AOAOA<char4, float4, float4, char4, op>  },
                    { FMXImpl_AOAOA<schar, float, float, ushort, op>, FMXImpl_AOAOA<char2, float2, float2, ushort2, op>, FMXImpl_AOAOA<char3, float3, float3, ushort3, op>, FMXImpl_AOAOA<char4, float4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, float, float, short, op>, FMXImpl_AOAOA<char2, float2, float2, short2, op>, FMXImpl_AOAOA<char3, float3, float3, short3, op>, FMXImpl_AOAOA<char4, float4, float4, short4, op>  },
                    { FMXImpl_AOAOA<schar, float, float, int, op>, FMXImpl_AOAOA<char2, float2, float2, int2, op>, FMXImpl_AOAOA<char3, float3, float3, int3, op>, FMXImpl_AOAOA<char4, float4, float4, int4, op>  },
                    { FMXImpl_AOAOA<schar, float, float, float, op>, FMXImpl_AOAOA<char2, float2, float2, float2, op>, FMXImpl_AOAOA<char3, float3, float3, float3, op>, FMXImpl_AOAOA<char4, float4, float4, float4, op>  },
                    { FMXImpl_AOAOA<schar, float, float, double, op>, FMXImpl_AOAOA<char2, float2, float2, double2, op>, FMXImpl_AOAOA<char3, float3, float3, double3, op>, FMXImpl_AOAOA<char4, float4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, float, double, uchar, op>, FMXImpl_AOAOA<char2, float2, double2, uchar2, op>, FMXImpl_AOAOA<char3, float3, double3, uchar3, op>, FMXImpl_AOAOA<char4, float4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, float, double, schar, op>, FMXImpl_AOAOA<char2, float2, double2, char2, op>, FMXImpl_AOAOA<char3, float3, double3, char3, op>, FMXImpl_AOAOA<char4, float4, double4, char4, op>  },
                    { FMXImpl_AOAOA<schar, float, double, ushort, op>, FMXImpl_AOAOA<char2, float2, double2, ushort2, op>, FMXImpl_AOAOA<char3, float3, double3, ushort3, op>, FMXImpl_AOAOA<char4, float4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, float, double, short, op>, FMXImpl_AOAOA<char2, float2, double2, short2, op>, FMXImpl_AOAOA<char3, float3, double3, short3, op>, FMXImpl_AOAOA<char4, float4, double4, short4, op>  },
                    { FMXImpl_AOAOA<schar, float, double, int, op>, FMXImpl_AOAOA<char2, float2, double2, int2, op>, FMXImpl_AOAOA<char3, float3, double3, int3, op>, FMXImpl_AOAOA<char4, float4, double4, int4, op>  },
                    { FMXImpl_AOAOA<schar, float, double, float, op>, FMXImpl_AOAOA<char2, float2, double2, float2, op>, FMXImpl_AOAOA<char3, float3, double3, float3, op>, FMXImpl_AOAOA<char4, float4, double4, float4, op>  },
                    { FMXImpl_AOAOA<schar, float, double, double, op>, FMXImpl_AOAOA<char2, float2, double2, double2, op>, FMXImpl_AOAOA<char3, float3, double3, double3, op>, FMXImpl_AOAOA<char4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<schar, double, uchar, uchar, op>, FMXImpl_AOAOA<char2, double2, uchar2, uchar2, op>, FMXImpl_AOAOA<char3, double3, uchar3, uchar3, op>, FMXImpl_AOAOA<char4, double4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, double, uchar, schar, op>, FMXImpl_AOAOA<char2, double2, uchar2, char2, op>, FMXImpl_AOAOA<char3, double3, uchar3, char3, op>, FMXImpl_AOAOA<char4, double4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<schar, double, uchar, ushort, op>, FMXImpl_AOAOA<char2, double2, uchar2, ushort2, op>, FMXImpl_AOAOA<char3, double3, uchar3, ushort3, op>, FMXImpl_AOAOA<char4, double4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, double, uchar, short, op>, FMXImpl_AOAOA<char2, double2, uchar2, short2, op>, FMXImpl_AOAOA<char3, double3, uchar3, short3, op>, FMXImpl_AOAOA<char4, double4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<schar, double, uchar, int, op>, FMXImpl_AOAOA<char2, double2, uchar2, int2, op>, FMXImpl_AOAOA<char3, double3, uchar3, int3, op>, FMXImpl_AOAOA<char4, double4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<schar, double, uchar, float, op>, FMXImpl_AOAOA<char2, double2, uchar2, float2, op>, FMXImpl_AOAOA<char3, double3, uchar3, float3, op>, FMXImpl_AOAOA<char4, double4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<schar, double, uchar, double, op>, FMXImpl_AOAOA<char2, double2, uchar2, double2, op>, FMXImpl_AOAOA<char3, double3, uchar3, double3, op>, FMXImpl_AOAOA<char4, double4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, double, schar, uchar, op>, FMXImpl_AOAOA<char2, double2, char2, uchar2, op>, FMXImpl_AOAOA<char3, double3, char3, uchar3, op>, FMXImpl_AOAOA<char4, double4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, double, schar, schar, op>, FMXImpl_AOAOA<char2, double2, char2, char2, op>, FMXImpl_AOAOA<char3, double3, char3, char3, op>, FMXImpl_AOAOA<char4, double4, char4, char4, op>  },
                    { FMXImpl_AOAOA<schar, double, schar, ushort, op>, FMXImpl_AOAOA<char2, double2, char2, ushort2, op>, FMXImpl_AOAOA<char3, double3, char3, ushort3, op>, FMXImpl_AOAOA<char4, double4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, double, schar, short, op>, FMXImpl_AOAOA<char2, double2, char2, short2, op>, FMXImpl_AOAOA<char3, double3, char3, short3, op>, FMXImpl_AOAOA<char4, double4, char4, short4, op>  },
                    { FMXImpl_AOAOA<schar, double, schar, int, op>, FMXImpl_AOAOA<char2, double2, char2, int2, op>, FMXImpl_AOAOA<char3, double3, char3, int3, op>, FMXImpl_AOAOA<char4, double4, char4, int4, op>  },
                    { FMXImpl_AOAOA<schar, double, schar, float, op>, FMXImpl_AOAOA<char2, double2, char2, float2, op>, FMXImpl_AOAOA<char3, double3, char3, float3, op>, FMXImpl_AOAOA<char4, double4, char4, float4, op>  },
                    { FMXImpl_AOAOA<schar, double, schar, double, op>, FMXImpl_AOAOA<char2, double2, char2, double2, op>, FMXImpl_AOAOA<char3, double3, char3, double3, op>, FMXImpl_AOAOA<char4, double4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, double, ushort, uchar, op>, FMXImpl_AOAOA<char2, double2, ushort2, uchar2, op>, FMXImpl_AOAOA<char3, double3, ushort3, uchar3, op>, FMXImpl_AOAOA<char4, double4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, double, ushort, schar, op>, FMXImpl_AOAOA<char2, double2, ushort2, char2, op>, FMXImpl_AOAOA<char3, double3, ushort3, char3, op>, FMXImpl_AOAOA<char4, double4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<schar, double, ushort, ushort, op>, FMXImpl_AOAOA<char2, double2, ushort2, ushort2, op>, FMXImpl_AOAOA<char3, double3, ushort3, ushort3, op>, FMXImpl_AOAOA<char4, double4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, double, ushort, short, op>, FMXImpl_AOAOA<char2, double2, ushort2, short2, op>, FMXImpl_AOAOA<char3, double3, ushort3, short3, op>, FMXImpl_AOAOA<char4, double4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<schar, double, ushort, int, op>, FMXImpl_AOAOA<char2, double2, ushort2, int2, op>, FMXImpl_AOAOA<char3, double3, ushort3, int3, op>, FMXImpl_AOAOA<char4, double4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<schar, double, ushort, float, op>, FMXImpl_AOAOA<char2, double2, ushort2, float2, op>, FMXImpl_AOAOA<char3, double3, ushort3, float3, op>, FMXImpl_AOAOA<char4, double4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<schar, double, ushort, double, op>, FMXImpl_AOAOA<char2, double2, ushort2, double2, op>, FMXImpl_AOAOA<char3, double3, ushort3, double3, op>, FMXImpl_AOAOA<char4, double4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, double, short, uchar, op>, FMXImpl_AOAOA<char2, double2, short2, uchar2, op>, FMXImpl_AOAOA<char3, double3, short3, uchar3, op>, FMXImpl_AOAOA<char4, double4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, double, short, schar, op>, FMXImpl_AOAOA<char2, double2, short2, char2, op>, FMXImpl_AOAOA<char3, double3, short3, char3, op>, FMXImpl_AOAOA<char4, double4, short4, char4, op>  },
                    { FMXImpl_AOAOA<schar, double, short, ushort, op>, FMXImpl_AOAOA<char2, double2, short2, ushort2, op>, FMXImpl_AOAOA<char3, double3, short3, ushort3, op>, FMXImpl_AOAOA<char4, double4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, double, short, short, op>, FMXImpl_AOAOA<char2, double2, short2, short2, op>, FMXImpl_AOAOA<char3, double3, short3, short3, op>, FMXImpl_AOAOA<char4, double4, short4, short4, op>  },
                    { FMXImpl_AOAOA<schar, double, short, int, op>, FMXImpl_AOAOA<char2, double2, short2, int2, op>, FMXImpl_AOAOA<char3, double3, short3, int3, op>, FMXImpl_AOAOA<char4, double4, short4, int4, op>  },
                    { FMXImpl_AOAOA<schar, double, short, float, op>, FMXImpl_AOAOA<char2, double2, short2, float2, op>, FMXImpl_AOAOA<char3, double3, short3, float3, op>, FMXImpl_AOAOA<char4, double4, short4, float4, op>  },
                    { FMXImpl_AOAOA<schar, double, short, double, op>, FMXImpl_AOAOA<char2, double2, short2, double2, op>, FMXImpl_AOAOA<char3, double3, short3, double3, op>, FMXImpl_AOAOA<char4, double4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, double, int, uchar, op>, FMXImpl_AOAOA<char2, double2, int2, uchar2, op>, FMXImpl_AOAOA<char3, double3, int3, uchar3, op>, FMXImpl_AOAOA<char4, double4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, double, int, schar, op>, FMXImpl_AOAOA<char2, double2, int2, char2, op>, FMXImpl_AOAOA<char3, double3, int3, char3, op>, FMXImpl_AOAOA<char4, double4, int4, char4, op>  },
                    { FMXImpl_AOAOA<schar, double, int, ushort, op>, FMXImpl_AOAOA<char2, double2, int2, ushort2, op>, FMXImpl_AOAOA<char3, double3, int3, ushort3, op>, FMXImpl_AOAOA<char4, double4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, double, int, short, op>, FMXImpl_AOAOA<char2, double2, int2, short2, op>, FMXImpl_AOAOA<char3, double3, int3, short3, op>, FMXImpl_AOAOA<char4, double4, int4, short4, op>  },
                    { FMXImpl_AOAOA<schar, double, int, int, op>, FMXImpl_AOAOA<char2, double2, int2, int2, op>, FMXImpl_AOAOA<char3, double3, int3, int3, op>, FMXImpl_AOAOA<char4, double4, int4, int4, op>  },
                    { FMXImpl_AOAOA<schar, double, int, float, op>, FMXImpl_AOAOA<char2, double2, int2, float2, op>, FMXImpl_AOAOA<char3, double3, int3, float3, op>, FMXImpl_AOAOA<char4, double4, int4, float4, op>  },
                    { FMXImpl_AOAOA<schar, double, int, double, op>, FMXImpl_AOAOA<char2, double2, int2, double2, op>, FMXImpl_AOAOA<char3, double3, int3, double3, op>, FMXImpl_AOAOA<char4, double4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, double, float, uchar, op>, FMXImpl_AOAOA<char2, double2, float2, uchar2, op>, FMXImpl_AOAOA<char3, double3, float3, uchar3, op>, FMXImpl_AOAOA<char4, double4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, double, float, schar, op>, FMXImpl_AOAOA<char2, double2, float2, char2, op>, FMXImpl_AOAOA<char3, double3, float3, char3, op>, FMXImpl_AOAOA<char4, double4, float4, char4, op>  },
                    { FMXImpl_AOAOA<schar, double, float, ushort, op>, FMXImpl_AOAOA<char2, double2, float2, ushort2, op>, FMXImpl_AOAOA<char3, double3, float3, ushort3, op>, FMXImpl_AOAOA<char4, double4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, double, float, short, op>, FMXImpl_AOAOA<char2, double2, float2, short2, op>, FMXImpl_AOAOA<char3, double3, float3, short3, op>, FMXImpl_AOAOA<char4, double4, float4, short4, op>  },
                    { FMXImpl_AOAOA<schar, double, float, int, op>, FMXImpl_AOAOA<char2, double2, float2, int2, op>, FMXImpl_AOAOA<char3, double3, float3, int3, op>, FMXImpl_AOAOA<char4, double4, float4, int4, op>  },
                    { FMXImpl_AOAOA<schar, double, float, float, op>, FMXImpl_AOAOA<char2, double2, float2, float2, op>, FMXImpl_AOAOA<char3, double3, float3, float3, op>, FMXImpl_AOAOA<char4, double4, float4, float4, op>  },
                    { FMXImpl_AOAOA<schar, double, float, double, op>, FMXImpl_AOAOA<char2, double2, float2, double2, op>, FMXImpl_AOAOA<char3, double3, float3, double3, op>, FMXImpl_AOAOA<char4, double4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<schar, double, double, uchar, op>, FMXImpl_AOAOA<char2, double2, double2, uchar2, op>, FMXImpl_AOAOA<char3, double3, double3, uchar3, op>, FMXImpl_AOAOA<char4, double4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<schar, double, double, schar, op>, FMXImpl_AOAOA<char2, double2, double2, char2, op>, FMXImpl_AOAOA<char3, double3, double3, char3, op>, FMXImpl_AOAOA<char4, double4, double4, char4, op>  },
                    { FMXImpl_AOAOA<schar, double, double, ushort, op>, FMXImpl_AOAOA<char2, double2, double2, ushort2, op>, FMXImpl_AOAOA<char3, double3, double3, ushort3, op>, FMXImpl_AOAOA<char4, double4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<schar, double, double, short, op>, FMXImpl_AOAOA<char2, double2, double2, short2, op>, FMXImpl_AOAOA<char3, double3, double3, short3, op>, FMXImpl_AOAOA<char4, double4, double4, short4, op>  },
                    { FMXImpl_AOAOA<schar, double, double, int, op>, FMXImpl_AOAOA<char2, double2, double2, int2, op>, FMXImpl_AOAOA<char3, double3, double3, int3, op>, FMXImpl_AOAOA<char4, double4, double4, int4, op>  },
                    { FMXImpl_AOAOA<schar, double, double, float, op>, FMXImpl_AOAOA<char2, double2, double2, float2, op>, FMXImpl_AOAOA<char3, double3, double3, float3, op>, FMXImpl_AOAOA<char4, double4, double4, float4, op>  },
                    { FMXImpl_AOAOA<schar, double, double, double, op>, FMXImpl_AOAOA<char2, double2, double2, double2, op>, FMXImpl_AOAOA<char3, double3, double3, double3, op>, FMXImpl_AOAOA<char4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { FMXImpl_AOAOA<ushort, uchar, uchar, uchar, op>, FMXImpl_AOAOA<ushort2, uchar2, uchar2, uchar2, op>, FMXImpl_AOAOA<ushort3, uchar3, uchar3, uchar3, op>, FMXImpl_AOAOA<ushort4, uchar4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, uchar, schar, op>, FMXImpl_AOAOA<ushort2, uchar2, uchar2, char2, op>, FMXImpl_AOAOA<ushort3, uchar3, uchar3, char3, op>, FMXImpl_AOAOA<ushort4, uchar4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, uchar, ushort, op>, FMXImpl_AOAOA<ushort2, uchar2, uchar2, ushort2, op>, FMXImpl_AOAOA<ushort3, uchar3, uchar3, ushort3, op>, FMXImpl_AOAOA<ushort4, uchar4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, uchar, short, op>, FMXImpl_AOAOA<ushort2, uchar2, uchar2, short2, op>, FMXImpl_AOAOA<ushort3, uchar3, uchar3, short3, op>, FMXImpl_AOAOA<ushort4, uchar4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, uchar, int, op>, FMXImpl_AOAOA<ushort2, uchar2, uchar2, int2, op>, FMXImpl_AOAOA<ushort3, uchar3, uchar3, int3, op>, FMXImpl_AOAOA<ushort4, uchar4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, uchar, float, op>, FMXImpl_AOAOA<ushort2, uchar2, uchar2, float2, op>, FMXImpl_AOAOA<ushort3, uchar3, uchar3, float3, op>, FMXImpl_AOAOA<ushort4, uchar4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, uchar, double, op>, FMXImpl_AOAOA<ushort2, uchar2, uchar2, double2, op>, FMXImpl_AOAOA<ushort3, uchar3, uchar3, double3, op>, FMXImpl_AOAOA<ushort4, uchar4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, uchar, schar, uchar, op>, FMXImpl_AOAOA<ushort2, uchar2, char2, uchar2, op>, FMXImpl_AOAOA<ushort3, uchar3, char3, uchar3, op>, FMXImpl_AOAOA<ushort4, uchar4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, schar, schar, op>, FMXImpl_AOAOA<ushort2, uchar2, char2, char2, op>, FMXImpl_AOAOA<ushort3, uchar3, char3, char3, op>, FMXImpl_AOAOA<ushort4, uchar4, char4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, schar, ushort, op>, FMXImpl_AOAOA<ushort2, uchar2, char2, ushort2, op>, FMXImpl_AOAOA<ushort3, uchar3, char3, ushort3, op>, FMXImpl_AOAOA<ushort4, uchar4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, schar, short, op>, FMXImpl_AOAOA<ushort2, uchar2, char2, short2, op>, FMXImpl_AOAOA<ushort3, uchar3, char3, short3, op>, FMXImpl_AOAOA<ushort4, uchar4, char4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, schar, int, op>, FMXImpl_AOAOA<ushort2, uchar2, char2, int2, op>, FMXImpl_AOAOA<ushort3, uchar3, char3, int3, op>, FMXImpl_AOAOA<ushort4, uchar4, char4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, schar, float, op>, FMXImpl_AOAOA<ushort2, uchar2, char2, float2, op>, FMXImpl_AOAOA<ushort3, uchar3, char3, float3, op>, FMXImpl_AOAOA<ushort4, uchar4, char4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, schar, double, op>, FMXImpl_AOAOA<ushort2, uchar2, char2, double2, op>, FMXImpl_AOAOA<ushort3, uchar3, char3, double3, op>, FMXImpl_AOAOA<ushort4, uchar4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, uchar, ushort, uchar, op>, FMXImpl_AOAOA<ushort2, uchar2, ushort2, uchar2, op>, FMXImpl_AOAOA<ushort3, uchar3, ushort3, uchar3, op>, FMXImpl_AOAOA<ushort4, uchar4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, ushort, schar, op>, FMXImpl_AOAOA<ushort2, uchar2, ushort2, char2, op>, FMXImpl_AOAOA<ushort3, uchar3, ushort3, char3, op>, FMXImpl_AOAOA<ushort4, uchar4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, ushort, ushort, op>, FMXImpl_AOAOA<ushort2, uchar2, ushort2, ushort2, op>, FMXImpl_AOAOA<ushort3, uchar3, ushort3, ushort3, op>, FMXImpl_AOAOA<ushort4, uchar4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, ushort, short, op>, FMXImpl_AOAOA<ushort2, uchar2, ushort2, short2, op>, FMXImpl_AOAOA<ushort3, uchar3, ushort3, short3, op>, FMXImpl_AOAOA<ushort4, uchar4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, ushort, int, op>, FMXImpl_AOAOA<ushort2, uchar2, ushort2, int2, op>, FMXImpl_AOAOA<ushort3, uchar3, ushort3, int3, op>, FMXImpl_AOAOA<ushort4, uchar4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, ushort, float, op>, FMXImpl_AOAOA<ushort2, uchar2, ushort2, float2, op>, FMXImpl_AOAOA<ushort3, uchar3, ushort3, float3, op>, FMXImpl_AOAOA<ushort4, uchar4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, ushort, double, op>, FMXImpl_AOAOA<ushort2, uchar2, ushort2, double2, op>, FMXImpl_AOAOA<ushort3, uchar3, ushort3, double3, op>, FMXImpl_AOAOA<ushort4, uchar4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, uchar, short, uchar, op>, FMXImpl_AOAOA<ushort2, uchar2, short2, uchar2, op>, FMXImpl_AOAOA<ushort3, uchar3, short3, uchar3, op>, FMXImpl_AOAOA<ushort4, uchar4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, short, schar, op>, FMXImpl_AOAOA<ushort2, uchar2, short2, char2, op>, FMXImpl_AOAOA<ushort3, uchar3, short3, char3, op>, FMXImpl_AOAOA<ushort4, uchar4, short4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, short, ushort, op>, FMXImpl_AOAOA<ushort2, uchar2, short2, ushort2, op>, FMXImpl_AOAOA<ushort3, uchar3, short3, ushort3, op>, FMXImpl_AOAOA<ushort4, uchar4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, short, short, op>, FMXImpl_AOAOA<ushort2, uchar2, short2, short2, op>, FMXImpl_AOAOA<ushort3, uchar3, short3, short3, op>, FMXImpl_AOAOA<ushort4, uchar4, short4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, short, int, op>, FMXImpl_AOAOA<ushort2, uchar2, short2, int2, op>, FMXImpl_AOAOA<ushort3, uchar3, short3, int3, op>, FMXImpl_AOAOA<ushort4, uchar4, short4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, short, float, op>, FMXImpl_AOAOA<ushort2, uchar2, short2, float2, op>, FMXImpl_AOAOA<ushort3, uchar3, short3, float3, op>, FMXImpl_AOAOA<ushort4, uchar4, short4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, short, double, op>, FMXImpl_AOAOA<ushort2, uchar2, short2, double2, op>, FMXImpl_AOAOA<ushort3, uchar3, short3, double3, op>, FMXImpl_AOAOA<ushort4, uchar4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, uchar, int, uchar, op>, FMXImpl_AOAOA<ushort2, uchar2, int2, uchar2, op>, FMXImpl_AOAOA<ushort3, uchar3, int3, uchar3, op>, FMXImpl_AOAOA<ushort4, uchar4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, int, schar, op>, FMXImpl_AOAOA<ushort2, uchar2, int2, char2, op>, FMXImpl_AOAOA<ushort3, uchar3, int3, char3, op>, FMXImpl_AOAOA<ushort4, uchar4, int4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, int, ushort, op>, FMXImpl_AOAOA<ushort2, uchar2, int2, ushort2, op>, FMXImpl_AOAOA<ushort3, uchar3, int3, ushort3, op>, FMXImpl_AOAOA<ushort4, uchar4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, int, short, op>, FMXImpl_AOAOA<ushort2, uchar2, int2, short2, op>, FMXImpl_AOAOA<ushort3, uchar3, int3, short3, op>, FMXImpl_AOAOA<ushort4, uchar4, int4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, int, int, op>, FMXImpl_AOAOA<ushort2, uchar2, int2, int2, op>, FMXImpl_AOAOA<ushort3, uchar3, int3, int3, op>, FMXImpl_AOAOA<ushort4, uchar4, int4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, int, float, op>, FMXImpl_AOAOA<ushort2, uchar2, int2, float2, op>, FMXImpl_AOAOA<ushort3, uchar3, int3, float3, op>, FMXImpl_AOAOA<ushort4, uchar4, int4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, int, double, op>, FMXImpl_AOAOA<ushort2, uchar2, int2, double2, op>, FMXImpl_AOAOA<ushort3, uchar3, int3, double3, op>, FMXImpl_AOAOA<ushort4, uchar4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, uchar, float, uchar, op>, FMXImpl_AOAOA<ushort2, uchar2, float2, uchar2, op>, FMXImpl_AOAOA<ushort3, uchar3, float3, uchar3, op>, FMXImpl_AOAOA<ushort4, uchar4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, float, schar, op>, FMXImpl_AOAOA<ushort2, uchar2, float2, char2, op>, FMXImpl_AOAOA<ushort3, uchar3, float3, char3, op>, FMXImpl_AOAOA<ushort4, uchar4, float4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, float, ushort, op>, FMXImpl_AOAOA<ushort2, uchar2, float2, ushort2, op>, FMXImpl_AOAOA<ushort3, uchar3, float3, ushort3, op>, FMXImpl_AOAOA<ushort4, uchar4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, float, short, op>, FMXImpl_AOAOA<ushort2, uchar2, float2, short2, op>, FMXImpl_AOAOA<ushort3, uchar3, float3, short3, op>, FMXImpl_AOAOA<ushort4, uchar4, float4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, float, int, op>, FMXImpl_AOAOA<ushort2, uchar2, float2, int2, op>, FMXImpl_AOAOA<ushort3, uchar3, float3, int3, op>, FMXImpl_AOAOA<ushort4, uchar4, float4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, float, float, op>, FMXImpl_AOAOA<ushort2, uchar2, float2, float2, op>, FMXImpl_AOAOA<ushort3, uchar3, float3, float3, op>, FMXImpl_AOAOA<ushort4, uchar4, float4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, float, double, op>, FMXImpl_AOAOA<ushort2, uchar2, float2, double2, op>, FMXImpl_AOAOA<ushort3, uchar3, float3, double3, op>, FMXImpl_AOAOA<ushort4, uchar4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, uchar, double, uchar, op>, FMXImpl_AOAOA<ushort2, uchar2, double2, uchar2, op>, FMXImpl_AOAOA<ushort3, uchar3, double3, uchar3, op>, FMXImpl_AOAOA<ushort4, uchar4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, double, schar, op>, FMXImpl_AOAOA<ushort2, uchar2, double2, char2, op>, FMXImpl_AOAOA<ushort3, uchar3, double3, char3, op>, FMXImpl_AOAOA<ushort4, uchar4, double4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, double, ushort, op>, FMXImpl_AOAOA<ushort2, uchar2, double2, ushort2, op>, FMXImpl_AOAOA<ushort3, uchar3, double3, ushort3, op>, FMXImpl_AOAOA<ushort4, uchar4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, double, short, op>, FMXImpl_AOAOA<ushort2, uchar2, double2, short2, op>, FMXImpl_AOAOA<ushort3, uchar3, double3, short3, op>, FMXImpl_AOAOA<ushort4, uchar4, double4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, double, int, op>, FMXImpl_AOAOA<ushort2, uchar2, double2, int2, op>, FMXImpl_AOAOA<ushort3, uchar3, double3, int3, op>, FMXImpl_AOAOA<ushort4, uchar4, double4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, double, float, op>, FMXImpl_AOAOA<ushort2, uchar2, double2, float2, op>, FMXImpl_AOAOA<ushort3, uchar3, double3, float3, op>, FMXImpl_AOAOA<ushort4, uchar4, double4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, uchar, double, double, op>, FMXImpl_AOAOA<ushort2, uchar2, double2, double2, op>, FMXImpl_AOAOA<ushort3, uchar3, double3, double3, op>, FMXImpl_AOAOA<ushort4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<ushort, schar, uchar, uchar, op>, FMXImpl_AOAOA<ushort2, char2, uchar2, uchar2, op>, FMXImpl_AOAOA<ushort3, char3, uchar3, uchar3, op>, FMXImpl_AOAOA<ushort4, char4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, uchar, schar, op>, FMXImpl_AOAOA<ushort2, char2, uchar2, char2, op>, FMXImpl_AOAOA<ushort3, char3, uchar3, char3, op>, FMXImpl_AOAOA<ushort4, char4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, uchar, ushort, op>, FMXImpl_AOAOA<ushort2, char2, uchar2, ushort2, op>, FMXImpl_AOAOA<ushort3, char3, uchar3, ushort3, op>, FMXImpl_AOAOA<ushort4, char4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, uchar, short, op>, FMXImpl_AOAOA<ushort2, char2, uchar2, short2, op>, FMXImpl_AOAOA<ushort3, char3, uchar3, short3, op>, FMXImpl_AOAOA<ushort4, char4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, uchar, int, op>, FMXImpl_AOAOA<ushort2, char2, uchar2, int2, op>, FMXImpl_AOAOA<ushort3, char3, uchar3, int3, op>, FMXImpl_AOAOA<ushort4, char4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, uchar, float, op>, FMXImpl_AOAOA<ushort2, char2, uchar2, float2, op>, FMXImpl_AOAOA<ushort3, char3, uchar3, float3, op>, FMXImpl_AOAOA<ushort4, char4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, uchar, double, op>, FMXImpl_AOAOA<ushort2, char2, uchar2, double2, op>, FMXImpl_AOAOA<ushort3, char3, uchar3, double3, op>, FMXImpl_AOAOA<ushort4, char4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, schar, schar, uchar, op>, FMXImpl_AOAOA<ushort2, char2, char2, uchar2, op>, FMXImpl_AOAOA<ushort3, char3, char3, uchar3, op>, FMXImpl_AOAOA<ushort4, char4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, schar, schar, op>, FMXImpl_AOAOA<ushort2, char2, char2, char2, op>, FMXImpl_AOAOA<ushort3, char3, char3, char3, op>, FMXImpl_AOAOA<ushort4, char4, char4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, schar, ushort, op>, FMXImpl_AOAOA<ushort2, char2, char2, ushort2, op>, FMXImpl_AOAOA<ushort3, char3, char3, ushort3, op>, FMXImpl_AOAOA<ushort4, char4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, schar, short, op>, FMXImpl_AOAOA<ushort2, char2, char2, short2, op>, FMXImpl_AOAOA<ushort3, char3, char3, short3, op>, FMXImpl_AOAOA<ushort4, char4, char4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, schar, int, op>, FMXImpl_AOAOA<ushort2, char2, char2, int2, op>, FMXImpl_AOAOA<ushort3, char3, char3, int3, op>, FMXImpl_AOAOA<ushort4, char4, char4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, schar, float, op>, FMXImpl_AOAOA<ushort2, char2, char2, float2, op>, FMXImpl_AOAOA<ushort3, char3, char3, float3, op>, FMXImpl_AOAOA<ushort4, char4, char4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, schar, double, op>, FMXImpl_AOAOA<ushort2, char2, char2, double2, op>, FMXImpl_AOAOA<ushort3, char3, char3, double3, op>, FMXImpl_AOAOA<ushort4, char4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, schar, ushort, uchar, op>, FMXImpl_AOAOA<ushort2, char2, ushort2, uchar2, op>, FMXImpl_AOAOA<ushort3, char3, ushort3, uchar3, op>, FMXImpl_AOAOA<ushort4, char4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, ushort, schar, op>, FMXImpl_AOAOA<ushort2, char2, ushort2, char2, op>, FMXImpl_AOAOA<ushort3, char3, ushort3, char3, op>, FMXImpl_AOAOA<ushort4, char4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, ushort, ushort, op>, FMXImpl_AOAOA<ushort2, char2, ushort2, ushort2, op>, FMXImpl_AOAOA<ushort3, char3, ushort3, ushort3, op>, FMXImpl_AOAOA<ushort4, char4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, ushort, short, op>, FMXImpl_AOAOA<ushort2, char2, ushort2, short2, op>, FMXImpl_AOAOA<ushort3, char3, ushort3, short3, op>, FMXImpl_AOAOA<ushort4, char4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, ushort, int, op>, FMXImpl_AOAOA<ushort2, char2, ushort2, int2, op>, FMXImpl_AOAOA<ushort3, char3, ushort3, int3, op>, FMXImpl_AOAOA<ushort4, char4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, ushort, float, op>, FMXImpl_AOAOA<ushort2, char2, ushort2, float2, op>, FMXImpl_AOAOA<ushort3, char3, ushort3, float3, op>, FMXImpl_AOAOA<ushort4, char4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, ushort, double, op>, FMXImpl_AOAOA<ushort2, char2, ushort2, double2, op>, FMXImpl_AOAOA<ushort3, char3, ushort3, double3, op>, FMXImpl_AOAOA<ushort4, char4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, schar, short, uchar, op>, FMXImpl_AOAOA<ushort2, char2, short2, uchar2, op>, FMXImpl_AOAOA<ushort3, char3, short3, uchar3, op>, FMXImpl_AOAOA<ushort4, char4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, short, schar, op>, FMXImpl_AOAOA<ushort2, char2, short2, char2, op>, FMXImpl_AOAOA<ushort3, char3, short3, char3, op>, FMXImpl_AOAOA<ushort4, char4, short4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, short, ushort, op>, FMXImpl_AOAOA<ushort2, char2, short2, ushort2, op>, FMXImpl_AOAOA<ushort3, char3, short3, ushort3, op>, FMXImpl_AOAOA<ushort4, char4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, short, short, op>, FMXImpl_AOAOA<ushort2, char2, short2, short2, op>, FMXImpl_AOAOA<ushort3, char3, short3, short3, op>, FMXImpl_AOAOA<ushort4, char4, short4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, short, int, op>, FMXImpl_AOAOA<ushort2, char2, short2, int2, op>, FMXImpl_AOAOA<ushort3, char3, short3, int3, op>, FMXImpl_AOAOA<ushort4, char4, short4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, short, float, op>, FMXImpl_AOAOA<ushort2, char2, short2, float2, op>, FMXImpl_AOAOA<ushort3, char3, short3, float3, op>, FMXImpl_AOAOA<ushort4, char4, short4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, short, double, op>, FMXImpl_AOAOA<ushort2, char2, short2, double2, op>, FMXImpl_AOAOA<ushort3, char3, short3, double3, op>, FMXImpl_AOAOA<ushort4, char4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, schar, int, uchar, op>, FMXImpl_AOAOA<ushort2, char2, int2, uchar2, op>, FMXImpl_AOAOA<ushort3, char3, int3, uchar3, op>, FMXImpl_AOAOA<ushort4, char4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, int, schar, op>, FMXImpl_AOAOA<ushort2, char2, int2, char2, op>, FMXImpl_AOAOA<ushort3, char3, int3, char3, op>, FMXImpl_AOAOA<ushort4, char4, int4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, int, ushort, op>, FMXImpl_AOAOA<ushort2, char2, int2, ushort2, op>, FMXImpl_AOAOA<ushort3, char3, int3, ushort3, op>, FMXImpl_AOAOA<ushort4, char4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, int, short, op>, FMXImpl_AOAOA<ushort2, char2, int2, short2, op>, FMXImpl_AOAOA<ushort3, char3, int3, short3, op>, FMXImpl_AOAOA<ushort4, char4, int4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, int, int, op>, FMXImpl_AOAOA<ushort2, char2, int2, int2, op>, FMXImpl_AOAOA<ushort3, char3, int3, int3, op>, FMXImpl_AOAOA<ushort4, char4, int4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, int, float, op>, FMXImpl_AOAOA<ushort2, char2, int2, float2, op>, FMXImpl_AOAOA<ushort3, char3, int3, float3, op>, FMXImpl_AOAOA<ushort4, char4, int4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, int, double, op>, FMXImpl_AOAOA<ushort2, char2, int2, double2, op>, FMXImpl_AOAOA<ushort3, char3, int3, double3, op>, FMXImpl_AOAOA<ushort4, char4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, schar, float, uchar, op>, FMXImpl_AOAOA<ushort2, char2, float2, uchar2, op>, FMXImpl_AOAOA<ushort3, char3, float3, uchar3, op>, FMXImpl_AOAOA<ushort4, char4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, float, schar, op>, FMXImpl_AOAOA<ushort2, char2, float2, char2, op>, FMXImpl_AOAOA<ushort3, char3, float3, char3, op>, FMXImpl_AOAOA<ushort4, char4, float4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, float, ushort, op>, FMXImpl_AOAOA<ushort2, char2, float2, ushort2, op>, FMXImpl_AOAOA<ushort3, char3, float3, ushort3, op>, FMXImpl_AOAOA<ushort4, char4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, float, short, op>, FMXImpl_AOAOA<ushort2, char2, float2, short2, op>, FMXImpl_AOAOA<ushort3, char3, float3, short3, op>, FMXImpl_AOAOA<ushort4, char4, float4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, float, int, op>, FMXImpl_AOAOA<ushort2, char2, float2, int2, op>, FMXImpl_AOAOA<ushort3, char3, float3, int3, op>, FMXImpl_AOAOA<ushort4, char4, float4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, float, float, op>, FMXImpl_AOAOA<ushort2, char2, float2, float2, op>, FMXImpl_AOAOA<ushort3, char3, float3, float3, op>, FMXImpl_AOAOA<ushort4, char4, float4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, float, double, op>, FMXImpl_AOAOA<ushort2, char2, float2, double2, op>, FMXImpl_AOAOA<ushort3, char3, float3, double3, op>, FMXImpl_AOAOA<ushort4, char4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, schar, double, uchar, op>, FMXImpl_AOAOA<ushort2, char2, double2, uchar2, op>, FMXImpl_AOAOA<ushort3, char3, double3, uchar3, op>, FMXImpl_AOAOA<ushort4, char4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, double, schar, op>, FMXImpl_AOAOA<ushort2, char2, double2, char2, op>, FMXImpl_AOAOA<ushort3, char3, double3, char3, op>, FMXImpl_AOAOA<ushort4, char4, double4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, double, ushort, op>, FMXImpl_AOAOA<ushort2, char2, double2, ushort2, op>, FMXImpl_AOAOA<ushort3, char3, double3, ushort3, op>, FMXImpl_AOAOA<ushort4, char4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, double, short, op>, FMXImpl_AOAOA<ushort2, char2, double2, short2, op>, FMXImpl_AOAOA<ushort3, char3, double3, short3, op>, FMXImpl_AOAOA<ushort4, char4, double4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, double, int, op>, FMXImpl_AOAOA<ushort2, char2, double2, int2, op>, FMXImpl_AOAOA<ushort3, char3, double3, int3, op>, FMXImpl_AOAOA<ushort4, char4, double4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, double, float, op>, FMXImpl_AOAOA<ushort2, char2, double2, float2, op>, FMXImpl_AOAOA<ushort3, char3, double3, float3, op>, FMXImpl_AOAOA<ushort4, char4, double4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, schar, double, double, op>, FMXImpl_AOAOA<ushort2, char2, double2, double2, op>, FMXImpl_AOAOA<ushort3, char3, double3, double3, op>, FMXImpl_AOAOA<ushort4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<ushort, ushort, uchar, uchar, op>, FMXImpl_AOAOA<ushort2, ushort2, uchar2, uchar2, op>, FMXImpl_AOAOA<ushort3, ushort3, uchar3, uchar3, op>, FMXImpl_AOAOA<ushort4, ushort4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, uchar, schar, op>, FMXImpl_AOAOA<ushort2, ushort2, uchar2, char2, op>, FMXImpl_AOAOA<ushort3, ushort3, uchar3, char3, op>, FMXImpl_AOAOA<ushort4, ushort4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, uchar, ushort, op>, FMXImpl_AOAOA<ushort2, ushort2, uchar2, ushort2, op>, FMXImpl_AOAOA<ushort3, ushort3, uchar3, ushort3, op>, FMXImpl_AOAOA<ushort4, ushort4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, uchar, short, op>, FMXImpl_AOAOA<ushort2, ushort2, uchar2, short2, op>, FMXImpl_AOAOA<ushort3, ushort3, uchar3, short3, op>, FMXImpl_AOAOA<ushort4, ushort4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, uchar, int, op>, FMXImpl_AOAOA<ushort2, ushort2, uchar2, int2, op>, FMXImpl_AOAOA<ushort3, ushort3, uchar3, int3, op>, FMXImpl_AOAOA<ushort4, ushort4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, uchar, float, op>, FMXImpl_AOAOA<ushort2, ushort2, uchar2, float2, op>, FMXImpl_AOAOA<ushort3, ushort3, uchar3, float3, op>, FMXImpl_AOAOA<ushort4, ushort4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, uchar, double, op>, FMXImpl_AOAOA<ushort2, ushort2, uchar2, double2, op>, FMXImpl_AOAOA<ushort3, ushort3, uchar3, double3, op>, FMXImpl_AOAOA<ushort4, ushort4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, ushort, schar, uchar, op>, FMXImpl_AOAOA<ushort2, ushort2, char2, uchar2, op>, FMXImpl_AOAOA<ushort3, ushort3, char3, uchar3, op>, FMXImpl_AOAOA<ushort4, ushort4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, schar, schar, op>, FMXImpl_AOAOA<ushort2, ushort2, char2, char2, op>, FMXImpl_AOAOA<ushort3, ushort3, char3, char3, op>, FMXImpl_AOAOA<ushort4, ushort4, char4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, schar, ushort, op>, FMXImpl_AOAOA<ushort2, ushort2, char2, ushort2, op>, FMXImpl_AOAOA<ushort3, ushort3, char3, ushort3, op>, FMXImpl_AOAOA<ushort4, ushort4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, schar, short, op>, FMXImpl_AOAOA<ushort2, ushort2, char2, short2, op>, FMXImpl_AOAOA<ushort3, ushort3, char3, short3, op>, FMXImpl_AOAOA<ushort4, ushort4, char4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, schar, int, op>, FMXImpl_AOAOA<ushort2, ushort2, char2, int2, op>, FMXImpl_AOAOA<ushort3, ushort3, char3, int3, op>, FMXImpl_AOAOA<ushort4, ushort4, char4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, schar, float, op>, FMXImpl_AOAOA<ushort2, ushort2, char2, float2, op>, FMXImpl_AOAOA<ushort3, ushort3, char3, float3, op>, FMXImpl_AOAOA<ushort4, ushort4, char4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, schar, double, op>, FMXImpl_AOAOA<ushort2, ushort2, char2, double2, op>, FMXImpl_AOAOA<ushort3, ushort3, char3, double3, op>, FMXImpl_AOAOA<ushort4, ushort4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, ushort, ushort, uchar, op>, FMXImpl_AOAOA<ushort2, ushort2, ushort2, uchar2, op>, FMXImpl_AOAOA<ushort3, ushort3, ushort3, uchar3, op>, FMXImpl_AOAOA<ushort4, ushort4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, ushort, schar, op>, FMXImpl_AOAOA<ushort2, ushort2, ushort2, char2, op>, FMXImpl_AOAOA<ushort3, ushort3, ushort3, char3, op>, FMXImpl_AOAOA<ushort4, ushort4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, ushort, ushort, op>, FMXImpl_AOAOA<ushort2, ushort2, ushort2, ushort2, op>, FMXImpl_AOAOA<ushort3, ushort3, ushort3, ushort3, op>, FMXImpl_AOAOA<ushort4, ushort4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, ushort, short, op>, FMXImpl_AOAOA<ushort2, ushort2, ushort2, short2, op>, FMXImpl_AOAOA<ushort3, ushort3, ushort3, short3, op>, FMXImpl_AOAOA<ushort4, ushort4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, ushort, int, op>, FMXImpl_AOAOA<ushort2, ushort2, ushort2, int2, op>, FMXImpl_AOAOA<ushort3, ushort3, ushort3, int3, op>, FMXImpl_AOAOA<ushort4, ushort4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, ushort, float, op>, FMXImpl_AOAOA<ushort2, ushort2, ushort2, float2, op>, FMXImpl_AOAOA<ushort3, ushort3, ushort3, float3, op>, FMXImpl_AOAOA<ushort4, ushort4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, ushort, double, op>, FMXImpl_AOAOA<ushort2, ushort2, ushort2, double2, op>, FMXImpl_AOAOA<ushort3, ushort3, ushort3, double3, op>, FMXImpl_AOAOA<ushort4, ushort4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, ushort, short, uchar, op>, FMXImpl_AOAOA<ushort2, ushort2, short2, uchar2, op>, FMXImpl_AOAOA<ushort3, ushort3, short3, uchar3, op>, FMXImpl_AOAOA<ushort4, ushort4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, short, schar, op>, FMXImpl_AOAOA<ushort2, ushort2, short2, char2, op>, FMXImpl_AOAOA<ushort3, ushort3, short3, char3, op>, FMXImpl_AOAOA<ushort4, ushort4, short4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, short, ushort, op>, FMXImpl_AOAOA<ushort2, ushort2, short2, ushort2, op>, FMXImpl_AOAOA<ushort3, ushort3, short3, ushort3, op>, FMXImpl_AOAOA<ushort4, ushort4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, short, short, op>, FMXImpl_AOAOA<ushort2, ushort2, short2, short2, op>, FMXImpl_AOAOA<ushort3, ushort3, short3, short3, op>, FMXImpl_AOAOA<ushort4, ushort4, short4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, short, int, op>, FMXImpl_AOAOA<ushort2, ushort2, short2, int2, op>, FMXImpl_AOAOA<ushort3, ushort3, short3, int3, op>, FMXImpl_AOAOA<ushort4, ushort4, short4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, short, float, op>, FMXImpl_AOAOA<ushort2, ushort2, short2, float2, op>, FMXImpl_AOAOA<ushort3, ushort3, short3, float3, op>, FMXImpl_AOAOA<ushort4, ushort4, short4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, short, double, op>, FMXImpl_AOAOA<ushort2, ushort2, short2, double2, op>, FMXImpl_AOAOA<ushort3, ushort3, short3, double3, op>, FMXImpl_AOAOA<ushort4, ushort4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, ushort, int, uchar, op>, FMXImpl_AOAOA<ushort2, ushort2, int2, uchar2, op>, FMXImpl_AOAOA<ushort3, ushort3, int3, uchar3, op>, FMXImpl_AOAOA<ushort4, ushort4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, int, schar, op>, FMXImpl_AOAOA<ushort2, ushort2, int2, char2, op>, FMXImpl_AOAOA<ushort3, ushort3, int3, char3, op>, FMXImpl_AOAOA<ushort4, ushort4, int4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, int, ushort, op>, FMXImpl_AOAOA<ushort2, ushort2, int2, ushort2, op>, FMXImpl_AOAOA<ushort3, ushort3, int3, ushort3, op>, FMXImpl_AOAOA<ushort4, ushort4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, int, short, op>, FMXImpl_AOAOA<ushort2, ushort2, int2, short2, op>, FMXImpl_AOAOA<ushort3, ushort3, int3, short3, op>, FMXImpl_AOAOA<ushort4, ushort4, int4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, int, int, op>, FMXImpl_AOAOA<ushort2, ushort2, int2, int2, op>, FMXImpl_AOAOA<ushort3, ushort3, int3, int3, op>, FMXImpl_AOAOA<ushort4, ushort4, int4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, int, float, op>, FMXImpl_AOAOA<ushort2, ushort2, int2, float2, op>, FMXImpl_AOAOA<ushort3, ushort3, int3, float3, op>, FMXImpl_AOAOA<ushort4, ushort4, int4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, int, double, op>, FMXImpl_AOAOA<ushort2, ushort2, int2, double2, op>, FMXImpl_AOAOA<ushort3, ushort3, int3, double3, op>, FMXImpl_AOAOA<ushort4, ushort4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, ushort, float, uchar, op>, FMXImpl_AOAOA<ushort2, ushort2, float2, uchar2, op>, FMXImpl_AOAOA<ushort3, ushort3, float3, uchar3, op>, FMXImpl_AOAOA<ushort4, ushort4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, float, schar, op>, FMXImpl_AOAOA<ushort2, ushort2, float2, char2, op>, FMXImpl_AOAOA<ushort3, ushort3, float3, char3, op>, FMXImpl_AOAOA<ushort4, ushort4, float4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, float, ushort, op>, FMXImpl_AOAOA<ushort2, ushort2, float2, ushort2, op>, FMXImpl_AOAOA<ushort3, ushort3, float3, ushort3, op>, FMXImpl_AOAOA<ushort4, ushort4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, float, short, op>, FMXImpl_AOAOA<ushort2, ushort2, float2, short2, op>, FMXImpl_AOAOA<ushort3, ushort3, float3, short3, op>, FMXImpl_AOAOA<ushort4, ushort4, float4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, float, int, op>, FMXImpl_AOAOA<ushort2, ushort2, float2, int2, op>, FMXImpl_AOAOA<ushort3, ushort3, float3, int3, op>, FMXImpl_AOAOA<ushort4, ushort4, float4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, float, float, op>, FMXImpl_AOAOA<ushort2, ushort2, float2, float2, op>, FMXImpl_AOAOA<ushort3, ushort3, float3, float3, op>, FMXImpl_AOAOA<ushort4, ushort4, float4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, float, double, op>, FMXImpl_AOAOA<ushort2, ushort2, float2, double2, op>, FMXImpl_AOAOA<ushort3, ushort3, float3, double3, op>, FMXImpl_AOAOA<ushort4, ushort4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, ushort, double, uchar, op>, FMXImpl_AOAOA<ushort2, ushort2, double2, uchar2, op>, FMXImpl_AOAOA<ushort3, ushort3, double3, uchar3, op>, FMXImpl_AOAOA<ushort4, ushort4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, double, schar, op>, FMXImpl_AOAOA<ushort2, ushort2, double2, char2, op>, FMXImpl_AOAOA<ushort3, ushort3, double3, char3, op>, FMXImpl_AOAOA<ushort4, ushort4, double4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, double, ushort, op>, FMXImpl_AOAOA<ushort2, ushort2, double2, ushort2, op>, FMXImpl_AOAOA<ushort3, ushort3, double3, ushort3, op>, FMXImpl_AOAOA<ushort4, ushort4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, double, short, op>, FMXImpl_AOAOA<ushort2, ushort2, double2, short2, op>, FMXImpl_AOAOA<ushort3, ushort3, double3, short3, op>, FMXImpl_AOAOA<ushort4, ushort4, double4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, double, int, op>, FMXImpl_AOAOA<ushort2, ushort2, double2, int2, op>, FMXImpl_AOAOA<ushort3, ushort3, double3, int3, op>, FMXImpl_AOAOA<ushort4, ushort4, double4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, double, float, op>, FMXImpl_AOAOA<ushort2, ushort2, double2, float2, op>, FMXImpl_AOAOA<ushort3, ushort3, double3, float3, op>, FMXImpl_AOAOA<ushort4, ushort4, double4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, ushort, double, double, op>, FMXImpl_AOAOA<ushort2, ushort2, double2, double2, op>, FMXImpl_AOAOA<ushort3, ushort3, double3, double3, op>, FMXImpl_AOAOA<ushort4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<ushort, short, uchar, uchar, op>, FMXImpl_AOAOA<ushort2, short2, uchar2, uchar2, op>, FMXImpl_AOAOA<ushort3, short3, uchar3, uchar3, op>, FMXImpl_AOAOA<ushort4, short4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, short, uchar, schar, op>, FMXImpl_AOAOA<ushort2, short2, uchar2, char2, op>, FMXImpl_AOAOA<ushort3, short3, uchar3, char3, op>, FMXImpl_AOAOA<ushort4, short4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, short, uchar, ushort, op>, FMXImpl_AOAOA<ushort2, short2, uchar2, ushort2, op>, FMXImpl_AOAOA<ushort3, short3, uchar3, ushort3, op>, FMXImpl_AOAOA<ushort4, short4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, short, uchar, short, op>, FMXImpl_AOAOA<ushort2, short2, uchar2, short2, op>, FMXImpl_AOAOA<ushort3, short3, uchar3, short3, op>, FMXImpl_AOAOA<ushort4, short4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, short, uchar, int, op>, FMXImpl_AOAOA<ushort2, short2, uchar2, int2, op>, FMXImpl_AOAOA<ushort3, short3, uchar3, int3, op>, FMXImpl_AOAOA<ushort4, short4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, short, uchar, float, op>, FMXImpl_AOAOA<ushort2, short2, uchar2, float2, op>, FMXImpl_AOAOA<ushort3, short3, uchar3, float3, op>, FMXImpl_AOAOA<ushort4, short4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, short, uchar, double, op>, FMXImpl_AOAOA<ushort2, short2, uchar2, double2, op>, FMXImpl_AOAOA<ushort3, short3, uchar3, double3, op>, FMXImpl_AOAOA<ushort4, short4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, short, schar, uchar, op>, FMXImpl_AOAOA<ushort2, short2, char2, uchar2, op>, FMXImpl_AOAOA<ushort3, short3, char3, uchar3, op>, FMXImpl_AOAOA<ushort4, short4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, short, schar, schar, op>, FMXImpl_AOAOA<ushort2, short2, char2, char2, op>, FMXImpl_AOAOA<ushort3, short3, char3, char3, op>, FMXImpl_AOAOA<ushort4, short4, char4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, short, schar, ushort, op>, FMXImpl_AOAOA<ushort2, short2, char2, ushort2, op>, FMXImpl_AOAOA<ushort3, short3, char3, ushort3, op>, FMXImpl_AOAOA<ushort4, short4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, short, schar, short, op>, FMXImpl_AOAOA<ushort2, short2, char2, short2, op>, FMXImpl_AOAOA<ushort3, short3, char3, short3, op>, FMXImpl_AOAOA<ushort4, short4, char4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, short, schar, int, op>, FMXImpl_AOAOA<ushort2, short2, char2, int2, op>, FMXImpl_AOAOA<ushort3, short3, char3, int3, op>, FMXImpl_AOAOA<ushort4, short4, char4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, short, schar, float, op>, FMXImpl_AOAOA<ushort2, short2, char2, float2, op>, FMXImpl_AOAOA<ushort3, short3, char3, float3, op>, FMXImpl_AOAOA<ushort4, short4, char4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, short, schar, double, op>, FMXImpl_AOAOA<ushort2, short2, char2, double2, op>, FMXImpl_AOAOA<ushort3, short3, char3, double3, op>, FMXImpl_AOAOA<ushort4, short4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, short, ushort, uchar, op>, FMXImpl_AOAOA<ushort2, short2, ushort2, uchar2, op>, FMXImpl_AOAOA<ushort3, short3, ushort3, uchar3, op>, FMXImpl_AOAOA<ushort4, short4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, short, ushort, schar, op>, FMXImpl_AOAOA<ushort2, short2, ushort2, char2, op>, FMXImpl_AOAOA<ushort3, short3, ushort3, char3, op>, FMXImpl_AOAOA<ushort4, short4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, short, ushort, ushort, op>, FMXImpl_AOAOA<ushort2, short2, ushort2, ushort2, op>, FMXImpl_AOAOA<ushort3, short3, ushort3, ushort3, op>, FMXImpl_AOAOA<ushort4, short4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, short, ushort, short, op>, FMXImpl_AOAOA<ushort2, short2, ushort2, short2, op>, FMXImpl_AOAOA<ushort3, short3, ushort3, short3, op>, FMXImpl_AOAOA<ushort4, short4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, short, ushort, int, op>, FMXImpl_AOAOA<ushort2, short2, ushort2, int2, op>, FMXImpl_AOAOA<ushort3, short3, ushort3, int3, op>, FMXImpl_AOAOA<ushort4, short4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, short, ushort, float, op>, FMXImpl_AOAOA<ushort2, short2, ushort2, float2, op>, FMXImpl_AOAOA<ushort3, short3, ushort3, float3, op>, FMXImpl_AOAOA<ushort4, short4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, short, ushort, double, op>, FMXImpl_AOAOA<ushort2, short2, ushort2, double2, op>, FMXImpl_AOAOA<ushort3, short3, ushort3, double3, op>, FMXImpl_AOAOA<ushort4, short4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, short, short, uchar, op>, FMXImpl_AOAOA<ushort2, short2, short2, uchar2, op>, FMXImpl_AOAOA<ushort3, short3, short3, uchar3, op>, FMXImpl_AOAOA<ushort4, short4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, short, short, schar, op>, FMXImpl_AOAOA<ushort2, short2, short2, char2, op>, FMXImpl_AOAOA<ushort3, short3, short3, char3, op>, FMXImpl_AOAOA<ushort4, short4, short4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, short, short, ushort, op>, FMXImpl_AOAOA<ushort2, short2, short2, ushort2, op>, FMXImpl_AOAOA<ushort3, short3, short3, ushort3, op>, FMXImpl_AOAOA<ushort4, short4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, short, short, short, op>, FMXImpl_AOAOA<ushort2, short2, short2, short2, op>, FMXImpl_AOAOA<ushort3, short3, short3, short3, op>, FMXImpl_AOAOA<ushort4, short4, short4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, short, short, int, op>, FMXImpl_AOAOA<ushort2, short2, short2, int2, op>, FMXImpl_AOAOA<ushort3, short3, short3, int3, op>, FMXImpl_AOAOA<ushort4, short4, short4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, short, short, float, op>, FMXImpl_AOAOA<ushort2, short2, short2, float2, op>, FMXImpl_AOAOA<ushort3, short3, short3, float3, op>, FMXImpl_AOAOA<ushort4, short4, short4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, short, short, double, op>, FMXImpl_AOAOA<ushort2, short2, short2, double2, op>, FMXImpl_AOAOA<ushort3, short3, short3, double3, op>, FMXImpl_AOAOA<ushort4, short4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, short, int, uchar, op>, FMXImpl_AOAOA<ushort2, short2, int2, uchar2, op>, FMXImpl_AOAOA<ushort3, short3, int3, uchar3, op>, FMXImpl_AOAOA<ushort4, short4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, short, int, schar, op>, FMXImpl_AOAOA<ushort2, short2, int2, char2, op>, FMXImpl_AOAOA<ushort3, short3, int3, char3, op>, FMXImpl_AOAOA<ushort4, short4, int4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, short, int, ushort, op>, FMXImpl_AOAOA<ushort2, short2, int2, ushort2, op>, FMXImpl_AOAOA<ushort3, short3, int3, ushort3, op>, FMXImpl_AOAOA<ushort4, short4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, short, int, short, op>, FMXImpl_AOAOA<ushort2, short2, int2, short2, op>, FMXImpl_AOAOA<ushort3, short3, int3, short3, op>, FMXImpl_AOAOA<ushort4, short4, int4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, short, int, int, op>, FMXImpl_AOAOA<ushort2, short2, int2, int2, op>, FMXImpl_AOAOA<ushort3, short3, int3, int3, op>, FMXImpl_AOAOA<ushort4, short4, int4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, short, int, float, op>, FMXImpl_AOAOA<ushort2, short2, int2, float2, op>, FMXImpl_AOAOA<ushort3, short3, int3, float3, op>, FMXImpl_AOAOA<ushort4, short4, int4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, short, int, double, op>, FMXImpl_AOAOA<ushort2, short2, int2, double2, op>, FMXImpl_AOAOA<ushort3, short3, int3, double3, op>, FMXImpl_AOAOA<ushort4, short4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, short, float, uchar, op>, FMXImpl_AOAOA<ushort2, short2, float2, uchar2, op>, FMXImpl_AOAOA<ushort3, short3, float3, uchar3, op>, FMXImpl_AOAOA<ushort4, short4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, short, float, schar, op>, FMXImpl_AOAOA<ushort2, short2, float2, char2, op>, FMXImpl_AOAOA<ushort3, short3, float3, char3, op>, FMXImpl_AOAOA<ushort4, short4, float4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, short, float, ushort, op>, FMXImpl_AOAOA<ushort2, short2, float2, ushort2, op>, FMXImpl_AOAOA<ushort3, short3, float3, ushort3, op>, FMXImpl_AOAOA<ushort4, short4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, short, float, short, op>, FMXImpl_AOAOA<ushort2, short2, float2, short2, op>, FMXImpl_AOAOA<ushort3, short3, float3, short3, op>, FMXImpl_AOAOA<ushort4, short4, float4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, short, float, int, op>, FMXImpl_AOAOA<ushort2, short2, float2, int2, op>, FMXImpl_AOAOA<ushort3, short3, float3, int3, op>, FMXImpl_AOAOA<ushort4, short4, float4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, short, float, float, op>, FMXImpl_AOAOA<ushort2, short2, float2, float2, op>, FMXImpl_AOAOA<ushort3, short3, float3, float3, op>, FMXImpl_AOAOA<ushort4, short4, float4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, short, float, double, op>, FMXImpl_AOAOA<ushort2, short2, float2, double2, op>, FMXImpl_AOAOA<ushort3, short3, float3, double3, op>, FMXImpl_AOAOA<ushort4, short4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, short, double, uchar, op>, FMXImpl_AOAOA<ushort2, short2, double2, uchar2, op>, FMXImpl_AOAOA<ushort3, short3, double3, uchar3, op>, FMXImpl_AOAOA<ushort4, short4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, short, double, schar, op>, FMXImpl_AOAOA<ushort2, short2, double2, char2, op>, FMXImpl_AOAOA<ushort3, short3, double3, char3, op>, FMXImpl_AOAOA<ushort4, short4, double4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, short, double, ushort, op>, FMXImpl_AOAOA<ushort2, short2, double2, ushort2, op>, FMXImpl_AOAOA<ushort3, short3, double3, ushort3, op>, FMXImpl_AOAOA<ushort4, short4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, short, double, short, op>, FMXImpl_AOAOA<ushort2, short2, double2, short2, op>, FMXImpl_AOAOA<ushort3, short3, double3, short3, op>, FMXImpl_AOAOA<ushort4, short4, double4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, short, double, int, op>, FMXImpl_AOAOA<ushort2, short2, double2, int2, op>, FMXImpl_AOAOA<ushort3, short3, double3, int3, op>, FMXImpl_AOAOA<ushort4, short4, double4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, short, double, float, op>, FMXImpl_AOAOA<ushort2, short2, double2, float2, op>, FMXImpl_AOAOA<ushort3, short3, double3, float3, op>, FMXImpl_AOAOA<ushort4, short4, double4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, short, double, double, op>, FMXImpl_AOAOA<ushort2, short2, double2, double2, op>, FMXImpl_AOAOA<ushort3, short3, double3, double3, op>, FMXImpl_AOAOA<ushort4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<ushort, int, uchar, uchar, op>, FMXImpl_AOAOA<ushort2, int2, uchar2, uchar2, op>, FMXImpl_AOAOA<ushort3, int3, uchar3, uchar3, op>, FMXImpl_AOAOA<ushort4, int4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, int, uchar, schar, op>, FMXImpl_AOAOA<ushort2, int2, uchar2, char2, op>, FMXImpl_AOAOA<ushort3, int3, uchar3, char3, op>, FMXImpl_AOAOA<ushort4, int4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, int, uchar, ushort, op>, FMXImpl_AOAOA<ushort2, int2, uchar2, ushort2, op>, FMXImpl_AOAOA<ushort3, int3, uchar3, ushort3, op>, FMXImpl_AOAOA<ushort4, int4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, int, uchar, short, op>, FMXImpl_AOAOA<ushort2, int2, uchar2, short2, op>, FMXImpl_AOAOA<ushort3, int3, uchar3, short3, op>, FMXImpl_AOAOA<ushort4, int4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, int, uchar, int, op>, FMXImpl_AOAOA<ushort2, int2, uchar2, int2, op>, FMXImpl_AOAOA<ushort3, int3, uchar3, int3, op>, FMXImpl_AOAOA<ushort4, int4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, int, uchar, float, op>, FMXImpl_AOAOA<ushort2, int2, uchar2, float2, op>, FMXImpl_AOAOA<ushort3, int3, uchar3, float3, op>, FMXImpl_AOAOA<ushort4, int4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, int, uchar, double, op>, FMXImpl_AOAOA<ushort2, int2, uchar2, double2, op>, FMXImpl_AOAOA<ushort3, int3, uchar3, double3, op>, FMXImpl_AOAOA<ushort4, int4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, int, schar, uchar, op>, FMXImpl_AOAOA<ushort2, int2, char2, uchar2, op>, FMXImpl_AOAOA<ushort3, int3, char3, uchar3, op>, FMXImpl_AOAOA<ushort4, int4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, int, schar, schar, op>, FMXImpl_AOAOA<ushort2, int2, char2, char2, op>, FMXImpl_AOAOA<ushort3, int3, char3, char3, op>, FMXImpl_AOAOA<ushort4, int4, char4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, int, schar, ushort, op>, FMXImpl_AOAOA<ushort2, int2, char2, ushort2, op>, FMXImpl_AOAOA<ushort3, int3, char3, ushort3, op>, FMXImpl_AOAOA<ushort4, int4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, int, schar, short, op>, FMXImpl_AOAOA<ushort2, int2, char2, short2, op>, FMXImpl_AOAOA<ushort3, int3, char3, short3, op>, FMXImpl_AOAOA<ushort4, int4, char4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, int, schar, int, op>, FMXImpl_AOAOA<ushort2, int2, char2, int2, op>, FMXImpl_AOAOA<ushort3, int3, char3, int3, op>, FMXImpl_AOAOA<ushort4, int4, char4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, int, schar, float, op>, FMXImpl_AOAOA<ushort2, int2, char2, float2, op>, FMXImpl_AOAOA<ushort3, int3, char3, float3, op>, FMXImpl_AOAOA<ushort4, int4, char4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, int, schar, double, op>, FMXImpl_AOAOA<ushort2, int2, char2, double2, op>, FMXImpl_AOAOA<ushort3, int3, char3, double3, op>, FMXImpl_AOAOA<ushort4, int4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, int, ushort, uchar, op>, FMXImpl_AOAOA<ushort2, int2, ushort2, uchar2, op>, FMXImpl_AOAOA<ushort3, int3, ushort3, uchar3, op>, FMXImpl_AOAOA<ushort4, int4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, int, ushort, schar, op>, FMXImpl_AOAOA<ushort2, int2, ushort2, char2, op>, FMXImpl_AOAOA<ushort3, int3, ushort3, char3, op>, FMXImpl_AOAOA<ushort4, int4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, int, ushort, ushort, op>, FMXImpl_AOAOA<ushort2, int2, ushort2, ushort2, op>, FMXImpl_AOAOA<ushort3, int3, ushort3, ushort3, op>, FMXImpl_AOAOA<ushort4, int4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, int, ushort, short, op>, FMXImpl_AOAOA<ushort2, int2, ushort2, short2, op>, FMXImpl_AOAOA<ushort3, int3, ushort3, short3, op>, FMXImpl_AOAOA<ushort4, int4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, int, ushort, int, op>, FMXImpl_AOAOA<ushort2, int2, ushort2, int2, op>, FMXImpl_AOAOA<ushort3, int3, ushort3, int3, op>, FMXImpl_AOAOA<ushort4, int4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, int, ushort, float, op>, FMXImpl_AOAOA<ushort2, int2, ushort2, float2, op>, FMXImpl_AOAOA<ushort3, int3, ushort3, float3, op>, FMXImpl_AOAOA<ushort4, int4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, int, ushort, double, op>, FMXImpl_AOAOA<ushort2, int2, ushort2, double2, op>, FMXImpl_AOAOA<ushort3, int3, ushort3, double3, op>, FMXImpl_AOAOA<ushort4, int4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, int, short, uchar, op>, FMXImpl_AOAOA<ushort2, int2, short2, uchar2, op>, FMXImpl_AOAOA<ushort3, int3, short3, uchar3, op>, FMXImpl_AOAOA<ushort4, int4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, int, short, schar, op>, FMXImpl_AOAOA<ushort2, int2, short2, char2, op>, FMXImpl_AOAOA<ushort3, int3, short3, char3, op>, FMXImpl_AOAOA<ushort4, int4, short4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, int, short, ushort, op>, FMXImpl_AOAOA<ushort2, int2, short2, ushort2, op>, FMXImpl_AOAOA<ushort3, int3, short3, ushort3, op>, FMXImpl_AOAOA<ushort4, int4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, int, short, short, op>, FMXImpl_AOAOA<ushort2, int2, short2, short2, op>, FMXImpl_AOAOA<ushort3, int3, short3, short3, op>, FMXImpl_AOAOA<ushort4, int4, short4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, int, short, int, op>, FMXImpl_AOAOA<ushort2, int2, short2, int2, op>, FMXImpl_AOAOA<ushort3, int3, short3, int3, op>, FMXImpl_AOAOA<ushort4, int4, short4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, int, short, float, op>, FMXImpl_AOAOA<ushort2, int2, short2, float2, op>, FMXImpl_AOAOA<ushort3, int3, short3, float3, op>, FMXImpl_AOAOA<ushort4, int4, short4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, int, short, double, op>, FMXImpl_AOAOA<ushort2, int2, short2, double2, op>, FMXImpl_AOAOA<ushort3, int3, short3, double3, op>, FMXImpl_AOAOA<ushort4, int4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, int, int, uchar, op>, FMXImpl_AOAOA<ushort2, int2, int2, uchar2, op>, FMXImpl_AOAOA<ushort3, int3, int3, uchar3, op>, FMXImpl_AOAOA<ushort4, int4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, int, int, schar, op>, FMXImpl_AOAOA<ushort2, int2, int2, char2, op>, FMXImpl_AOAOA<ushort3, int3, int3, char3, op>, FMXImpl_AOAOA<ushort4, int4, int4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, int, int, ushort, op>, FMXImpl_AOAOA<ushort2, int2, int2, ushort2, op>, FMXImpl_AOAOA<ushort3, int3, int3, ushort3, op>, FMXImpl_AOAOA<ushort4, int4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, int, int, short, op>, FMXImpl_AOAOA<ushort2, int2, int2, short2, op>, FMXImpl_AOAOA<ushort3, int3, int3, short3, op>, FMXImpl_AOAOA<ushort4, int4, int4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, int, int, int, op>, FMXImpl_AOAOA<ushort2, int2, int2, int2, op>, FMXImpl_AOAOA<ushort3, int3, int3, int3, op>, FMXImpl_AOAOA<ushort4, int4, int4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, int, int, float, op>, FMXImpl_AOAOA<ushort2, int2, int2, float2, op>, FMXImpl_AOAOA<ushort3, int3, int3, float3, op>, FMXImpl_AOAOA<ushort4, int4, int4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, int, int, double, op>, FMXImpl_AOAOA<ushort2, int2, int2, double2, op>, FMXImpl_AOAOA<ushort3, int3, int3, double3, op>, FMXImpl_AOAOA<ushort4, int4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, int, float, uchar, op>, FMXImpl_AOAOA<ushort2, int2, float2, uchar2, op>, FMXImpl_AOAOA<ushort3, int3, float3, uchar3, op>, FMXImpl_AOAOA<ushort4, int4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, int, float, schar, op>, FMXImpl_AOAOA<ushort2, int2, float2, char2, op>, FMXImpl_AOAOA<ushort3, int3, float3, char3, op>, FMXImpl_AOAOA<ushort4, int4, float4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, int, float, ushort, op>, FMXImpl_AOAOA<ushort2, int2, float2, ushort2, op>, FMXImpl_AOAOA<ushort3, int3, float3, ushort3, op>, FMXImpl_AOAOA<ushort4, int4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, int, float, short, op>, FMXImpl_AOAOA<ushort2, int2, float2, short2, op>, FMXImpl_AOAOA<ushort3, int3, float3, short3, op>, FMXImpl_AOAOA<ushort4, int4, float4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, int, float, int, op>, FMXImpl_AOAOA<ushort2, int2, float2, int2, op>, FMXImpl_AOAOA<ushort3, int3, float3, int3, op>, FMXImpl_AOAOA<ushort4, int4, float4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, int, float, float, op>, FMXImpl_AOAOA<ushort2, int2, float2, float2, op>, FMXImpl_AOAOA<ushort3, int3, float3, float3, op>, FMXImpl_AOAOA<ushort4, int4, float4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, int, float, double, op>, FMXImpl_AOAOA<ushort2, int2, float2, double2, op>, FMXImpl_AOAOA<ushort3, int3, float3, double3, op>, FMXImpl_AOAOA<ushort4, int4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, int, double, uchar, op>, FMXImpl_AOAOA<ushort2, int2, double2, uchar2, op>, FMXImpl_AOAOA<ushort3, int3, double3, uchar3, op>, FMXImpl_AOAOA<ushort4, int4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, int, double, schar, op>, FMXImpl_AOAOA<ushort2, int2, double2, char2, op>, FMXImpl_AOAOA<ushort3, int3, double3, char3, op>, FMXImpl_AOAOA<ushort4, int4, double4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, int, double, ushort, op>, FMXImpl_AOAOA<ushort2, int2, double2, ushort2, op>, FMXImpl_AOAOA<ushort3, int3, double3, ushort3, op>, FMXImpl_AOAOA<ushort4, int4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, int, double, short, op>, FMXImpl_AOAOA<ushort2, int2, double2, short2, op>, FMXImpl_AOAOA<ushort3, int3, double3, short3, op>, FMXImpl_AOAOA<ushort4, int4, double4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, int, double, int, op>, FMXImpl_AOAOA<ushort2, int2, double2, int2, op>, FMXImpl_AOAOA<ushort3, int3, double3, int3, op>, FMXImpl_AOAOA<ushort4, int4, double4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, int, double, float, op>, FMXImpl_AOAOA<ushort2, int2, double2, float2, op>, FMXImpl_AOAOA<ushort3, int3, double3, float3, op>, FMXImpl_AOAOA<ushort4, int4, double4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, int, double, double, op>, FMXImpl_AOAOA<ushort2, int2, double2, double2, op>, FMXImpl_AOAOA<ushort3, int3, double3, double3, op>, FMXImpl_AOAOA<ushort4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<ushort, float, uchar, uchar, op>, FMXImpl_AOAOA<ushort2, float2, uchar2, uchar2, op>, FMXImpl_AOAOA<ushort3, float3, uchar3, uchar3, op>, FMXImpl_AOAOA<ushort4, float4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, float, uchar, schar, op>, FMXImpl_AOAOA<ushort2, float2, uchar2, char2, op>, FMXImpl_AOAOA<ushort3, float3, uchar3, char3, op>, FMXImpl_AOAOA<ushort4, float4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, float, uchar, ushort, op>, FMXImpl_AOAOA<ushort2, float2, uchar2, ushort2, op>, FMXImpl_AOAOA<ushort3, float3, uchar3, ushort3, op>, FMXImpl_AOAOA<ushort4, float4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, float, uchar, short, op>, FMXImpl_AOAOA<ushort2, float2, uchar2, short2, op>, FMXImpl_AOAOA<ushort3, float3, uchar3, short3, op>, FMXImpl_AOAOA<ushort4, float4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, float, uchar, int, op>, FMXImpl_AOAOA<ushort2, float2, uchar2, int2, op>, FMXImpl_AOAOA<ushort3, float3, uchar3, int3, op>, FMXImpl_AOAOA<ushort4, float4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, float, uchar, float, op>, FMXImpl_AOAOA<ushort2, float2, uchar2, float2, op>, FMXImpl_AOAOA<ushort3, float3, uchar3, float3, op>, FMXImpl_AOAOA<ushort4, float4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, float, uchar, double, op>, FMXImpl_AOAOA<ushort2, float2, uchar2, double2, op>, FMXImpl_AOAOA<ushort3, float3, uchar3, double3, op>, FMXImpl_AOAOA<ushort4, float4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, float, schar, uchar, op>, FMXImpl_AOAOA<ushort2, float2, char2, uchar2, op>, FMXImpl_AOAOA<ushort3, float3, char3, uchar3, op>, FMXImpl_AOAOA<ushort4, float4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, float, schar, schar, op>, FMXImpl_AOAOA<ushort2, float2, char2, char2, op>, FMXImpl_AOAOA<ushort3, float3, char3, char3, op>, FMXImpl_AOAOA<ushort4, float4, char4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, float, schar, ushort, op>, FMXImpl_AOAOA<ushort2, float2, char2, ushort2, op>, FMXImpl_AOAOA<ushort3, float3, char3, ushort3, op>, FMXImpl_AOAOA<ushort4, float4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, float, schar, short, op>, FMXImpl_AOAOA<ushort2, float2, char2, short2, op>, FMXImpl_AOAOA<ushort3, float3, char3, short3, op>, FMXImpl_AOAOA<ushort4, float4, char4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, float, schar, int, op>, FMXImpl_AOAOA<ushort2, float2, char2, int2, op>, FMXImpl_AOAOA<ushort3, float3, char3, int3, op>, FMXImpl_AOAOA<ushort4, float4, char4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, float, schar, float, op>, FMXImpl_AOAOA<ushort2, float2, char2, float2, op>, FMXImpl_AOAOA<ushort3, float3, char3, float3, op>, FMXImpl_AOAOA<ushort4, float4, char4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, float, schar, double, op>, FMXImpl_AOAOA<ushort2, float2, char2, double2, op>, FMXImpl_AOAOA<ushort3, float3, char3, double3, op>, FMXImpl_AOAOA<ushort4, float4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, float, ushort, uchar, op>, FMXImpl_AOAOA<ushort2, float2, ushort2, uchar2, op>, FMXImpl_AOAOA<ushort3, float3, ushort3, uchar3, op>, FMXImpl_AOAOA<ushort4, float4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, float, ushort, schar, op>, FMXImpl_AOAOA<ushort2, float2, ushort2, char2, op>, FMXImpl_AOAOA<ushort3, float3, ushort3, char3, op>, FMXImpl_AOAOA<ushort4, float4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, float, ushort, ushort, op>, FMXImpl_AOAOA<ushort2, float2, ushort2, ushort2, op>, FMXImpl_AOAOA<ushort3, float3, ushort3, ushort3, op>, FMXImpl_AOAOA<ushort4, float4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, float, ushort, short, op>, FMXImpl_AOAOA<ushort2, float2, ushort2, short2, op>, FMXImpl_AOAOA<ushort3, float3, ushort3, short3, op>, FMXImpl_AOAOA<ushort4, float4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, float, ushort, int, op>, FMXImpl_AOAOA<ushort2, float2, ushort2, int2, op>, FMXImpl_AOAOA<ushort3, float3, ushort3, int3, op>, FMXImpl_AOAOA<ushort4, float4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, float, ushort, float, op>, FMXImpl_AOAOA<ushort2, float2, ushort2, float2, op>, FMXImpl_AOAOA<ushort3, float3, ushort3, float3, op>, FMXImpl_AOAOA<ushort4, float4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, float, ushort, double, op>, FMXImpl_AOAOA<ushort2, float2, ushort2, double2, op>, FMXImpl_AOAOA<ushort3, float3, ushort3, double3, op>, FMXImpl_AOAOA<ushort4, float4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, float, short, uchar, op>, FMXImpl_AOAOA<ushort2, float2, short2, uchar2, op>, FMXImpl_AOAOA<ushort3, float3, short3, uchar3, op>, FMXImpl_AOAOA<ushort4, float4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, float, short, schar, op>, FMXImpl_AOAOA<ushort2, float2, short2, char2, op>, FMXImpl_AOAOA<ushort3, float3, short3, char3, op>, FMXImpl_AOAOA<ushort4, float4, short4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, float, short, ushort, op>, FMXImpl_AOAOA<ushort2, float2, short2, ushort2, op>, FMXImpl_AOAOA<ushort3, float3, short3, ushort3, op>, FMXImpl_AOAOA<ushort4, float4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, float, short, short, op>, FMXImpl_AOAOA<ushort2, float2, short2, short2, op>, FMXImpl_AOAOA<ushort3, float3, short3, short3, op>, FMXImpl_AOAOA<ushort4, float4, short4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, float, short, int, op>, FMXImpl_AOAOA<ushort2, float2, short2, int2, op>, FMXImpl_AOAOA<ushort3, float3, short3, int3, op>, FMXImpl_AOAOA<ushort4, float4, short4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, float, short, float, op>, FMXImpl_AOAOA<ushort2, float2, short2, float2, op>, FMXImpl_AOAOA<ushort3, float3, short3, float3, op>, FMXImpl_AOAOA<ushort4, float4, short4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, float, short, double, op>, FMXImpl_AOAOA<ushort2, float2, short2, double2, op>, FMXImpl_AOAOA<ushort3, float3, short3, double3, op>, FMXImpl_AOAOA<ushort4, float4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, float, int, uchar, op>, FMXImpl_AOAOA<ushort2, float2, int2, uchar2, op>, FMXImpl_AOAOA<ushort3, float3, int3, uchar3, op>, FMXImpl_AOAOA<ushort4, float4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, float, int, schar, op>, FMXImpl_AOAOA<ushort2, float2, int2, char2, op>, FMXImpl_AOAOA<ushort3, float3, int3, char3, op>, FMXImpl_AOAOA<ushort4, float4, int4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, float, int, ushort, op>, FMXImpl_AOAOA<ushort2, float2, int2, ushort2, op>, FMXImpl_AOAOA<ushort3, float3, int3, ushort3, op>, FMXImpl_AOAOA<ushort4, float4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, float, int, short, op>, FMXImpl_AOAOA<ushort2, float2, int2, short2, op>, FMXImpl_AOAOA<ushort3, float3, int3, short3, op>, FMXImpl_AOAOA<ushort4, float4, int4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, float, int, int, op>, FMXImpl_AOAOA<ushort2, float2, int2, int2, op>, FMXImpl_AOAOA<ushort3, float3, int3, int3, op>, FMXImpl_AOAOA<ushort4, float4, int4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, float, int, float, op>, FMXImpl_AOAOA<ushort2, float2, int2, float2, op>, FMXImpl_AOAOA<ushort3, float3, int3, float3, op>, FMXImpl_AOAOA<ushort4, float4, int4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, float, int, double, op>, FMXImpl_AOAOA<ushort2, float2, int2, double2, op>, FMXImpl_AOAOA<ushort3, float3, int3, double3, op>, FMXImpl_AOAOA<ushort4, float4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, float, float, uchar, op>, FMXImpl_AOAOA<ushort2, float2, float2, uchar2, op>, FMXImpl_AOAOA<ushort3, float3, float3, uchar3, op>, FMXImpl_AOAOA<ushort4, float4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, float, float, schar, op>, FMXImpl_AOAOA<ushort2, float2, float2, char2, op>, FMXImpl_AOAOA<ushort3, float3, float3, char3, op>, FMXImpl_AOAOA<ushort4, float4, float4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, float, float, ushort, op>, FMXImpl_AOAOA<ushort2, float2, float2, ushort2, op>, FMXImpl_AOAOA<ushort3, float3, float3, ushort3, op>, FMXImpl_AOAOA<ushort4, float4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, float, float, short, op>, FMXImpl_AOAOA<ushort2, float2, float2, short2, op>, FMXImpl_AOAOA<ushort3, float3, float3, short3, op>, FMXImpl_AOAOA<ushort4, float4, float4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, float, float, int, op>, FMXImpl_AOAOA<ushort2, float2, float2, int2, op>, FMXImpl_AOAOA<ushort3, float3, float3, int3, op>, FMXImpl_AOAOA<ushort4, float4, float4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, float, float, float, op>, FMXImpl_AOAOA<ushort2, float2, float2, float2, op>, FMXImpl_AOAOA<ushort3, float3, float3, float3, op>, FMXImpl_AOAOA<ushort4, float4, float4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, float, float, double, op>, FMXImpl_AOAOA<ushort2, float2, float2, double2, op>, FMXImpl_AOAOA<ushort3, float3, float3, double3, op>, FMXImpl_AOAOA<ushort4, float4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, float, double, uchar, op>, FMXImpl_AOAOA<ushort2, float2, double2, uchar2, op>, FMXImpl_AOAOA<ushort3, float3, double3, uchar3, op>, FMXImpl_AOAOA<ushort4, float4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, float, double, schar, op>, FMXImpl_AOAOA<ushort2, float2, double2, char2, op>, FMXImpl_AOAOA<ushort3, float3, double3, char3, op>, FMXImpl_AOAOA<ushort4, float4, double4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, float, double, ushort, op>, FMXImpl_AOAOA<ushort2, float2, double2, ushort2, op>, FMXImpl_AOAOA<ushort3, float3, double3, ushort3, op>, FMXImpl_AOAOA<ushort4, float4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, float, double, short, op>, FMXImpl_AOAOA<ushort2, float2, double2, short2, op>, FMXImpl_AOAOA<ushort3, float3, double3, short3, op>, FMXImpl_AOAOA<ushort4, float4, double4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, float, double, int, op>, FMXImpl_AOAOA<ushort2, float2, double2, int2, op>, FMXImpl_AOAOA<ushort3, float3, double3, int3, op>, FMXImpl_AOAOA<ushort4, float4, double4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, float, double, float, op>, FMXImpl_AOAOA<ushort2, float2, double2, float2, op>, FMXImpl_AOAOA<ushort3, float3, double3, float3, op>, FMXImpl_AOAOA<ushort4, float4, double4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, float, double, double, op>, FMXImpl_AOAOA<ushort2, float2, double2, double2, op>, FMXImpl_AOAOA<ushort3, float3, double3, double3, op>, FMXImpl_AOAOA<ushort4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<ushort, double, uchar, uchar, op>, FMXImpl_AOAOA<ushort2, double2, uchar2, uchar2, op>, FMXImpl_AOAOA<ushort3, double3, uchar3, uchar3, op>, FMXImpl_AOAOA<ushort4, double4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, double, uchar, schar, op>, FMXImpl_AOAOA<ushort2, double2, uchar2, char2, op>, FMXImpl_AOAOA<ushort3, double3, uchar3, char3, op>, FMXImpl_AOAOA<ushort4, double4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, double, uchar, ushort, op>, FMXImpl_AOAOA<ushort2, double2, uchar2, ushort2, op>, FMXImpl_AOAOA<ushort3, double3, uchar3, ushort3, op>, FMXImpl_AOAOA<ushort4, double4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, double, uchar, short, op>, FMXImpl_AOAOA<ushort2, double2, uchar2, short2, op>, FMXImpl_AOAOA<ushort3, double3, uchar3, short3, op>, FMXImpl_AOAOA<ushort4, double4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, double, uchar, int, op>, FMXImpl_AOAOA<ushort2, double2, uchar2, int2, op>, FMXImpl_AOAOA<ushort3, double3, uchar3, int3, op>, FMXImpl_AOAOA<ushort4, double4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, double, uchar, float, op>, FMXImpl_AOAOA<ushort2, double2, uchar2, float2, op>, FMXImpl_AOAOA<ushort3, double3, uchar3, float3, op>, FMXImpl_AOAOA<ushort4, double4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, double, uchar, double, op>, FMXImpl_AOAOA<ushort2, double2, uchar2, double2, op>, FMXImpl_AOAOA<ushort3, double3, uchar3, double3, op>, FMXImpl_AOAOA<ushort4, double4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, double, schar, uchar, op>, FMXImpl_AOAOA<ushort2, double2, char2, uchar2, op>, FMXImpl_AOAOA<ushort3, double3, char3, uchar3, op>, FMXImpl_AOAOA<ushort4, double4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, double, schar, schar, op>, FMXImpl_AOAOA<ushort2, double2, char2, char2, op>, FMXImpl_AOAOA<ushort3, double3, char3, char3, op>, FMXImpl_AOAOA<ushort4, double4, char4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, double, schar, ushort, op>, FMXImpl_AOAOA<ushort2, double2, char2, ushort2, op>, FMXImpl_AOAOA<ushort3, double3, char3, ushort3, op>, FMXImpl_AOAOA<ushort4, double4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, double, schar, short, op>, FMXImpl_AOAOA<ushort2, double2, char2, short2, op>, FMXImpl_AOAOA<ushort3, double3, char3, short3, op>, FMXImpl_AOAOA<ushort4, double4, char4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, double, schar, int, op>, FMXImpl_AOAOA<ushort2, double2, char2, int2, op>, FMXImpl_AOAOA<ushort3, double3, char3, int3, op>, FMXImpl_AOAOA<ushort4, double4, char4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, double, schar, float, op>, FMXImpl_AOAOA<ushort2, double2, char2, float2, op>, FMXImpl_AOAOA<ushort3, double3, char3, float3, op>, FMXImpl_AOAOA<ushort4, double4, char4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, double, schar, double, op>, FMXImpl_AOAOA<ushort2, double2, char2, double2, op>, FMXImpl_AOAOA<ushort3, double3, char3, double3, op>, FMXImpl_AOAOA<ushort4, double4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, double, ushort, uchar, op>, FMXImpl_AOAOA<ushort2, double2, ushort2, uchar2, op>, FMXImpl_AOAOA<ushort3, double3, ushort3, uchar3, op>, FMXImpl_AOAOA<ushort4, double4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, double, ushort, schar, op>, FMXImpl_AOAOA<ushort2, double2, ushort2, char2, op>, FMXImpl_AOAOA<ushort3, double3, ushort3, char3, op>, FMXImpl_AOAOA<ushort4, double4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, double, ushort, ushort, op>, FMXImpl_AOAOA<ushort2, double2, ushort2, ushort2, op>, FMXImpl_AOAOA<ushort3, double3, ushort3, ushort3, op>, FMXImpl_AOAOA<ushort4, double4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, double, ushort, short, op>, FMXImpl_AOAOA<ushort2, double2, ushort2, short2, op>, FMXImpl_AOAOA<ushort3, double3, ushort3, short3, op>, FMXImpl_AOAOA<ushort4, double4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, double, ushort, int, op>, FMXImpl_AOAOA<ushort2, double2, ushort2, int2, op>, FMXImpl_AOAOA<ushort3, double3, ushort3, int3, op>, FMXImpl_AOAOA<ushort4, double4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, double, ushort, float, op>, FMXImpl_AOAOA<ushort2, double2, ushort2, float2, op>, FMXImpl_AOAOA<ushort3, double3, ushort3, float3, op>, FMXImpl_AOAOA<ushort4, double4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, double, ushort, double, op>, FMXImpl_AOAOA<ushort2, double2, ushort2, double2, op>, FMXImpl_AOAOA<ushort3, double3, ushort3, double3, op>, FMXImpl_AOAOA<ushort4, double4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, double, short, uchar, op>, FMXImpl_AOAOA<ushort2, double2, short2, uchar2, op>, FMXImpl_AOAOA<ushort3, double3, short3, uchar3, op>, FMXImpl_AOAOA<ushort4, double4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, double, short, schar, op>, FMXImpl_AOAOA<ushort2, double2, short2, char2, op>, FMXImpl_AOAOA<ushort3, double3, short3, char3, op>, FMXImpl_AOAOA<ushort4, double4, short4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, double, short, ushort, op>, FMXImpl_AOAOA<ushort2, double2, short2, ushort2, op>, FMXImpl_AOAOA<ushort3, double3, short3, ushort3, op>, FMXImpl_AOAOA<ushort4, double4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, double, short, short, op>, FMXImpl_AOAOA<ushort2, double2, short2, short2, op>, FMXImpl_AOAOA<ushort3, double3, short3, short3, op>, FMXImpl_AOAOA<ushort4, double4, short4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, double, short, int, op>, FMXImpl_AOAOA<ushort2, double2, short2, int2, op>, FMXImpl_AOAOA<ushort3, double3, short3, int3, op>, FMXImpl_AOAOA<ushort4, double4, short4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, double, short, float, op>, FMXImpl_AOAOA<ushort2, double2, short2, float2, op>, FMXImpl_AOAOA<ushort3, double3, short3, float3, op>, FMXImpl_AOAOA<ushort4, double4, short4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, double, short, double, op>, FMXImpl_AOAOA<ushort2, double2, short2, double2, op>, FMXImpl_AOAOA<ushort3, double3, short3, double3, op>, FMXImpl_AOAOA<ushort4, double4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, double, int, uchar, op>, FMXImpl_AOAOA<ushort2, double2, int2, uchar2, op>, FMXImpl_AOAOA<ushort3, double3, int3, uchar3, op>, FMXImpl_AOAOA<ushort4, double4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, double, int, schar, op>, FMXImpl_AOAOA<ushort2, double2, int2, char2, op>, FMXImpl_AOAOA<ushort3, double3, int3, char3, op>, FMXImpl_AOAOA<ushort4, double4, int4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, double, int, ushort, op>, FMXImpl_AOAOA<ushort2, double2, int2, ushort2, op>, FMXImpl_AOAOA<ushort3, double3, int3, ushort3, op>, FMXImpl_AOAOA<ushort4, double4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, double, int, short, op>, FMXImpl_AOAOA<ushort2, double2, int2, short2, op>, FMXImpl_AOAOA<ushort3, double3, int3, short3, op>, FMXImpl_AOAOA<ushort4, double4, int4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, double, int, int, op>, FMXImpl_AOAOA<ushort2, double2, int2, int2, op>, FMXImpl_AOAOA<ushort3, double3, int3, int3, op>, FMXImpl_AOAOA<ushort4, double4, int4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, double, int, float, op>, FMXImpl_AOAOA<ushort2, double2, int2, float2, op>, FMXImpl_AOAOA<ushort3, double3, int3, float3, op>, FMXImpl_AOAOA<ushort4, double4, int4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, double, int, double, op>, FMXImpl_AOAOA<ushort2, double2, int2, double2, op>, FMXImpl_AOAOA<ushort3, double3, int3, double3, op>, FMXImpl_AOAOA<ushort4, double4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, double, float, uchar, op>, FMXImpl_AOAOA<ushort2, double2, float2, uchar2, op>, FMXImpl_AOAOA<ushort3, double3, float3, uchar3, op>, FMXImpl_AOAOA<ushort4, double4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, double, float, schar, op>, FMXImpl_AOAOA<ushort2, double2, float2, char2, op>, FMXImpl_AOAOA<ushort3, double3, float3, char3, op>, FMXImpl_AOAOA<ushort4, double4, float4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, double, float, ushort, op>, FMXImpl_AOAOA<ushort2, double2, float2, ushort2, op>, FMXImpl_AOAOA<ushort3, double3, float3, ushort3, op>, FMXImpl_AOAOA<ushort4, double4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, double, float, short, op>, FMXImpl_AOAOA<ushort2, double2, float2, short2, op>, FMXImpl_AOAOA<ushort3, double3, float3, short3, op>, FMXImpl_AOAOA<ushort4, double4, float4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, double, float, int, op>, FMXImpl_AOAOA<ushort2, double2, float2, int2, op>, FMXImpl_AOAOA<ushort3, double3, float3, int3, op>, FMXImpl_AOAOA<ushort4, double4, float4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, double, float, float, op>, FMXImpl_AOAOA<ushort2, double2, float2, float2, op>, FMXImpl_AOAOA<ushort3, double3, float3, float3, op>, FMXImpl_AOAOA<ushort4, double4, float4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, double, float, double, op>, FMXImpl_AOAOA<ushort2, double2, float2, double2, op>, FMXImpl_AOAOA<ushort3, double3, float3, double3, op>, FMXImpl_AOAOA<ushort4, double4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<ushort, double, double, uchar, op>, FMXImpl_AOAOA<ushort2, double2, double2, uchar2, op>, FMXImpl_AOAOA<ushort3, double3, double3, uchar3, op>, FMXImpl_AOAOA<ushort4, double4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<ushort, double, double, schar, op>, FMXImpl_AOAOA<ushort2, double2, double2, char2, op>, FMXImpl_AOAOA<ushort3, double3, double3, char3, op>, FMXImpl_AOAOA<ushort4, double4, double4, char4, op>  },
                    { FMXImpl_AOAOA<ushort, double, double, ushort, op>, FMXImpl_AOAOA<ushort2, double2, double2, ushort2, op>, FMXImpl_AOAOA<ushort3, double3, double3, ushort3, op>, FMXImpl_AOAOA<ushort4, double4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<ushort, double, double, short, op>, FMXImpl_AOAOA<ushort2, double2, double2, short2, op>, FMXImpl_AOAOA<ushort3, double3, double3, short3, op>, FMXImpl_AOAOA<ushort4, double4, double4, short4, op>  },
                    { FMXImpl_AOAOA<ushort, double, double, int, op>, FMXImpl_AOAOA<ushort2, double2, double2, int2, op>, FMXImpl_AOAOA<ushort3, double3, double3, int3, op>, FMXImpl_AOAOA<ushort4, double4, double4, int4, op>  },
                    { FMXImpl_AOAOA<ushort, double, double, float, op>, FMXImpl_AOAOA<ushort2, double2, double2, float2, op>, FMXImpl_AOAOA<ushort3, double3, double3, float3, op>, FMXImpl_AOAOA<ushort4, double4, double4, float4, op>  },
                    { FMXImpl_AOAOA<ushort, double, double, double, op>, FMXImpl_AOAOA<ushort2, double2, double2, double2, op>, FMXImpl_AOAOA<ushort3, double3, double3, double3, op>, FMXImpl_AOAOA<ushort4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { FMXImpl_AOAOA<short, uchar, uchar, uchar, op>, FMXImpl_AOAOA<short2, uchar2, uchar2, uchar2, op>, FMXImpl_AOAOA<short3, uchar3, uchar3, uchar3, op>, FMXImpl_AOAOA<short4, uchar4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, uchar, uchar, schar, op>, FMXImpl_AOAOA<short2, uchar2, uchar2, char2, op>, FMXImpl_AOAOA<short3, uchar3, uchar3, char3, op>, FMXImpl_AOAOA<short4, uchar4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<short, uchar, uchar, ushort, op>, FMXImpl_AOAOA<short2, uchar2, uchar2, ushort2, op>, FMXImpl_AOAOA<short3, uchar3, uchar3, ushort3, op>, FMXImpl_AOAOA<short4, uchar4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, uchar, uchar, short, op>, FMXImpl_AOAOA<short2, uchar2, uchar2, short2, op>, FMXImpl_AOAOA<short3, uchar3, uchar3, short3, op>, FMXImpl_AOAOA<short4, uchar4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<short, uchar, uchar, int, op>, FMXImpl_AOAOA<short2, uchar2, uchar2, int2, op>, FMXImpl_AOAOA<short3, uchar3, uchar3, int3, op>, FMXImpl_AOAOA<short4, uchar4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<short, uchar, uchar, float, op>, FMXImpl_AOAOA<short2, uchar2, uchar2, float2, op>, FMXImpl_AOAOA<short3, uchar3, uchar3, float3, op>, FMXImpl_AOAOA<short4, uchar4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<short, uchar, uchar, double, op>, FMXImpl_AOAOA<short2, uchar2, uchar2, double2, op>, FMXImpl_AOAOA<short3, uchar3, uchar3, double3, op>, FMXImpl_AOAOA<short4, uchar4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, uchar, schar, uchar, op>, FMXImpl_AOAOA<short2, uchar2, char2, uchar2, op>, FMXImpl_AOAOA<short3, uchar3, char3, uchar3, op>, FMXImpl_AOAOA<short4, uchar4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, uchar, schar, schar, op>, FMXImpl_AOAOA<short2, uchar2, char2, char2, op>, FMXImpl_AOAOA<short3, uchar3, char3, char3, op>, FMXImpl_AOAOA<short4, uchar4, char4, char4, op>  },
                    { FMXImpl_AOAOA<short, uchar, schar, ushort, op>, FMXImpl_AOAOA<short2, uchar2, char2, ushort2, op>, FMXImpl_AOAOA<short3, uchar3, char3, ushort3, op>, FMXImpl_AOAOA<short4, uchar4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, uchar, schar, short, op>, FMXImpl_AOAOA<short2, uchar2, char2, short2, op>, FMXImpl_AOAOA<short3, uchar3, char3, short3, op>, FMXImpl_AOAOA<short4, uchar4, char4, short4, op>  },
                    { FMXImpl_AOAOA<short, uchar, schar, int, op>, FMXImpl_AOAOA<short2, uchar2, char2, int2, op>, FMXImpl_AOAOA<short3, uchar3, char3, int3, op>, FMXImpl_AOAOA<short4, uchar4, char4, int4, op>  },
                    { FMXImpl_AOAOA<short, uchar, schar, float, op>, FMXImpl_AOAOA<short2, uchar2, char2, float2, op>, FMXImpl_AOAOA<short3, uchar3, char3, float3, op>, FMXImpl_AOAOA<short4, uchar4, char4, float4, op>  },
                    { FMXImpl_AOAOA<short, uchar, schar, double, op>, FMXImpl_AOAOA<short2, uchar2, char2, double2, op>, FMXImpl_AOAOA<short3, uchar3, char3, double3, op>, FMXImpl_AOAOA<short4, uchar4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, uchar, ushort, uchar, op>, FMXImpl_AOAOA<short2, uchar2, ushort2, uchar2, op>, FMXImpl_AOAOA<short3, uchar3, ushort3, uchar3, op>, FMXImpl_AOAOA<short4, uchar4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, uchar, ushort, schar, op>, FMXImpl_AOAOA<short2, uchar2, ushort2, char2, op>, FMXImpl_AOAOA<short3, uchar3, ushort3, char3, op>, FMXImpl_AOAOA<short4, uchar4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<short, uchar, ushort, ushort, op>, FMXImpl_AOAOA<short2, uchar2, ushort2, ushort2, op>, FMXImpl_AOAOA<short3, uchar3, ushort3, ushort3, op>, FMXImpl_AOAOA<short4, uchar4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, uchar, ushort, short, op>, FMXImpl_AOAOA<short2, uchar2, ushort2, short2, op>, FMXImpl_AOAOA<short3, uchar3, ushort3, short3, op>, FMXImpl_AOAOA<short4, uchar4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<short, uchar, ushort, int, op>, FMXImpl_AOAOA<short2, uchar2, ushort2, int2, op>, FMXImpl_AOAOA<short3, uchar3, ushort3, int3, op>, FMXImpl_AOAOA<short4, uchar4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<short, uchar, ushort, float, op>, FMXImpl_AOAOA<short2, uchar2, ushort2, float2, op>, FMXImpl_AOAOA<short3, uchar3, ushort3, float3, op>, FMXImpl_AOAOA<short4, uchar4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<short, uchar, ushort, double, op>, FMXImpl_AOAOA<short2, uchar2, ushort2, double2, op>, FMXImpl_AOAOA<short3, uchar3, ushort3, double3, op>, FMXImpl_AOAOA<short4, uchar4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, uchar, short, uchar, op>, FMXImpl_AOAOA<short2, uchar2, short2, uchar2, op>, FMXImpl_AOAOA<short3, uchar3, short3, uchar3, op>, FMXImpl_AOAOA<short4, uchar4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, uchar, short, schar, op>, FMXImpl_AOAOA<short2, uchar2, short2, char2, op>, FMXImpl_AOAOA<short3, uchar3, short3, char3, op>, FMXImpl_AOAOA<short4, uchar4, short4, char4, op>  },
                    { FMXImpl_AOAOA<short, uchar, short, ushort, op>, FMXImpl_AOAOA<short2, uchar2, short2, ushort2, op>, FMXImpl_AOAOA<short3, uchar3, short3, ushort3, op>, FMXImpl_AOAOA<short4, uchar4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, uchar, short, short, op>, FMXImpl_AOAOA<short2, uchar2, short2, short2, op>, FMXImpl_AOAOA<short3, uchar3, short3, short3, op>, FMXImpl_AOAOA<short4, uchar4, short4, short4, op>  },
                    { FMXImpl_AOAOA<short, uchar, short, int, op>, FMXImpl_AOAOA<short2, uchar2, short2, int2, op>, FMXImpl_AOAOA<short3, uchar3, short3, int3, op>, FMXImpl_AOAOA<short4, uchar4, short4, int4, op>  },
                    { FMXImpl_AOAOA<short, uchar, short, float, op>, FMXImpl_AOAOA<short2, uchar2, short2, float2, op>, FMXImpl_AOAOA<short3, uchar3, short3, float3, op>, FMXImpl_AOAOA<short4, uchar4, short4, float4, op>  },
                    { FMXImpl_AOAOA<short, uchar, short, double, op>, FMXImpl_AOAOA<short2, uchar2, short2, double2, op>, FMXImpl_AOAOA<short3, uchar3, short3, double3, op>, FMXImpl_AOAOA<short4, uchar4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, uchar, int, uchar, op>, FMXImpl_AOAOA<short2, uchar2, int2, uchar2, op>, FMXImpl_AOAOA<short3, uchar3, int3, uchar3, op>, FMXImpl_AOAOA<short4, uchar4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, uchar, int, schar, op>, FMXImpl_AOAOA<short2, uchar2, int2, char2, op>, FMXImpl_AOAOA<short3, uchar3, int3, char3, op>, FMXImpl_AOAOA<short4, uchar4, int4, char4, op>  },
                    { FMXImpl_AOAOA<short, uchar, int, ushort, op>, FMXImpl_AOAOA<short2, uchar2, int2, ushort2, op>, FMXImpl_AOAOA<short3, uchar3, int3, ushort3, op>, FMXImpl_AOAOA<short4, uchar4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, uchar, int, short, op>, FMXImpl_AOAOA<short2, uchar2, int2, short2, op>, FMXImpl_AOAOA<short3, uchar3, int3, short3, op>, FMXImpl_AOAOA<short4, uchar4, int4, short4, op>  },
                    { FMXImpl_AOAOA<short, uchar, int, int, op>, FMXImpl_AOAOA<short2, uchar2, int2, int2, op>, FMXImpl_AOAOA<short3, uchar3, int3, int3, op>, FMXImpl_AOAOA<short4, uchar4, int4, int4, op>  },
                    { FMXImpl_AOAOA<short, uchar, int, float, op>, FMXImpl_AOAOA<short2, uchar2, int2, float2, op>, FMXImpl_AOAOA<short3, uchar3, int3, float3, op>, FMXImpl_AOAOA<short4, uchar4, int4, float4, op>  },
                    { FMXImpl_AOAOA<short, uchar, int, double, op>, FMXImpl_AOAOA<short2, uchar2, int2, double2, op>, FMXImpl_AOAOA<short3, uchar3, int3, double3, op>, FMXImpl_AOAOA<short4, uchar4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, uchar, float, uchar, op>, FMXImpl_AOAOA<short2, uchar2, float2, uchar2, op>, FMXImpl_AOAOA<short3, uchar3, float3, uchar3, op>, FMXImpl_AOAOA<short4, uchar4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, uchar, float, schar, op>, FMXImpl_AOAOA<short2, uchar2, float2, char2, op>, FMXImpl_AOAOA<short3, uchar3, float3, char3, op>, FMXImpl_AOAOA<short4, uchar4, float4, char4, op>  },
                    { FMXImpl_AOAOA<short, uchar, float, ushort, op>, FMXImpl_AOAOA<short2, uchar2, float2, ushort2, op>, FMXImpl_AOAOA<short3, uchar3, float3, ushort3, op>, FMXImpl_AOAOA<short4, uchar4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, uchar, float, short, op>, FMXImpl_AOAOA<short2, uchar2, float2, short2, op>, FMXImpl_AOAOA<short3, uchar3, float3, short3, op>, FMXImpl_AOAOA<short4, uchar4, float4, short4, op>  },
                    { FMXImpl_AOAOA<short, uchar, float, int, op>, FMXImpl_AOAOA<short2, uchar2, float2, int2, op>, FMXImpl_AOAOA<short3, uchar3, float3, int3, op>, FMXImpl_AOAOA<short4, uchar4, float4, int4, op>  },
                    { FMXImpl_AOAOA<short, uchar, float, float, op>, FMXImpl_AOAOA<short2, uchar2, float2, float2, op>, FMXImpl_AOAOA<short3, uchar3, float3, float3, op>, FMXImpl_AOAOA<short4, uchar4, float4, float4, op>  },
                    { FMXImpl_AOAOA<short, uchar, float, double, op>, FMXImpl_AOAOA<short2, uchar2, float2, double2, op>, FMXImpl_AOAOA<short3, uchar3, float3, double3, op>, FMXImpl_AOAOA<short4, uchar4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, uchar, double, uchar, op>, FMXImpl_AOAOA<short2, uchar2, double2, uchar2, op>, FMXImpl_AOAOA<short3, uchar3, double3, uchar3, op>, FMXImpl_AOAOA<short4, uchar4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, uchar, double, schar, op>, FMXImpl_AOAOA<short2, uchar2, double2, char2, op>, FMXImpl_AOAOA<short3, uchar3, double3, char3, op>, FMXImpl_AOAOA<short4, uchar4, double4, char4, op>  },
                    { FMXImpl_AOAOA<short, uchar, double, ushort, op>, FMXImpl_AOAOA<short2, uchar2, double2, ushort2, op>, FMXImpl_AOAOA<short3, uchar3, double3, ushort3, op>, FMXImpl_AOAOA<short4, uchar4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, uchar, double, short, op>, FMXImpl_AOAOA<short2, uchar2, double2, short2, op>, FMXImpl_AOAOA<short3, uchar3, double3, short3, op>, FMXImpl_AOAOA<short4, uchar4, double4, short4, op>  },
                    { FMXImpl_AOAOA<short, uchar, double, int, op>, FMXImpl_AOAOA<short2, uchar2, double2, int2, op>, FMXImpl_AOAOA<short3, uchar3, double3, int3, op>, FMXImpl_AOAOA<short4, uchar4, double4, int4, op>  },
                    { FMXImpl_AOAOA<short, uchar, double, float, op>, FMXImpl_AOAOA<short2, uchar2, double2, float2, op>, FMXImpl_AOAOA<short3, uchar3, double3, float3, op>, FMXImpl_AOAOA<short4, uchar4, double4, float4, op>  },
                    { FMXImpl_AOAOA<short, uchar, double, double, op>, FMXImpl_AOAOA<short2, uchar2, double2, double2, op>, FMXImpl_AOAOA<short3, uchar3, double3, double3, op>, FMXImpl_AOAOA<short4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<short, schar, uchar, uchar, op>, FMXImpl_AOAOA<short2, char2, uchar2, uchar2, op>, FMXImpl_AOAOA<short3, char3, uchar3, uchar3, op>, FMXImpl_AOAOA<short4, char4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, schar, uchar, schar, op>, FMXImpl_AOAOA<short2, char2, uchar2, char2, op>, FMXImpl_AOAOA<short3, char3, uchar3, char3, op>, FMXImpl_AOAOA<short4, char4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<short, schar, uchar, ushort, op>, FMXImpl_AOAOA<short2, char2, uchar2, ushort2, op>, FMXImpl_AOAOA<short3, char3, uchar3, ushort3, op>, FMXImpl_AOAOA<short4, char4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, schar, uchar, short, op>, FMXImpl_AOAOA<short2, char2, uchar2, short2, op>, FMXImpl_AOAOA<short3, char3, uchar3, short3, op>, FMXImpl_AOAOA<short4, char4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<short, schar, uchar, int, op>, FMXImpl_AOAOA<short2, char2, uchar2, int2, op>, FMXImpl_AOAOA<short3, char3, uchar3, int3, op>, FMXImpl_AOAOA<short4, char4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<short, schar, uchar, float, op>, FMXImpl_AOAOA<short2, char2, uchar2, float2, op>, FMXImpl_AOAOA<short3, char3, uchar3, float3, op>, FMXImpl_AOAOA<short4, char4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<short, schar, uchar, double, op>, FMXImpl_AOAOA<short2, char2, uchar2, double2, op>, FMXImpl_AOAOA<short3, char3, uchar3, double3, op>, FMXImpl_AOAOA<short4, char4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, schar, schar, uchar, op>, FMXImpl_AOAOA<short2, char2, char2, uchar2, op>, FMXImpl_AOAOA<short3, char3, char3, uchar3, op>, FMXImpl_AOAOA<short4, char4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, schar, schar, schar, op>, FMXImpl_AOAOA<short2, char2, char2, char2, op>, FMXImpl_AOAOA<short3, char3, char3, char3, op>, FMXImpl_AOAOA<short4, char4, char4, char4, op>  },
                    { FMXImpl_AOAOA<short, schar, schar, ushort, op>, FMXImpl_AOAOA<short2, char2, char2, ushort2, op>, FMXImpl_AOAOA<short3, char3, char3, ushort3, op>, FMXImpl_AOAOA<short4, char4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, schar, schar, short, op>, FMXImpl_AOAOA<short2, char2, char2, short2, op>, FMXImpl_AOAOA<short3, char3, char3, short3, op>, FMXImpl_AOAOA<short4, char4, char4, short4, op>  },
                    { FMXImpl_AOAOA<short, schar, schar, int, op>, FMXImpl_AOAOA<short2, char2, char2, int2, op>, FMXImpl_AOAOA<short3, char3, char3, int3, op>, FMXImpl_AOAOA<short4, char4, char4, int4, op>  },
                    { FMXImpl_AOAOA<short, schar, schar, float, op>, FMXImpl_AOAOA<short2, char2, char2, float2, op>, FMXImpl_AOAOA<short3, char3, char3, float3, op>, FMXImpl_AOAOA<short4, char4, char4, float4, op>  },
                    { FMXImpl_AOAOA<short, schar, schar, double, op>, FMXImpl_AOAOA<short2, char2, char2, double2, op>, FMXImpl_AOAOA<short3, char3, char3, double3, op>, FMXImpl_AOAOA<short4, char4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, schar, ushort, uchar, op>, FMXImpl_AOAOA<short2, char2, ushort2, uchar2, op>, FMXImpl_AOAOA<short3, char3, ushort3, uchar3, op>, FMXImpl_AOAOA<short4, char4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, schar, ushort, schar, op>, FMXImpl_AOAOA<short2, char2, ushort2, char2, op>, FMXImpl_AOAOA<short3, char3, ushort3, char3, op>, FMXImpl_AOAOA<short4, char4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<short, schar, ushort, ushort, op>, FMXImpl_AOAOA<short2, char2, ushort2, ushort2, op>, FMXImpl_AOAOA<short3, char3, ushort3, ushort3, op>, FMXImpl_AOAOA<short4, char4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, schar, ushort, short, op>, FMXImpl_AOAOA<short2, char2, ushort2, short2, op>, FMXImpl_AOAOA<short3, char3, ushort3, short3, op>, FMXImpl_AOAOA<short4, char4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<short, schar, ushort, int, op>, FMXImpl_AOAOA<short2, char2, ushort2, int2, op>, FMXImpl_AOAOA<short3, char3, ushort3, int3, op>, FMXImpl_AOAOA<short4, char4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<short, schar, ushort, float, op>, FMXImpl_AOAOA<short2, char2, ushort2, float2, op>, FMXImpl_AOAOA<short3, char3, ushort3, float3, op>, FMXImpl_AOAOA<short4, char4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<short, schar, ushort, double, op>, FMXImpl_AOAOA<short2, char2, ushort2, double2, op>, FMXImpl_AOAOA<short3, char3, ushort3, double3, op>, FMXImpl_AOAOA<short4, char4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, schar, short, uchar, op>, FMXImpl_AOAOA<short2, char2, short2, uchar2, op>, FMXImpl_AOAOA<short3, char3, short3, uchar3, op>, FMXImpl_AOAOA<short4, char4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, schar, short, schar, op>, FMXImpl_AOAOA<short2, char2, short2, char2, op>, FMXImpl_AOAOA<short3, char3, short3, char3, op>, FMXImpl_AOAOA<short4, char4, short4, char4, op>  },
                    { FMXImpl_AOAOA<short, schar, short, ushort, op>, FMXImpl_AOAOA<short2, char2, short2, ushort2, op>, FMXImpl_AOAOA<short3, char3, short3, ushort3, op>, FMXImpl_AOAOA<short4, char4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, schar, short, short, op>, FMXImpl_AOAOA<short2, char2, short2, short2, op>, FMXImpl_AOAOA<short3, char3, short3, short3, op>, FMXImpl_AOAOA<short4, char4, short4, short4, op>  },
                    { FMXImpl_AOAOA<short, schar, short, int, op>, FMXImpl_AOAOA<short2, char2, short2, int2, op>, FMXImpl_AOAOA<short3, char3, short3, int3, op>, FMXImpl_AOAOA<short4, char4, short4, int4, op>  },
                    { FMXImpl_AOAOA<short, schar, short, float, op>, FMXImpl_AOAOA<short2, char2, short2, float2, op>, FMXImpl_AOAOA<short3, char3, short3, float3, op>, FMXImpl_AOAOA<short4, char4, short4, float4, op>  },
                    { FMXImpl_AOAOA<short, schar, short, double, op>, FMXImpl_AOAOA<short2, char2, short2, double2, op>, FMXImpl_AOAOA<short3, char3, short3, double3, op>, FMXImpl_AOAOA<short4, char4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, schar, int, uchar, op>, FMXImpl_AOAOA<short2, char2, int2, uchar2, op>, FMXImpl_AOAOA<short3, char3, int3, uchar3, op>, FMXImpl_AOAOA<short4, char4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, schar, int, schar, op>, FMXImpl_AOAOA<short2, char2, int2, char2, op>, FMXImpl_AOAOA<short3, char3, int3, char3, op>, FMXImpl_AOAOA<short4, char4, int4, char4, op>  },
                    { FMXImpl_AOAOA<short, schar, int, ushort, op>, FMXImpl_AOAOA<short2, char2, int2, ushort2, op>, FMXImpl_AOAOA<short3, char3, int3, ushort3, op>, FMXImpl_AOAOA<short4, char4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, schar, int, short, op>, FMXImpl_AOAOA<short2, char2, int2, short2, op>, FMXImpl_AOAOA<short3, char3, int3, short3, op>, FMXImpl_AOAOA<short4, char4, int4, short4, op>  },
                    { FMXImpl_AOAOA<short, schar, int, int, op>, FMXImpl_AOAOA<short2, char2, int2, int2, op>, FMXImpl_AOAOA<short3, char3, int3, int3, op>, FMXImpl_AOAOA<short4, char4, int4, int4, op>  },
                    { FMXImpl_AOAOA<short, schar, int, float, op>, FMXImpl_AOAOA<short2, char2, int2, float2, op>, FMXImpl_AOAOA<short3, char3, int3, float3, op>, FMXImpl_AOAOA<short4, char4, int4, float4, op>  },
                    { FMXImpl_AOAOA<short, schar, int, double, op>, FMXImpl_AOAOA<short2, char2, int2, double2, op>, FMXImpl_AOAOA<short3, char3, int3, double3, op>, FMXImpl_AOAOA<short4, char4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, schar, float, uchar, op>, FMXImpl_AOAOA<short2, char2, float2, uchar2, op>, FMXImpl_AOAOA<short3, char3, float3, uchar3, op>, FMXImpl_AOAOA<short4, char4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, schar, float, schar, op>, FMXImpl_AOAOA<short2, char2, float2, char2, op>, FMXImpl_AOAOA<short3, char3, float3, char3, op>, FMXImpl_AOAOA<short4, char4, float4, char4, op>  },
                    { FMXImpl_AOAOA<short, schar, float, ushort, op>, FMXImpl_AOAOA<short2, char2, float2, ushort2, op>, FMXImpl_AOAOA<short3, char3, float3, ushort3, op>, FMXImpl_AOAOA<short4, char4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, schar, float, short, op>, FMXImpl_AOAOA<short2, char2, float2, short2, op>, FMXImpl_AOAOA<short3, char3, float3, short3, op>, FMXImpl_AOAOA<short4, char4, float4, short4, op>  },
                    { FMXImpl_AOAOA<short, schar, float, int, op>, FMXImpl_AOAOA<short2, char2, float2, int2, op>, FMXImpl_AOAOA<short3, char3, float3, int3, op>, FMXImpl_AOAOA<short4, char4, float4, int4, op>  },
                    { FMXImpl_AOAOA<short, schar, float, float, op>, FMXImpl_AOAOA<short2, char2, float2, float2, op>, FMXImpl_AOAOA<short3, char3, float3, float3, op>, FMXImpl_AOAOA<short4, char4, float4, float4, op>  },
                    { FMXImpl_AOAOA<short, schar, float, double, op>, FMXImpl_AOAOA<short2, char2, float2, double2, op>, FMXImpl_AOAOA<short3, char3, float3, double3, op>, FMXImpl_AOAOA<short4, char4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, schar, double, uchar, op>, FMXImpl_AOAOA<short2, char2, double2, uchar2, op>, FMXImpl_AOAOA<short3, char3, double3, uchar3, op>, FMXImpl_AOAOA<short4, char4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, schar, double, schar, op>, FMXImpl_AOAOA<short2, char2, double2, char2, op>, FMXImpl_AOAOA<short3, char3, double3, char3, op>, FMXImpl_AOAOA<short4, char4, double4, char4, op>  },
                    { FMXImpl_AOAOA<short, schar, double, ushort, op>, FMXImpl_AOAOA<short2, char2, double2, ushort2, op>, FMXImpl_AOAOA<short3, char3, double3, ushort3, op>, FMXImpl_AOAOA<short4, char4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, schar, double, short, op>, FMXImpl_AOAOA<short2, char2, double2, short2, op>, FMXImpl_AOAOA<short3, char3, double3, short3, op>, FMXImpl_AOAOA<short4, char4, double4, short4, op>  },
                    { FMXImpl_AOAOA<short, schar, double, int, op>, FMXImpl_AOAOA<short2, char2, double2, int2, op>, FMXImpl_AOAOA<short3, char3, double3, int3, op>, FMXImpl_AOAOA<short4, char4, double4, int4, op>  },
                    { FMXImpl_AOAOA<short, schar, double, float, op>, FMXImpl_AOAOA<short2, char2, double2, float2, op>, FMXImpl_AOAOA<short3, char3, double3, float3, op>, FMXImpl_AOAOA<short4, char4, double4, float4, op>  },
                    { FMXImpl_AOAOA<short, schar, double, double, op>, FMXImpl_AOAOA<short2, char2, double2, double2, op>, FMXImpl_AOAOA<short3, char3, double3, double3, op>, FMXImpl_AOAOA<short4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<short, ushort, uchar, uchar, op>, FMXImpl_AOAOA<short2, ushort2, uchar2, uchar2, op>, FMXImpl_AOAOA<short3, ushort3, uchar3, uchar3, op>, FMXImpl_AOAOA<short4, ushort4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, ushort, uchar, schar, op>, FMXImpl_AOAOA<short2, ushort2, uchar2, char2, op>, FMXImpl_AOAOA<short3, ushort3, uchar3, char3, op>, FMXImpl_AOAOA<short4, ushort4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<short, ushort, uchar, ushort, op>, FMXImpl_AOAOA<short2, ushort2, uchar2, ushort2, op>, FMXImpl_AOAOA<short3, ushort3, uchar3, ushort3, op>, FMXImpl_AOAOA<short4, ushort4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, ushort, uchar, short, op>, FMXImpl_AOAOA<short2, ushort2, uchar2, short2, op>, FMXImpl_AOAOA<short3, ushort3, uchar3, short3, op>, FMXImpl_AOAOA<short4, ushort4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<short, ushort, uchar, int, op>, FMXImpl_AOAOA<short2, ushort2, uchar2, int2, op>, FMXImpl_AOAOA<short3, ushort3, uchar3, int3, op>, FMXImpl_AOAOA<short4, ushort4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<short, ushort, uchar, float, op>, FMXImpl_AOAOA<short2, ushort2, uchar2, float2, op>, FMXImpl_AOAOA<short3, ushort3, uchar3, float3, op>, FMXImpl_AOAOA<short4, ushort4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<short, ushort, uchar, double, op>, FMXImpl_AOAOA<short2, ushort2, uchar2, double2, op>, FMXImpl_AOAOA<short3, ushort3, uchar3, double3, op>, FMXImpl_AOAOA<short4, ushort4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, ushort, schar, uchar, op>, FMXImpl_AOAOA<short2, ushort2, char2, uchar2, op>, FMXImpl_AOAOA<short3, ushort3, char3, uchar3, op>, FMXImpl_AOAOA<short4, ushort4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, ushort, schar, schar, op>, FMXImpl_AOAOA<short2, ushort2, char2, char2, op>, FMXImpl_AOAOA<short3, ushort3, char3, char3, op>, FMXImpl_AOAOA<short4, ushort4, char4, char4, op>  },
                    { FMXImpl_AOAOA<short, ushort, schar, ushort, op>, FMXImpl_AOAOA<short2, ushort2, char2, ushort2, op>, FMXImpl_AOAOA<short3, ushort3, char3, ushort3, op>, FMXImpl_AOAOA<short4, ushort4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, ushort, schar, short, op>, FMXImpl_AOAOA<short2, ushort2, char2, short2, op>, FMXImpl_AOAOA<short3, ushort3, char3, short3, op>, FMXImpl_AOAOA<short4, ushort4, char4, short4, op>  },
                    { FMXImpl_AOAOA<short, ushort, schar, int, op>, FMXImpl_AOAOA<short2, ushort2, char2, int2, op>, FMXImpl_AOAOA<short3, ushort3, char3, int3, op>, FMXImpl_AOAOA<short4, ushort4, char4, int4, op>  },
                    { FMXImpl_AOAOA<short, ushort, schar, float, op>, FMXImpl_AOAOA<short2, ushort2, char2, float2, op>, FMXImpl_AOAOA<short3, ushort3, char3, float3, op>, FMXImpl_AOAOA<short4, ushort4, char4, float4, op>  },
                    { FMXImpl_AOAOA<short, ushort, schar, double, op>, FMXImpl_AOAOA<short2, ushort2, char2, double2, op>, FMXImpl_AOAOA<short3, ushort3, char3, double3, op>, FMXImpl_AOAOA<short4, ushort4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, ushort, ushort, uchar, op>, FMXImpl_AOAOA<short2, ushort2, ushort2, uchar2, op>, FMXImpl_AOAOA<short3, ushort3, ushort3, uchar3, op>, FMXImpl_AOAOA<short4, ushort4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, ushort, ushort, schar, op>, FMXImpl_AOAOA<short2, ushort2, ushort2, char2, op>, FMXImpl_AOAOA<short3, ushort3, ushort3, char3, op>, FMXImpl_AOAOA<short4, ushort4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<short, ushort, ushort, ushort, op>, FMXImpl_AOAOA<short2, ushort2, ushort2, ushort2, op>, FMXImpl_AOAOA<short3, ushort3, ushort3, ushort3, op>, FMXImpl_AOAOA<short4, ushort4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, ushort, ushort, short, op>, FMXImpl_AOAOA<short2, ushort2, ushort2, short2, op>, FMXImpl_AOAOA<short3, ushort3, ushort3, short3, op>, FMXImpl_AOAOA<short4, ushort4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<short, ushort, ushort, int, op>, FMXImpl_AOAOA<short2, ushort2, ushort2, int2, op>, FMXImpl_AOAOA<short3, ushort3, ushort3, int3, op>, FMXImpl_AOAOA<short4, ushort4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<short, ushort, ushort, float, op>, FMXImpl_AOAOA<short2, ushort2, ushort2, float2, op>, FMXImpl_AOAOA<short3, ushort3, ushort3, float3, op>, FMXImpl_AOAOA<short4, ushort4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<short, ushort, ushort, double, op>, FMXImpl_AOAOA<short2, ushort2, ushort2, double2, op>, FMXImpl_AOAOA<short3, ushort3, ushort3, double3, op>, FMXImpl_AOAOA<short4, ushort4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, ushort, short, uchar, op>, FMXImpl_AOAOA<short2, ushort2, short2, uchar2, op>, FMXImpl_AOAOA<short3, ushort3, short3, uchar3, op>, FMXImpl_AOAOA<short4, ushort4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, ushort, short, schar, op>, FMXImpl_AOAOA<short2, ushort2, short2, char2, op>, FMXImpl_AOAOA<short3, ushort3, short3, char3, op>, FMXImpl_AOAOA<short4, ushort4, short4, char4, op>  },
                    { FMXImpl_AOAOA<short, ushort, short, ushort, op>, FMXImpl_AOAOA<short2, ushort2, short2, ushort2, op>, FMXImpl_AOAOA<short3, ushort3, short3, ushort3, op>, FMXImpl_AOAOA<short4, ushort4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, ushort, short, short, op>, FMXImpl_AOAOA<short2, ushort2, short2, short2, op>, FMXImpl_AOAOA<short3, ushort3, short3, short3, op>, FMXImpl_AOAOA<short4, ushort4, short4, short4, op>  },
                    { FMXImpl_AOAOA<short, ushort, short, int, op>, FMXImpl_AOAOA<short2, ushort2, short2, int2, op>, FMXImpl_AOAOA<short3, ushort3, short3, int3, op>, FMXImpl_AOAOA<short4, ushort4, short4, int4, op>  },
                    { FMXImpl_AOAOA<short, ushort, short, float, op>, FMXImpl_AOAOA<short2, ushort2, short2, float2, op>, FMXImpl_AOAOA<short3, ushort3, short3, float3, op>, FMXImpl_AOAOA<short4, ushort4, short4, float4, op>  },
                    { FMXImpl_AOAOA<short, ushort, short, double, op>, FMXImpl_AOAOA<short2, ushort2, short2, double2, op>, FMXImpl_AOAOA<short3, ushort3, short3, double3, op>, FMXImpl_AOAOA<short4, ushort4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, ushort, int, uchar, op>, FMXImpl_AOAOA<short2, ushort2, int2, uchar2, op>, FMXImpl_AOAOA<short3, ushort3, int3, uchar3, op>, FMXImpl_AOAOA<short4, ushort4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, ushort, int, schar, op>, FMXImpl_AOAOA<short2, ushort2, int2, char2, op>, FMXImpl_AOAOA<short3, ushort3, int3, char3, op>, FMXImpl_AOAOA<short4, ushort4, int4, char4, op>  },
                    { FMXImpl_AOAOA<short, ushort, int, ushort, op>, FMXImpl_AOAOA<short2, ushort2, int2, ushort2, op>, FMXImpl_AOAOA<short3, ushort3, int3, ushort3, op>, FMXImpl_AOAOA<short4, ushort4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, ushort, int, short, op>, FMXImpl_AOAOA<short2, ushort2, int2, short2, op>, FMXImpl_AOAOA<short3, ushort3, int3, short3, op>, FMXImpl_AOAOA<short4, ushort4, int4, short4, op>  },
                    { FMXImpl_AOAOA<short, ushort, int, int, op>, FMXImpl_AOAOA<short2, ushort2, int2, int2, op>, FMXImpl_AOAOA<short3, ushort3, int3, int3, op>, FMXImpl_AOAOA<short4, ushort4, int4, int4, op>  },
                    { FMXImpl_AOAOA<short, ushort, int, float, op>, FMXImpl_AOAOA<short2, ushort2, int2, float2, op>, FMXImpl_AOAOA<short3, ushort3, int3, float3, op>, FMXImpl_AOAOA<short4, ushort4, int4, float4, op>  },
                    { FMXImpl_AOAOA<short, ushort, int, double, op>, FMXImpl_AOAOA<short2, ushort2, int2, double2, op>, FMXImpl_AOAOA<short3, ushort3, int3, double3, op>, FMXImpl_AOAOA<short4, ushort4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, ushort, float, uchar, op>, FMXImpl_AOAOA<short2, ushort2, float2, uchar2, op>, FMXImpl_AOAOA<short3, ushort3, float3, uchar3, op>, FMXImpl_AOAOA<short4, ushort4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, ushort, float, schar, op>, FMXImpl_AOAOA<short2, ushort2, float2, char2, op>, FMXImpl_AOAOA<short3, ushort3, float3, char3, op>, FMXImpl_AOAOA<short4, ushort4, float4, char4, op>  },
                    { FMXImpl_AOAOA<short, ushort, float, ushort, op>, FMXImpl_AOAOA<short2, ushort2, float2, ushort2, op>, FMXImpl_AOAOA<short3, ushort3, float3, ushort3, op>, FMXImpl_AOAOA<short4, ushort4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, ushort, float, short, op>, FMXImpl_AOAOA<short2, ushort2, float2, short2, op>, FMXImpl_AOAOA<short3, ushort3, float3, short3, op>, FMXImpl_AOAOA<short4, ushort4, float4, short4, op>  },
                    { FMXImpl_AOAOA<short, ushort, float, int, op>, FMXImpl_AOAOA<short2, ushort2, float2, int2, op>, FMXImpl_AOAOA<short3, ushort3, float3, int3, op>, FMXImpl_AOAOA<short4, ushort4, float4, int4, op>  },
                    { FMXImpl_AOAOA<short, ushort, float, float, op>, FMXImpl_AOAOA<short2, ushort2, float2, float2, op>, FMXImpl_AOAOA<short3, ushort3, float3, float3, op>, FMXImpl_AOAOA<short4, ushort4, float4, float4, op>  },
                    { FMXImpl_AOAOA<short, ushort, float, double, op>, FMXImpl_AOAOA<short2, ushort2, float2, double2, op>, FMXImpl_AOAOA<short3, ushort3, float3, double3, op>, FMXImpl_AOAOA<short4, ushort4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, ushort, double, uchar, op>, FMXImpl_AOAOA<short2, ushort2, double2, uchar2, op>, FMXImpl_AOAOA<short3, ushort3, double3, uchar3, op>, FMXImpl_AOAOA<short4, ushort4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, ushort, double, schar, op>, FMXImpl_AOAOA<short2, ushort2, double2, char2, op>, FMXImpl_AOAOA<short3, ushort3, double3, char3, op>, FMXImpl_AOAOA<short4, ushort4, double4, char4, op>  },
                    { FMXImpl_AOAOA<short, ushort, double, ushort, op>, FMXImpl_AOAOA<short2, ushort2, double2, ushort2, op>, FMXImpl_AOAOA<short3, ushort3, double3, ushort3, op>, FMXImpl_AOAOA<short4, ushort4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, ushort, double, short, op>, FMXImpl_AOAOA<short2, ushort2, double2, short2, op>, FMXImpl_AOAOA<short3, ushort3, double3, short3, op>, FMXImpl_AOAOA<short4, ushort4, double4, short4, op>  },
                    { FMXImpl_AOAOA<short, ushort, double, int, op>, FMXImpl_AOAOA<short2, ushort2, double2, int2, op>, FMXImpl_AOAOA<short3, ushort3, double3, int3, op>, FMXImpl_AOAOA<short4, ushort4, double4, int4, op>  },
                    { FMXImpl_AOAOA<short, ushort, double, float, op>, FMXImpl_AOAOA<short2, ushort2, double2, float2, op>, FMXImpl_AOAOA<short3, ushort3, double3, float3, op>, FMXImpl_AOAOA<short4, ushort4, double4, float4, op>  },
                    { FMXImpl_AOAOA<short, ushort, double, double, op>, FMXImpl_AOAOA<short2, ushort2, double2, double2, op>, FMXImpl_AOAOA<short3, ushort3, double3, double3, op>, FMXImpl_AOAOA<short4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<short, short, uchar, uchar, op>, FMXImpl_AOAOA<short2, short2, uchar2, uchar2, op>, FMXImpl_AOAOA<short3, short3, uchar3, uchar3, op>, FMXImpl_AOAOA<short4, short4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, short, uchar, schar, op>, FMXImpl_AOAOA<short2, short2, uchar2, char2, op>, FMXImpl_AOAOA<short3, short3, uchar3, char3, op>, FMXImpl_AOAOA<short4, short4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<short, short, uchar, ushort, op>, FMXImpl_AOAOA<short2, short2, uchar2, ushort2, op>, FMXImpl_AOAOA<short3, short3, uchar3, ushort3, op>, FMXImpl_AOAOA<short4, short4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, short, uchar, short, op>, FMXImpl_AOAOA<short2, short2, uchar2, short2, op>, FMXImpl_AOAOA<short3, short3, uchar3, short3, op>, FMXImpl_AOAOA<short4, short4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<short, short, uchar, int, op>, FMXImpl_AOAOA<short2, short2, uchar2, int2, op>, FMXImpl_AOAOA<short3, short3, uchar3, int3, op>, FMXImpl_AOAOA<short4, short4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<short, short, uchar, float, op>, FMXImpl_AOAOA<short2, short2, uchar2, float2, op>, FMXImpl_AOAOA<short3, short3, uchar3, float3, op>, FMXImpl_AOAOA<short4, short4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<short, short, uchar, double, op>, FMXImpl_AOAOA<short2, short2, uchar2, double2, op>, FMXImpl_AOAOA<short3, short3, uchar3, double3, op>, FMXImpl_AOAOA<short4, short4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, short, schar, uchar, op>, FMXImpl_AOAOA<short2, short2, char2, uchar2, op>, FMXImpl_AOAOA<short3, short3, char3, uchar3, op>, FMXImpl_AOAOA<short4, short4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, short, schar, schar, op>, FMXImpl_AOAOA<short2, short2, char2, char2, op>, FMXImpl_AOAOA<short3, short3, char3, char3, op>, FMXImpl_AOAOA<short4, short4, char4, char4, op>  },
                    { FMXImpl_AOAOA<short, short, schar, ushort, op>, FMXImpl_AOAOA<short2, short2, char2, ushort2, op>, FMXImpl_AOAOA<short3, short3, char3, ushort3, op>, FMXImpl_AOAOA<short4, short4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, short, schar, short, op>, FMXImpl_AOAOA<short2, short2, char2, short2, op>, FMXImpl_AOAOA<short3, short3, char3, short3, op>, FMXImpl_AOAOA<short4, short4, char4, short4, op>  },
                    { FMXImpl_AOAOA<short, short, schar, int, op>, FMXImpl_AOAOA<short2, short2, char2, int2, op>, FMXImpl_AOAOA<short3, short3, char3, int3, op>, FMXImpl_AOAOA<short4, short4, char4, int4, op>  },
                    { FMXImpl_AOAOA<short, short, schar, float, op>, FMXImpl_AOAOA<short2, short2, char2, float2, op>, FMXImpl_AOAOA<short3, short3, char3, float3, op>, FMXImpl_AOAOA<short4, short4, char4, float4, op>  },
                    { FMXImpl_AOAOA<short, short, schar, double, op>, FMXImpl_AOAOA<short2, short2, char2, double2, op>, FMXImpl_AOAOA<short3, short3, char3, double3, op>, FMXImpl_AOAOA<short4, short4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, short, ushort, uchar, op>, FMXImpl_AOAOA<short2, short2, ushort2, uchar2, op>, FMXImpl_AOAOA<short3, short3, ushort3, uchar3, op>, FMXImpl_AOAOA<short4, short4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, short, ushort, schar, op>, FMXImpl_AOAOA<short2, short2, ushort2, char2, op>, FMXImpl_AOAOA<short3, short3, ushort3, char3, op>, FMXImpl_AOAOA<short4, short4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<short, short, ushort, ushort, op>, FMXImpl_AOAOA<short2, short2, ushort2, ushort2, op>, FMXImpl_AOAOA<short3, short3, ushort3, ushort3, op>, FMXImpl_AOAOA<short4, short4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, short, ushort, short, op>, FMXImpl_AOAOA<short2, short2, ushort2, short2, op>, FMXImpl_AOAOA<short3, short3, ushort3, short3, op>, FMXImpl_AOAOA<short4, short4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<short, short, ushort, int, op>, FMXImpl_AOAOA<short2, short2, ushort2, int2, op>, FMXImpl_AOAOA<short3, short3, ushort3, int3, op>, FMXImpl_AOAOA<short4, short4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<short, short, ushort, float, op>, FMXImpl_AOAOA<short2, short2, ushort2, float2, op>, FMXImpl_AOAOA<short3, short3, ushort3, float3, op>, FMXImpl_AOAOA<short4, short4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<short, short, ushort, double, op>, FMXImpl_AOAOA<short2, short2, ushort2, double2, op>, FMXImpl_AOAOA<short3, short3, ushort3, double3, op>, FMXImpl_AOAOA<short4, short4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, short, short, uchar, op>, FMXImpl_AOAOA<short2, short2, short2, uchar2, op>, FMXImpl_AOAOA<short3, short3, short3, uchar3, op>, FMXImpl_AOAOA<short4, short4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, short, short, schar, op>, FMXImpl_AOAOA<short2, short2, short2, char2, op>, FMXImpl_AOAOA<short3, short3, short3, char3, op>, FMXImpl_AOAOA<short4, short4, short4, char4, op>  },
                    { FMXImpl_AOAOA<short, short, short, ushort, op>, FMXImpl_AOAOA<short2, short2, short2, ushort2, op>, FMXImpl_AOAOA<short3, short3, short3, ushort3, op>, FMXImpl_AOAOA<short4, short4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, short, short, short, op>, FMXImpl_AOAOA<short2, short2, short2, short2, op>, FMXImpl_AOAOA<short3, short3, short3, short3, op>, FMXImpl_AOAOA<short4, short4, short4, short4, op>  },
                    { FMXImpl_AOAOA<short, short, short, int, op>, FMXImpl_AOAOA<short2, short2, short2, int2, op>, FMXImpl_AOAOA<short3, short3, short3, int3, op>, FMXImpl_AOAOA<short4, short4, short4, int4, op>  },
                    { FMXImpl_AOAOA<short, short, short, float, op>, FMXImpl_AOAOA<short2, short2, short2, float2, op>, FMXImpl_AOAOA<short3, short3, short3, float3, op>, FMXImpl_AOAOA<short4, short4, short4, float4, op>  },
                    { FMXImpl_AOAOA<short, short, short, double, op>, FMXImpl_AOAOA<short2, short2, short2, double2, op>, FMXImpl_AOAOA<short3, short3, short3, double3, op>, FMXImpl_AOAOA<short4, short4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, short, int, uchar, op>, FMXImpl_AOAOA<short2, short2, int2, uchar2, op>, FMXImpl_AOAOA<short3, short3, int3, uchar3, op>, FMXImpl_AOAOA<short4, short4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, short, int, schar, op>, FMXImpl_AOAOA<short2, short2, int2, char2, op>, FMXImpl_AOAOA<short3, short3, int3, char3, op>, FMXImpl_AOAOA<short4, short4, int4, char4, op>  },
                    { FMXImpl_AOAOA<short, short, int, ushort, op>, FMXImpl_AOAOA<short2, short2, int2, ushort2, op>, FMXImpl_AOAOA<short3, short3, int3, ushort3, op>, FMXImpl_AOAOA<short4, short4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, short, int, short, op>, FMXImpl_AOAOA<short2, short2, int2, short2, op>, FMXImpl_AOAOA<short3, short3, int3, short3, op>, FMXImpl_AOAOA<short4, short4, int4, short4, op>  },
                    { FMXImpl_AOAOA<short, short, int, int, op>, FMXImpl_AOAOA<short2, short2, int2, int2, op>, FMXImpl_AOAOA<short3, short3, int3, int3, op>, FMXImpl_AOAOA<short4, short4, int4, int4, op>  },
                    { FMXImpl_AOAOA<short, short, int, float, op>, FMXImpl_AOAOA<short2, short2, int2, float2, op>, FMXImpl_AOAOA<short3, short3, int3, float3, op>, FMXImpl_AOAOA<short4, short4, int4, float4, op>  },
                    { FMXImpl_AOAOA<short, short, int, double, op>, FMXImpl_AOAOA<short2, short2, int2, double2, op>, FMXImpl_AOAOA<short3, short3, int3, double3, op>, FMXImpl_AOAOA<short4, short4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, short, float, uchar, op>, FMXImpl_AOAOA<short2, short2, float2, uchar2, op>, FMXImpl_AOAOA<short3, short3, float3, uchar3, op>, FMXImpl_AOAOA<short4, short4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, short, float, schar, op>, FMXImpl_AOAOA<short2, short2, float2, char2, op>, FMXImpl_AOAOA<short3, short3, float3, char3, op>, FMXImpl_AOAOA<short4, short4, float4, char4, op>  },
                    { FMXImpl_AOAOA<short, short, float, ushort, op>, FMXImpl_AOAOA<short2, short2, float2, ushort2, op>, FMXImpl_AOAOA<short3, short3, float3, ushort3, op>, FMXImpl_AOAOA<short4, short4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, short, float, short, op>, FMXImpl_AOAOA<short2, short2, float2, short2, op>, FMXImpl_AOAOA<short3, short3, float3, short3, op>, FMXImpl_AOAOA<short4, short4, float4, short4, op>  },
                    { FMXImpl_AOAOA<short, short, float, int, op>, FMXImpl_AOAOA<short2, short2, float2, int2, op>, FMXImpl_AOAOA<short3, short3, float3, int3, op>, FMXImpl_AOAOA<short4, short4, float4, int4, op>  },
                    { FMXImpl_AOAOA<short, short, float, float, op>, FMXImpl_AOAOA<short2, short2, float2, float2, op>, FMXImpl_AOAOA<short3, short3, float3, float3, op>, FMXImpl_AOAOA<short4, short4, float4, float4, op>  },
                    { FMXImpl_AOAOA<short, short, float, double, op>, FMXImpl_AOAOA<short2, short2, float2, double2, op>, FMXImpl_AOAOA<short3, short3, float3, double3, op>, FMXImpl_AOAOA<short4, short4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, short, double, uchar, op>, FMXImpl_AOAOA<short2, short2, double2, uchar2, op>, FMXImpl_AOAOA<short3, short3, double3, uchar3, op>, FMXImpl_AOAOA<short4, short4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, short, double, schar, op>, FMXImpl_AOAOA<short2, short2, double2, char2, op>, FMXImpl_AOAOA<short3, short3, double3, char3, op>, FMXImpl_AOAOA<short4, short4, double4, char4, op>  },
                    { FMXImpl_AOAOA<short, short, double, ushort, op>, FMXImpl_AOAOA<short2, short2, double2, ushort2, op>, FMXImpl_AOAOA<short3, short3, double3, ushort3, op>, FMXImpl_AOAOA<short4, short4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, short, double, short, op>, FMXImpl_AOAOA<short2, short2, double2, short2, op>, FMXImpl_AOAOA<short3, short3, double3, short3, op>, FMXImpl_AOAOA<short4, short4, double4, short4, op>  },
                    { FMXImpl_AOAOA<short, short, double, int, op>, FMXImpl_AOAOA<short2, short2, double2, int2, op>, FMXImpl_AOAOA<short3, short3, double3, int3, op>, FMXImpl_AOAOA<short4, short4, double4, int4, op>  },
                    { FMXImpl_AOAOA<short, short, double, float, op>, FMXImpl_AOAOA<short2, short2, double2, float2, op>, FMXImpl_AOAOA<short3, short3, double3, float3, op>, FMXImpl_AOAOA<short4, short4, double4, float4, op>  },
                    { FMXImpl_AOAOA<short, short, double, double, op>, FMXImpl_AOAOA<short2, short2, double2, double2, op>, FMXImpl_AOAOA<short3, short3, double3, double3, op>, FMXImpl_AOAOA<short4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<short, int, uchar, uchar, op>, FMXImpl_AOAOA<short2, int2, uchar2, uchar2, op>, FMXImpl_AOAOA<short3, int3, uchar3, uchar3, op>, FMXImpl_AOAOA<short4, int4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, int, uchar, schar, op>, FMXImpl_AOAOA<short2, int2, uchar2, char2, op>, FMXImpl_AOAOA<short3, int3, uchar3, char3, op>, FMXImpl_AOAOA<short4, int4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<short, int, uchar, ushort, op>, FMXImpl_AOAOA<short2, int2, uchar2, ushort2, op>, FMXImpl_AOAOA<short3, int3, uchar3, ushort3, op>, FMXImpl_AOAOA<short4, int4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, int, uchar, short, op>, FMXImpl_AOAOA<short2, int2, uchar2, short2, op>, FMXImpl_AOAOA<short3, int3, uchar3, short3, op>, FMXImpl_AOAOA<short4, int4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<short, int, uchar, int, op>, FMXImpl_AOAOA<short2, int2, uchar2, int2, op>, FMXImpl_AOAOA<short3, int3, uchar3, int3, op>, FMXImpl_AOAOA<short4, int4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<short, int, uchar, float, op>, FMXImpl_AOAOA<short2, int2, uchar2, float2, op>, FMXImpl_AOAOA<short3, int3, uchar3, float3, op>, FMXImpl_AOAOA<short4, int4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<short, int, uchar, double, op>, FMXImpl_AOAOA<short2, int2, uchar2, double2, op>, FMXImpl_AOAOA<short3, int3, uchar3, double3, op>, FMXImpl_AOAOA<short4, int4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, int, schar, uchar, op>, FMXImpl_AOAOA<short2, int2, char2, uchar2, op>, FMXImpl_AOAOA<short3, int3, char3, uchar3, op>, FMXImpl_AOAOA<short4, int4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, int, schar, schar, op>, FMXImpl_AOAOA<short2, int2, char2, char2, op>, FMXImpl_AOAOA<short3, int3, char3, char3, op>, FMXImpl_AOAOA<short4, int4, char4, char4, op>  },
                    { FMXImpl_AOAOA<short, int, schar, ushort, op>, FMXImpl_AOAOA<short2, int2, char2, ushort2, op>, FMXImpl_AOAOA<short3, int3, char3, ushort3, op>, FMXImpl_AOAOA<short4, int4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, int, schar, short, op>, FMXImpl_AOAOA<short2, int2, char2, short2, op>, FMXImpl_AOAOA<short3, int3, char3, short3, op>, FMXImpl_AOAOA<short4, int4, char4, short4, op>  },
                    { FMXImpl_AOAOA<short, int, schar, int, op>, FMXImpl_AOAOA<short2, int2, char2, int2, op>, FMXImpl_AOAOA<short3, int3, char3, int3, op>, FMXImpl_AOAOA<short4, int4, char4, int4, op>  },
                    { FMXImpl_AOAOA<short, int, schar, float, op>, FMXImpl_AOAOA<short2, int2, char2, float2, op>, FMXImpl_AOAOA<short3, int3, char3, float3, op>, FMXImpl_AOAOA<short4, int4, char4, float4, op>  },
                    { FMXImpl_AOAOA<short, int, schar, double, op>, FMXImpl_AOAOA<short2, int2, char2, double2, op>, FMXImpl_AOAOA<short3, int3, char3, double3, op>, FMXImpl_AOAOA<short4, int4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, int, ushort, uchar, op>, FMXImpl_AOAOA<short2, int2, ushort2, uchar2, op>, FMXImpl_AOAOA<short3, int3, ushort3, uchar3, op>, FMXImpl_AOAOA<short4, int4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, int, ushort, schar, op>, FMXImpl_AOAOA<short2, int2, ushort2, char2, op>, FMXImpl_AOAOA<short3, int3, ushort3, char3, op>, FMXImpl_AOAOA<short4, int4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<short, int, ushort, ushort, op>, FMXImpl_AOAOA<short2, int2, ushort2, ushort2, op>, FMXImpl_AOAOA<short3, int3, ushort3, ushort3, op>, FMXImpl_AOAOA<short4, int4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, int, ushort, short, op>, FMXImpl_AOAOA<short2, int2, ushort2, short2, op>, FMXImpl_AOAOA<short3, int3, ushort3, short3, op>, FMXImpl_AOAOA<short4, int4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<short, int, ushort, int, op>, FMXImpl_AOAOA<short2, int2, ushort2, int2, op>, FMXImpl_AOAOA<short3, int3, ushort3, int3, op>, FMXImpl_AOAOA<short4, int4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<short, int, ushort, float, op>, FMXImpl_AOAOA<short2, int2, ushort2, float2, op>, FMXImpl_AOAOA<short3, int3, ushort3, float3, op>, FMXImpl_AOAOA<short4, int4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<short, int, ushort, double, op>, FMXImpl_AOAOA<short2, int2, ushort2, double2, op>, FMXImpl_AOAOA<short3, int3, ushort3, double3, op>, FMXImpl_AOAOA<short4, int4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, int, short, uchar, op>, FMXImpl_AOAOA<short2, int2, short2, uchar2, op>, FMXImpl_AOAOA<short3, int3, short3, uchar3, op>, FMXImpl_AOAOA<short4, int4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, int, short, schar, op>, FMXImpl_AOAOA<short2, int2, short2, char2, op>, FMXImpl_AOAOA<short3, int3, short3, char3, op>, FMXImpl_AOAOA<short4, int4, short4, char4, op>  },
                    { FMXImpl_AOAOA<short, int, short, ushort, op>, FMXImpl_AOAOA<short2, int2, short2, ushort2, op>, FMXImpl_AOAOA<short3, int3, short3, ushort3, op>, FMXImpl_AOAOA<short4, int4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, int, short, short, op>, FMXImpl_AOAOA<short2, int2, short2, short2, op>, FMXImpl_AOAOA<short3, int3, short3, short3, op>, FMXImpl_AOAOA<short4, int4, short4, short4, op>  },
                    { FMXImpl_AOAOA<short, int, short, int, op>, FMXImpl_AOAOA<short2, int2, short2, int2, op>, FMXImpl_AOAOA<short3, int3, short3, int3, op>, FMXImpl_AOAOA<short4, int4, short4, int4, op>  },
                    { FMXImpl_AOAOA<short, int, short, float, op>, FMXImpl_AOAOA<short2, int2, short2, float2, op>, FMXImpl_AOAOA<short3, int3, short3, float3, op>, FMXImpl_AOAOA<short4, int4, short4, float4, op>  },
                    { FMXImpl_AOAOA<short, int, short, double, op>, FMXImpl_AOAOA<short2, int2, short2, double2, op>, FMXImpl_AOAOA<short3, int3, short3, double3, op>, FMXImpl_AOAOA<short4, int4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, int, int, uchar, op>, FMXImpl_AOAOA<short2, int2, int2, uchar2, op>, FMXImpl_AOAOA<short3, int3, int3, uchar3, op>, FMXImpl_AOAOA<short4, int4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, int, int, schar, op>, FMXImpl_AOAOA<short2, int2, int2, char2, op>, FMXImpl_AOAOA<short3, int3, int3, char3, op>, FMXImpl_AOAOA<short4, int4, int4, char4, op>  },
                    { FMXImpl_AOAOA<short, int, int, ushort, op>, FMXImpl_AOAOA<short2, int2, int2, ushort2, op>, FMXImpl_AOAOA<short3, int3, int3, ushort3, op>, FMXImpl_AOAOA<short4, int4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, int, int, short, op>, FMXImpl_AOAOA<short2, int2, int2, short2, op>, FMXImpl_AOAOA<short3, int3, int3, short3, op>, FMXImpl_AOAOA<short4, int4, int4, short4, op>  },
                    { FMXImpl_AOAOA<short, int, int, int, op>, FMXImpl_AOAOA<short2, int2, int2, int2, op>, FMXImpl_AOAOA<short3, int3, int3, int3, op>, FMXImpl_AOAOA<short4, int4, int4, int4, op>  },
                    { FMXImpl_AOAOA<short, int, int, float, op>, FMXImpl_AOAOA<short2, int2, int2, float2, op>, FMXImpl_AOAOA<short3, int3, int3, float3, op>, FMXImpl_AOAOA<short4, int4, int4, float4, op>  },
                    { FMXImpl_AOAOA<short, int, int, double, op>, FMXImpl_AOAOA<short2, int2, int2, double2, op>, FMXImpl_AOAOA<short3, int3, int3, double3, op>, FMXImpl_AOAOA<short4, int4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, int, float, uchar, op>, FMXImpl_AOAOA<short2, int2, float2, uchar2, op>, FMXImpl_AOAOA<short3, int3, float3, uchar3, op>, FMXImpl_AOAOA<short4, int4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, int, float, schar, op>, FMXImpl_AOAOA<short2, int2, float2, char2, op>, FMXImpl_AOAOA<short3, int3, float3, char3, op>, FMXImpl_AOAOA<short4, int4, float4, char4, op>  },
                    { FMXImpl_AOAOA<short, int, float, ushort, op>, FMXImpl_AOAOA<short2, int2, float2, ushort2, op>, FMXImpl_AOAOA<short3, int3, float3, ushort3, op>, FMXImpl_AOAOA<short4, int4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, int, float, short, op>, FMXImpl_AOAOA<short2, int2, float2, short2, op>, FMXImpl_AOAOA<short3, int3, float3, short3, op>, FMXImpl_AOAOA<short4, int4, float4, short4, op>  },
                    { FMXImpl_AOAOA<short, int, float, int, op>, FMXImpl_AOAOA<short2, int2, float2, int2, op>, FMXImpl_AOAOA<short3, int3, float3, int3, op>, FMXImpl_AOAOA<short4, int4, float4, int4, op>  },
                    { FMXImpl_AOAOA<short, int, float, float, op>, FMXImpl_AOAOA<short2, int2, float2, float2, op>, FMXImpl_AOAOA<short3, int3, float3, float3, op>, FMXImpl_AOAOA<short4, int4, float4, float4, op>  },
                    { FMXImpl_AOAOA<short, int, float, double, op>, FMXImpl_AOAOA<short2, int2, float2, double2, op>, FMXImpl_AOAOA<short3, int3, float3, double3, op>, FMXImpl_AOAOA<short4, int4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, int, double, uchar, op>, FMXImpl_AOAOA<short2, int2, double2, uchar2, op>, FMXImpl_AOAOA<short3, int3, double3, uchar3, op>, FMXImpl_AOAOA<short4, int4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, int, double, schar, op>, FMXImpl_AOAOA<short2, int2, double2, char2, op>, FMXImpl_AOAOA<short3, int3, double3, char3, op>, FMXImpl_AOAOA<short4, int4, double4, char4, op>  },
                    { FMXImpl_AOAOA<short, int, double, ushort, op>, FMXImpl_AOAOA<short2, int2, double2, ushort2, op>, FMXImpl_AOAOA<short3, int3, double3, ushort3, op>, FMXImpl_AOAOA<short4, int4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, int, double, short, op>, FMXImpl_AOAOA<short2, int2, double2, short2, op>, FMXImpl_AOAOA<short3, int3, double3, short3, op>, FMXImpl_AOAOA<short4, int4, double4, short4, op>  },
                    { FMXImpl_AOAOA<short, int, double, int, op>, FMXImpl_AOAOA<short2, int2, double2, int2, op>, FMXImpl_AOAOA<short3, int3, double3, int3, op>, FMXImpl_AOAOA<short4, int4, double4, int4, op>  },
                    { FMXImpl_AOAOA<short, int, double, float, op>, FMXImpl_AOAOA<short2, int2, double2, float2, op>, FMXImpl_AOAOA<short3, int3, double3, float3, op>, FMXImpl_AOAOA<short4, int4, double4, float4, op>  },
                    { FMXImpl_AOAOA<short, int, double, double, op>, FMXImpl_AOAOA<short2, int2, double2, double2, op>, FMXImpl_AOAOA<short3, int3, double3, double3, op>, FMXImpl_AOAOA<short4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<short, float, uchar, uchar, op>, FMXImpl_AOAOA<short2, float2, uchar2, uchar2, op>, FMXImpl_AOAOA<short3, float3, uchar3, uchar3, op>, FMXImpl_AOAOA<short4, float4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, float, uchar, schar, op>, FMXImpl_AOAOA<short2, float2, uchar2, char2, op>, FMXImpl_AOAOA<short3, float3, uchar3, char3, op>, FMXImpl_AOAOA<short4, float4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<short, float, uchar, ushort, op>, FMXImpl_AOAOA<short2, float2, uchar2, ushort2, op>, FMXImpl_AOAOA<short3, float3, uchar3, ushort3, op>, FMXImpl_AOAOA<short4, float4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, float, uchar, short, op>, FMXImpl_AOAOA<short2, float2, uchar2, short2, op>, FMXImpl_AOAOA<short3, float3, uchar3, short3, op>, FMXImpl_AOAOA<short4, float4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<short, float, uchar, int, op>, FMXImpl_AOAOA<short2, float2, uchar2, int2, op>, FMXImpl_AOAOA<short3, float3, uchar3, int3, op>, FMXImpl_AOAOA<short4, float4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<short, float, uchar, float, op>, FMXImpl_AOAOA<short2, float2, uchar2, float2, op>, FMXImpl_AOAOA<short3, float3, uchar3, float3, op>, FMXImpl_AOAOA<short4, float4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<short, float, uchar, double, op>, FMXImpl_AOAOA<short2, float2, uchar2, double2, op>, FMXImpl_AOAOA<short3, float3, uchar3, double3, op>, FMXImpl_AOAOA<short4, float4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, float, schar, uchar, op>, FMXImpl_AOAOA<short2, float2, char2, uchar2, op>, FMXImpl_AOAOA<short3, float3, char3, uchar3, op>, FMXImpl_AOAOA<short4, float4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, float, schar, schar, op>, FMXImpl_AOAOA<short2, float2, char2, char2, op>, FMXImpl_AOAOA<short3, float3, char3, char3, op>, FMXImpl_AOAOA<short4, float4, char4, char4, op>  },
                    { FMXImpl_AOAOA<short, float, schar, ushort, op>, FMXImpl_AOAOA<short2, float2, char2, ushort2, op>, FMXImpl_AOAOA<short3, float3, char3, ushort3, op>, FMXImpl_AOAOA<short4, float4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, float, schar, short, op>, FMXImpl_AOAOA<short2, float2, char2, short2, op>, FMXImpl_AOAOA<short3, float3, char3, short3, op>, FMXImpl_AOAOA<short4, float4, char4, short4, op>  },
                    { FMXImpl_AOAOA<short, float, schar, int, op>, FMXImpl_AOAOA<short2, float2, char2, int2, op>, FMXImpl_AOAOA<short3, float3, char3, int3, op>, FMXImpl_AOAOA<short4, float4, char4, int4, op>  },
                    { FMXImpl_AOAOA<short, float, schar, float, op>, FMXImpl_AOAOA<short2, float2, char2, float2, op>, FMXImpl_AOAOA<short3, float3, char3, float3, op>, FMXImpl_AOAOA<short4, float4, char4, float4, op>  },
                    { FMXImpl_AOAOA<short, float, schar, double, op>, FMXImpl_AOAOA<short2, float2, char2, double2, op>, FMXImpl_AOAOA<short3, float3, char3, double3, op>, FMXImpl_AOAOA<short4, float4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, float, ushort, uchar, op>, FMXImpl_AOAOA<short2, float2, ushort2, uchar2, op>, FMXImpl_AOAOA<short3, float3, ushort3, uchar3, op>, FMXImpl_AOAOA<short4, float4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, float, ushort, schar, op>, FMXImpl_AOAOA<short2, float2, ushort2, char2, op>, FMXImpl_AOAOA<short3, float3, ushort3, char3, op>, FMXImpl_AOAOA<short4, float4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<short, float, ushort, ushort, op>, FMXImpl_AOAOA<short2, float2, ushort2, ushort2, op>, FMXImpl_AOAOA<short3, float3, ushort3, ushort3, op>, FMXImpl_AOAOA<short4, float4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, float, ushort, short, op>, FMXImpl_AOAOA<short2, float2, ushort2, short2, op>, FMXImpl_AOAOA<short3, float3, ushort3, short3, op>, FMXImpl_AOAOA<short4, float4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<short, float, ushort, int, op>, FMXImpl_AOAOA<short2, float2, ushort2, int2, op>, FMXImpl_AOAOA<short3, float3, ushort3, int3, op>, FMXImpl_AOAOA<short4, float4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<short, float, ushort, float, op>, FMXImpl_AOAOA<short2, float2, ushort2, float2, op>, FMXImpl_AOAOA<short3, float3, ushort3, float3, op>, FMXImpl_AOAOA<short4, float4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<short, float, ushort, double, op>, FMXImpl_AOAOA<short2, float2, ushort2, double2, op>, FMXImpl_AOAOA<short3, float3, ushort3, double3, op>, FMXImpl_AOAOA<short4, float4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, float, short, uchar, op>, FMXImpl_AOAOA<short2, float2, short2, uchar2, op>, FMXImpl_AOAOA<short3, float3, short3, uchar3, op>, FMXImpl_AOAOA<short4, float4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, float, short, schar, op>, FMXImpl_AOAOA<short2, float2, short2, char2, op>, FMXImpl_AOAOA<short3, float3, short3, char3, op>, FMXImpl_AOAOA<short4, float4, short4, char4, op>  },
                    { FMXImpl_AOAOA<short, float, short, ushort, op>, FMXImpl_AOAOA<short2, float2, short2, ushort2, op>, FMXImpl_AOAOA<short3, float3, short3, ushort3, op>, FMXImpl_AOAOA<short4, float4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, float, short, short, op>, FMXImpl_AOAOA<short2, float2, short2, short2, op>, FMXImpl_AOAOA<short3, float3, short3, short3, op>, FMXImpl_AOAOA<short4, float4, short4, short4, op>  },
                    { FMXImpl_AOAOA<short, float, short, int, op>, FMXImpl_AOAOA<short2, float2, short2, int2, op>, FMXImpl_AOAOA<short3, float3, short3, int3, op>, FMXImpl_AOAOA<short4, float4, short4, int4, op>  },
                    { FMXImpl_AOAOA<short, float, short, float, op>, FMXImpl_AOAOA<short2, float2, short2, float2, op>, FMXImpl_AOAOA<short3, float3, short3, float3, op>, FMXImpl_AOAOA<short4, float4, short4, float4, op>  },
                    { FMXImpl_AOAOA<short, float, short, double, op>, FMXImpl_AOAOA<short2, float2, short2, double2, op>, FMXImpl_AOAOA<short3, float3, short3, double3, op>, FMXImpl_AOAOA<short4, float4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, float, int, uchar, op>, FMXImpl_AOAOA<short2, float2, int2, uchar2, op>, FMXImpl_AOAOA<short3, float3, int3, uchar3, op>, FMXImpl_AOAOA<short4, float4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, float, int, schar, op>, FMXImpl_AOAOA<short2, float2, int2, char2, op>, FMXImpl_AOAOA<short3, float3, int3, char3, op>, FMXImpl_AOAOA<short4, float4, int4, char4, op>  },
                    { FMXImpl_AOAOA<short, float, int, ushort, op>, FMXImpl_AOAOA<short2, float2, int2, ushort2, op>, FMXImpl_AOAOA<short3, float3, int3, ushort3, op>, FMXImpl_AOAOA<short4, float4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, float, int, short, op>, FMXImpl_AOAOA<short2, float2, int2, short2, op>, FMXImpl_AOAOA<short3, float3, int3, short3, op>, FMXImpl_AOAOA<short4, float4, int4, short4, op>  },
                    { FMXImpl_AOAOA<short, float, int, int, op>, FMXImpl_AOAOA<short2, float2, int2, int2, op>, FMXImpl_AOAOA<short3, float3, int3, int3, op>, FMXImpl_AOAOA<short4, float4, int4, int4, op>  },
                    { FMXImpl_AOAOA<short, float, int, float, op>, FMXImpl_AOAOA<short2, float2, int2, float2, op>, FMXImpl_AOAOA<short3, float3, int3, float3, op>, FMXImpl_AOAOA<short4, float4, int4, float4, op>  },
                    { FMXImpl_AOAOA<short, float, int, double, op>, FMXImpl_AOAOA<short2, float2, int2, double2, op>, FMXImpl_AOAOA<short3, float3, int3, double3, op>, FMXImpl_AOAOA<short4, float4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, float, float, uchar, op>, FMXImpl_AOAOA<short2, float2, float2, uchar2, op>, FMXImpl_AOAOA<short3, float3, float3, uchar3, op>, FMXImpl_AOAOA<short4, float4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, float, float, schar, op>, FMXImpl_AOAOA<short2, float2, float2, char2, op>, FMXImpl_AOAOA<short3, float3, float3, char3, op>, FMXImpl_AOAOA<short4, float4, float4, char4, op>  },
                    { FMXImpl_AOAOA<short, float, float, ushort, op>, FMXImpl_AOAOA<short2, float2, float2, ushort2, op>, FMXImpl_AOAOA<short3, float3, float3, ushort3, op>, FMXImpl_AOAOA<short4, float4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, float, float, short, op>, FMXImpl_AOAOA<short2, float2, float2, short2, op>, FMXImpl_AOAOA<short3, float3, float3, short3, op>, FMXImpl_AOAOA<short4, float4, float4, short4, op>  },
                    { FMXImpl_AOAOA<short, float, float, int, op>, FMXImpl_AOAOA<short2, float2, float2, int2, op>, FMXImpl_AOAOA<short3, float3, float3, int3, op>, FMXImpl_AOAOA<short4, float4, float4, int4, op>  },
                    { FMXImpl_AOAOA<short, float, float, float, op>, FMXImpl_AOAOA<short2, float2, float2, float2, op>, FMXImpl_AOAOA<short3, float3, float3, float3, op>, FMXImpl_AOAOA<short4, float4, float4, float4, op>  },
                    { FMXImpl_AOAOA<short, float, float, double, op>, FMXImpl_AOAOA<short2, float2, float2, double2, op>, FMXImpl_AOAOA<short3, float3, float3, double3, op>, FMXImpl_AOAOA<short4, float4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, float, double, uchar, op>, FMXImpl_AOAOA<short2, float2, double2, uchar2, op>, FMXImpl_AOAOA<short3, float3, double3, uchar3, op>, FMXImpl_AOAOA<short4, float4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, float, double, schar, op>, FMXImpl_AOAOA<short2, float2, double2, char2, op>, FMXImpl_AOAOA<short3, float3, double3, char3, op>, FMXImpl_AOAOA<short4, float4, double4, char4, op>  },
                    { FMXImpl_AOAOA<short, float, double, ushort, op>, FMXImpl_AOAOA<short2, float2, double2, ushort2, op>, FMXImpl_AOAOA<short3, float3, double3, ushort3, op>, FMXImpl_AOAOA<short4, float4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, float, double, short, op>, FMXImpl_AOAOA<short2, float2, double2, short2, op>, FMXImpl_AOAOA<short3, float3, double3, short3, op>, FMXImpl_AOAOA<short4, float4, double4, short4, op>  },
                    { FMXImpl_AOAOA<short, float, double, int, op>, FMXImpl_AOAOA<short2, float2, double2, int2, op>, FMXImpl_AOAOA<short3, float3, double3, int3, op>, FMXImpl_AOAOA<short4, float4, double4, int4, op>  },
                    { FMXImpl_AOAOA<short, float, double, float, op>, FMXImpl_AOAOA<short2, float2, double2, float2, op>, FMXImpl_AOAOA<short3, float3, double3, float3, op>, FMXImpl_AOAOA<short4, float4, double4, float4, op>  },
                    { FMXImpl_AOAOA<short, float, double, double, op>, FMXImpl_AOAOA<short2, float2, double2, double2, op>, FMXImpl_AOAOA<short3, float3, double3, double3, op>, FMXImpl_AOAOA<short4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<short, double, uchar, uchar, op>, FMXImpl_AOAOA<short2, double2, uchar2, uchar2, op>, FMXImpl_AOAOA<short3, double3, uchar3, uchar3, op>, FMXImpl_AOAOA<short4, double4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, double, uchar, schar, op>, FMXImpl_AOAOA<short2, double2, uchar2, char2, op>, FMXImpl_AOAOA<short3, double3, uchar3, char3, op>, FMXImpl_AOAOA<short4, double4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<short, double, uchar, ushort, op>, FMXImpl_AOAOA<short2, double2, uchar2, ushort2, op>, FMXImpl_AOAOA<short3, double3, uchar3, ushort3, op>, FMXImpl_AOAOA<short4, double4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, double, uchar, short, op>, FMXImpl_AOAOA<short2, double2, uchar2, short2, op>, FMXImpl_AOAOA<short3, double3, uchar3, short3, op>, FMXImpl_AOAOA<short4, double4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<short, double, uchar, int, op>, FMXImpl_AOAOA<short2, double2, uchar2, int2, op>, FMXImpl_AOAOA<short3, double3, uchar3, int3, op>, FMXImpl_AOAOA<short4, double4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<short, double, uchar, float, op>, FMXImpl_AOAOA<short2, double2, uchar2, float2, op>, FMXImpl_AOAOA<short3, double3, uchar3, float3, op>, FMXImpl_AOAOA<short4, double4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<short, double, uchar, double, op>, FMXImpl_AOAOA<short2, double2, uchar2, double2, op>, FMXImpl_AOAOA<short3, double3, uchar3, double3, op>, FMXImpl_AOAOA<short4, double4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, double, schar, uchar, op>, FMXImpl_AOAOA<short2, double2, char2, uchar2, op>, FMXImpl_AOAOA<short3, double3, char3, uchar3, op>, FMXImpl_AOAOA<short4, double4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, double, schar, schar, op>, FMXImpl_AOAOA<short2, double2, char2, char2, op>, FMXImpl_AOAOA<short3, double3, char3, char3, op>, FMXImpl_AOAOA<short4, double4, char4, char4, op>  },
                    { FMXImpl_AOAOA<short, double, schar, ushort, op>, FMXImpl_AOAOA<short2, double2, char2, ushort2, op>, FMXImpl_AOAOA<short3, double3, char3, ushort3, op>, FMXImpl_AOAOA<short4, double4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, double, schar, short, op>, FMXImpl_AOAOA<short2, double2, char2, short2, op>, FMXImpl_AOAOA<short3, double3, char3, short3, op>, FMXImpl_AOAOA<short4, double4, char4, short4, op>  },
                    { FMXImpl_AOAOA<short, double, schar, int, op>, FMXImpl_AOAOA<short2, double2, char2, int2, op>, FMXImpl_AOAOA<short3, double3, char3, int3, op>, FMXImpl_AOAOA<short4, double4, char4, int4, op>  },
                    { FMXImpl_AOAOA<short, double, schar, float, op>, FMXImpl_AOAOA<short2, double2, char2, float2, op>, FMXImpl_AOAOA<short3, double3, char3, float3, op>, FMXImpl_AOAOA<short4, double4, char4, float4, op>  },
                    { FMXImpl_AOAOA<short, double, schar, double, op>, FMXImpl_AOAOA<short2, double2, char2, double2, op>, FMXImpl_AOAOA<short3, double3, char3, double3, op>, FMXImpl_AOAOA<short4, double4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, double, ushort, uchar, op>, FMXImpl_AOAOA<short2, double2, ushort2, uchar2, op>, FMXImpl_AOAOA<short3, double3, ushort3, uchar3, op>, FMXImpl_AOAOA<short4, double4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, double, ushort, schar, op>, FMXImpl_AOAOA<short2, double2, ushort2, char2, op>, FMXImpl_AOAOA<short3, double3, ushort3, char3, op>, FMXImpl_AOAOA<short4, double4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<short, double, ushort, ushort, op>, FMXImpl_AOAOA<short2, double2, ushort2, ushort2, op>, FMXImpl_AOAOA<short3, double3, ushort3, ushort3, op>, FMXImpl_AOAOA<short4, double4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, double, ushort, short, op>, FMXImpl_AOAOA<short2, double2, ushort2, short2, op>, FMXImpl_AOAOA<short3, double3, ushort3, short3, op>, FMXImpl_AOAOA<short4, double4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<short, double, ushort, int, op>, FMXImpl_AOAOA<short2, double2, ushort2, int2, op>, FMXImpl_AOAOA<short3, double3, ushort3, int3, op>, FMXImpl_AOAOA<short4, double4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<short, double, ushort, float, op>, FMXImpl_AOAOA<short2, double2, ushort2, float2, op>, FMXImpl_AOAOA<short3, double3, ushort3, float3, op>, FMXImpl_AOAOA<short4, double4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<short, double, ushort, double, op>, FMXImpl_AOAOA<short2, double2, ushort2, double2, op>, FMXImpl_AOAOA<short3, double3, ushort3, double3, op>, FMXImpl_AOAOA<short4, double4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, double, short, uchar, op>, FMXImpl_AOAOA<short2, double2, short2, uchar2, op>, FMXImpl_AOAOA<short3, double3, short3, uchar3, op>, FMXImpl_AOAOA<short4, double4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, double, short, schar, op>, FMXImpl_AOAOA<short2, double2, short2, char2, op>, FMXImpl_AOAOA<short3, double3, short3, char3, op>, FMXImpl_AOAOA<short4, double4, short4, char4, op>  },
                    { FMXImpl_AOAOA<short, double, short, ushort, op>, FMXImpl_AOAOA<short2, double2, short2, ushort2, op>, FMXImpl_AOAOA<short3, double3, short3, ushort3, op>, FMXImpl_AOAOA<short4, double4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, double, short, short, op>, FMXImpl_AOAOA<short2, double2, short2, short2, op>, FMXImpl_AOAOA<short3, double3, short3, short3, op>, FMXImpl_AOAOA<short4, double4, short4, short4, op>  },
                    { FMXImpl_AOAOA<short, double, short, int, op>, FMXImpl_AOAOA<short2, double2, short2, int2, op>, FMXImpl_AOAOA<short3, double3, short3, int3, op>, FMXImpl_AOAOA<short4, double4, short4, int4, op>  },
                    { FMXImpl_AOAOA<short, double, short, float, op>, FMXImpl_AOAOA<short2, double2, short2, float2, op>, FMXImpl_AOAOA<short3, double3, short3, float3, op>, FMXImpl_AOAOA<short4, double4, short4, float4, op>  },
                    { FMXImpl_AOAOA<short, double, short, double, op>, FMXImpl_AOAOA<short2, double2, short2, double2, op>, FMXImpl_AOAOA<short3, double3, short3, double3, op>, FMXImpl_AOAOA<short4, double4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, double, int, uchar, op>, FMXImpl_AOAOA<short2, double2, int2, uchar2, op>, FMXImpl_AOAOA<short3, double3, int3, uchar3, op>, FMXImpl_AOAOA<short4, double4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, double, int, schar, op>, FMXImpl_AOAOA<short2, double2, int2, char2, op>, FMXImpl_AOAOA<short3, double3, int3, char3, op>, FMXImpl_AOAOA<short4, double4, int4, char4, op>  },
                    { FMXImpl_AOAOA<short, double, int, ushort, op>, FMXImpl_AOAOA<short2, double2, int2, ushort2, op>, FMXImpl_AOAOA<short3, double3, int3, ushort3, op>, FMXImpl_AOAOA<short4, double4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, double, int, short, op>, FMXImpl_AOAOA<short2, double2, int2, short2, op>, FMXImpl_AOAOA<short3, double3, int3, short3, op>, FMXImpl_AOAOA<short4, double4, int4, short4, op>  },
                    { FMXImpl_AOAOA<short, double, int, int, op>, FMXImpl_AOAOA<short2, double2, int2, int2, op>, FMXImpl_AOAOA<short3, double3, int3, int3, op>, FMXImpl_AOAOA<short4, double4, int4, int4, op>  },
                    { FMXImpl_AOAOA<short, double, int, float, op>, FMXImpl_AOAOA<short2, double2, int2, float2, op>, FMXImpl_AOAOA<short3, double3, int3, float3, op>, FMXImpl_AOAOA<short4, double4, int4, float4, op>  },
                    { FMXImpl_AOAOA<short, double, int, double, op>, FMXImpl_AOAOA<short2, double2, int2, double2, op>, FMXImpl_AOAOA<short3, double3, int3, double3, op>, FMXImpl_AOAOA<short4, double4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, double, float, uchar, op>, FMXImpl_AOAOA<short2, double2, float2, uchar2, op>, FMXImpl_AOAOA<short3, double3, float3, uchar3, op>, FMXImpl_AOAOA<short4, double4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, double, float, schar, op>, FMXImpl_AOAOA<short2, double2, float2, char2, op>, FMXImpl_AOAOA<short3, double3, float3, char3, op>, FMXImpl_AOAOA<short4, double4, float4, char4, op>  },
                    { FMXImpl_AOAOA<short, double, float, ushort, op>, FMXImpl_AOAOA<short2, double2, float2, ushort2, op>, FMXImpl_AOAOA<short3, double3, float3, ushort3, op>, FMXImpl_AOAOA<short4, double4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, double, float, short, op>, FMXImpl_AOAOA<short2, double2, float2, short2, op>, FMXImpl_AOAOA<short3, double3, float3, short3, op>, FMXImpl_AOAOA<short4, double4, float4, short4, op>  },
                    { FMXImpl_AOAOA<short, double, float, int, op>, FMXImpl_AOAOA<short2, double2, float2, int2, op>, FMXImpl_AOAOA<short3, double3, float3, int3, op>, FMXImpl_AOAOA<short4, double4, float4, int4, op>  },
                    { FMXImpl_AOAOA<short, double, float, float, op>, FMXImpl_AOAOA<short2, double2, float2, float2, op>, FMXImpl_AOAOA<short3, double3, float3, float3, op>, FMXImpl_AOAOA<short4, double4, float4, float4, op>  },
                    { FMXImpl_AOAOA<short, double, float, double, op>, FMXImpl_AOAOA<short2, double2, float2, double2, op>, FMXImpl_AOAOA<short3, double3, float3, double3, op>, FMXImpl_AOAOA<short4, double4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<short, double, double, uchar, op>, FMXImpl_AOAOA<short2, double2, double2, uchar2, op>, FMXImpl_AOAOA<short3, double3, double3, uchar3, op>, FMXImpl_AOAOA<short4, double4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<short, double, double, schar, op>, FMXImpl_AOAOA<short2, double2, double2, char2, op>, FMXImpl_AOAOA<short3, double3, double3, char3, op>, FMXImpl_AOAOA<short4, double4, double4, char4, op>  },
                    { FMXImpl_AOAOA<short, double, double, ushort, op>, FMXImpl_AOAOA<short2, double2, double2, ushort2, op>, FMXImpl_AOAOA<short3, double3, double3, ushort3, op>, FMXImpl_AOAOA<short4, double4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<short, double, double, short, op>, FMXImpl_AOAOA<short2, double2, double2, short2, op>, FMXImpl_AOAOA<short3, double3, double3, short3, op>, FMXImpl_AOAOA<short4, double4, double4, short4, op>  },
                    { FMXImpl_AOAOA<short, double, double, int, op>, FMXImpl_AOAOA<short2, double2, double2, int2, op>, FMXImpl_AOAOA<short3, double3, double3, int3, op>, FMXImpl_AOAOA<short4, double4, double4, int4, op>  },
                    { FMXImpl_AOAOA<short, double, double, float, op>, FMXImpl_AOAOA<short2, double2, double2, float2, op>, FMXImpl_AOAOA<short3, double3, double3, float3, op>, FMXImpl_AOAOA<short4, double4, double4, float4, op>  },
                    { FMXImpl_AOAOA<short, double, double, double, op>, FMXImpl_AOAOA<short2, double2, double2, double2, op>, FMXImpl_AOAOA<short3, double3, double3, double3, op>, FMXImpl_AOAOA<short4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { FMXImpl_AOAOA<int, uchar, uchar, uchar, op>, FMXImpl_AOAOA<int2, uchar2, uchar2, uchar2, op>, FMXImpl_AOAOA<int3, uchar3, uchar3, uchar3, op>, FMXImpl_AOAOA<int4, uchar4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, uchar, uchar, schar, op>, FMXImpl_AOAOA<int2, uchar2, uchar2, char2, op>, FMXImpl_AOAOA<int3, uchar3, uchar3, char3, op>, FMXImpl_AOAOA<int4, uchar4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<int, uchar, uchar, ushort, op>, FMXImpl_AOAOA<int2, uchar2, uchar2, ushort2, op>, FMXImpl_AOAOA<int3, uchar3, uchar3, ushort3, op>, FMXImpl_AOAOA<int4, uchar4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, uchar, uchar, short, op>, FMXImpl_AOAOA<int2, uchar2, uchar2, short2, op>, FMXImpl_AOAOA<int3, uchar3, uchar3, short3, op>, FMXImpl_AOAOA<int4, uchar4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<int, uchar, uchar, int, op>, FMXImpl_AOAOA<int2, uchar2, uchar2, int2, op>, FMXImpl_AOAOA<int3, uchar3, uchar3, int3, op>, FMXImpl_AOAOA<int4, uchar4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<int, uchar, uchar, float, op>, FMXImpl_AOAOA<int2, uchar2, uchar2, float2, op>, FMXImpl_AOAOA<int3, uchar3, uchar3, float3, op>, FMXImpl_AOAOA<int4, uchar4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<int, uchar, uchar, double, op>, FMXImpl_AOAOA<int2, uchar2, uchar2, double2, op>, FMXImpl_AOAOA<int3, uchar3, uchar3, double3, op>, FMXImpl_AOAOA<int4, uchar4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, uchar, schar, uchar, op>, FMXImpl_AOAOA<int2, uchar2, char2, uchar2, op>, FMXImpl_AOAOA<int3, uchar3, char3, uchar3, op>, FMXImpl_AOAOA<int4, uchar4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, uchar, schar, schar, op>, FMXImpl_AOAOA<int2, uchar2, char2, char2, op>, FMXImpl_AOAOA<int3, uchar3, char3, char3, op>, FMXImpl_AOAOA<int4, uchar4, char4, char4, op>  },
                    { FMXImpl_AOAOA<int, uchar, schar, ushort, op>, FMXImpl_AOAOA<int2, uchar2, char2, ushort2, op>, FMXImpl_AOAOA<int3, uchar3, char3, ushort3, op>, FMXImpl_AOAOA<int4, uchar4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, uchar, schar, short, op>, FMXImpl_AOAOA<int2, uchar2, char2, short2, op>, FMXImpl_AOAOA<int3, uchar3, char3, short3, op>, FMXImpl_AOAOA<int4, uchar4, char4, short4, op>  },
                    { FMXImpl_AOAOA<int, uchar, schar, int, op>, FMXImpl_AOAOA<int2, uchar2, char2, int2, op>, FMXImpl_AOAOA<int3, uchar3, char3, int3, op>, FMXImpl_AOAOA<int4, uchar4, char4, int4, op>  },
                    { FMXImpl_AOAOA<int, uchar, schar, float, op>, FMXImpl_AOAOA<int2, uchar2, char2, float2, op>, FMXImpl_AOAOA<int3, uchar3, char3, float3, op>, FMXImpl_AOAOA<int4, uchar4, char4, float4, op>  },
                    { FMXImpl_AOAOA<int, uchar, schar, double, op>, FMXImpl_AOAOA<int2, uchar2, char2, double2, op>, FMXImpl_AOAOA<int3, uchar3, char3, double3, op>, FMXImpl_AOAOA<int4, uchar4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, uchar, ushort, uchar, op>, FMXImpl_AOAOA<int2, uchar2, ushort2, uchar2, op>, FMXImpl_AOAOA<int3, uchar3, ushort3, uchar3, op>, FMXImpl_AOAOA<int4, uchar4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, uchar, ushort, schar, op>, FMXImpl_AOAOA<int2, uchar2, ushort2, char2, op>, FMXImpl_AOAOA<int3, uchar3, ushort3, char3, op>, FMXImpl_AOAOA<int4, uchar4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<int, uchar, ushort, ushort, op>, FMXImpl_AOAOA<int2, uchar2, ushort2, ushort2, op>, FMXImpl_AOAOA<int3, uchar3, ushort3, ushort3, op>, FMXImpl_AOAOA<int4, uchar4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, uchar, ushort, short, op>, FMXImpl_AOAOA<int2, uchar2, ushort2, short2, op>, FMXImpl_AOAOA<int3, uchar3, ushort3, short3, op>, FMXImpl_AOAOA<int4, uchar4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<int, uchar, ushort, int, op>, FMXImpl_AOAOA<int2, uchar2, ushort2, int2, op>, FMXImpl_AOAOA<int3, uchar3, ushort3, int3, op>, FMXImpl_AOAOA<int4, uchar4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<int, uchar, ushort, float, op>, FMXImpl_AOAOA<int2, uchar2, ushort2, float2, op>, FMXImpl_AOAOA<int3, uchar3, ushort3, float3, op>, FMXImpl_AOAOA<int4, uchar4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<int, uchar, ushort, double, op>, FMXImpl_AOAOA<int2, uchar2, ushort2, double2, op>, FMXImpl_AOAOA<int3, uchar3, ushort3, double3, op>, FMXImpl_AOAOA<int4, uchar4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, uchar, short, uchar, op>, FMXImpl_AOAOA<int2, uchar2, short2, uchar2, op>, FMXImpl_AOAOA<int3, uchar3, short3, uchar3, op>, FMXImpl_AOAOA<int4, uchar4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, uchar, short, schar, op>, FMXImpl_AOAOA<int2, uchar2, short2, char2, op>, FMXImpl_AOAOA<int3, uchar3, short3, char3, op>, FMXImpl_AOAOA<int4, uchar4, short4, char4, op>  },
                    { FMXImpl_AOAOA<int, uchar, short, ushort, op>, FMXImpl_AOAOA<int2, uchar2, short2, ushort2, op>, FMXImpl_AOAOA<int3, uchar3, short3, ushort3, op>, FMXImpl_AOAOA<int4, uchar4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, uchar, short, short, op>, FMXImpl_AOAOA<int2, uchar2, short2, short2, op>, FMXImpl_AOAOA<int3, uchar3, short3, short3, op>, FMXImpl_AOAOA<int4, uchar4, short4, short4, op>  },
                    { FMXImpl_AOAOA<int, uchar, short, int, op>, FMXImpl_AOAOA<int2, uchar2, short2, int2, op>, FMXImpl_AOAOA<int3, uchar3, short3, int3, op>, FMXImpl_AOAOA<int4, uchar4, short4, int4, op>  },
                    { FMXImpl_AOAOA<int, uchar, short, float, op>, FMXImpl_AOAOA<int2, uchar2, short2, float2, op>, FMXImpl_AOAOA<int3, uchar3, short3, float3, op>, FMXImpl_AOAOA<int4, uchar4, short4, float4, op>  },
                    { FMXImpl_AOAOA<int, uchar, short, double, op>, FMXImpl_AOAOA<int2, uchar2, short2, double2, op>, FMXImpl_AOAOA<int3, uchar3, short3, double3, op>, FMXImpl_AOAOA<int4, uchar4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, uchar, int, uchar, op>, FMXImpl_AOAOA<int2, uchar2, int2, uchar2, op>, FMXImpl_AOAOA<int3, uchar3, int3, uchar3, op>, FMXImpl_AOAOA<int4, uchar4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, uchar, int, schar, op>, FMXImpl_AOAOA<int2, uchar2, int2, char2, op>, FMXImpl_AOAOA<int3, uchar3, int3, char3, op>, FMXImpl_AOAOA<int4, uchar4, int4, char4, op>  },
                    { FMXImpl_AOAOA<int, uchar, int, ushort, op>, FMXImpl_AOAOA<int2, uchar2, int2, ushort2, op>, FMXImpl_AOAOA<int3, uchar3, int3, ushort3, op>, FMXImpl_AOAOA<int4, uchar4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, uchar, int, short, op>, FMXImpl_AOAOA<int2, uchar2, int2, short2, op>, FMXImpl_AOAOA<int3, uchar3, int3, short3, op>, FMXImpl_AOAOA<int4, uchar4, int4, short4, op>  },
                    { FMXImpl_AOAOA<int, uchar, int, int, op>, FMXImpl_AOAOA<int2, uchar2, int2, int2, op>, FMXImpl_AOAOA<int3, uchar3, int3, int3, op>, FMXImpl_AOAOA<int4, uchar4, int4, int4, op>  },
                    { FMXImpl_AOAOA<int, uchar, int, float, op>, FMXImpl_AOAOA<int2, uchar2, int2, float2, op>, FMXImpl_AOAOA<int3, uchar3, int3, float3, op>, FMXImpl_AOAOA<int4, uchar4, int4, float4, op>  },
                    { FMXImpl_AOAOA<int, uchar, int, double, op>, FMXImpl_AOAOA<int2, uchar2, int2, double2, op>, FMXImpl_AOAOA<int3, uchar3, int3, double3, op>, FMXImpl_AOAOA<int4, uchar4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, uchar, float, uchar, op>, FMXImpl_AOAOA<int2, uchar2, float2, uchar2, op>, FMXImpl_AOAOA<int3, uchar3, float3, uchar3, op>, FMXImpl_AOAOA<int4, uchar4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, uchar, float, schar, op>, FMXImpl_AOAOA<int2, uchar2, float2, char2, op>, FMXImpl_AOAOA<int3, uchar3, float3, char3, op>, FMXImpl_AOAOA<int4, uchar4, float4, char4, op>  },
                    { FMXImpl_AOAOA<int, uchar, float, ushort, op>, FMXImpl_AOAOA<int2, uchar2, float2, ushort2, op>, FMXImpl_AOAOA<int3, uchar3, float3, ushort3, op>, FMXImpl_AOAOA<int4, uchar4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, uchar, float, short, op>, FMXImpl_AOAOA<int2, uchar2, float2, short2, op>, FMXImpl_AOAOA<int3, uchar3, float3, short3, op>, FMXImpl_AOAOA<int4, uchar4, float4, short4, op>  },
                    { FMXImpl_AOAOA<int, uchar, float, int, op>, FMXImpl_AOAOA<int2, uchar2, float2, int2, op>, FMXImpl_AOAOA<int3, uchar3, float3, int3, op>, FMXImpl_AOAOA<int4, uchar4, float4, int4, op>  },
                    { FMXImpl_AOAOA<int, uchar, float, float, op>, FMXImpl_AOAOA<int2, uchar2, float2, float2, op>, FMXImpl_AOAOA<int3, uchar3, float3, float3, op>, FMXImpl_AOAOA<int4, uchar4, float4, float4, op>  },
                    { FMXImpl_AOAOA<int, uchar, float, double, op>, FMXImpl_AOAOA<int2, uchar2, float2, double2, op>, FMXImpl_AOAOA<int3, uchar3, float3, double3, op>, FMXImpl_AOAOA<int4, uchar4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, uchar, double, uchar, op>, FMXImpl_AOAOA<int2, uchar2, double2, uchar2, op>, FMXImpl_AOAOA<int3, uchar3, double3, uchar3, op>, FMXImpl_AOAOA<int4, uchar4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, uchar, double, schar, op>, FMXImpl_AOAOA<int2, uchar2, double2, char2, op>, FMXImpl_AOAOA<int3, uchar3, double3, char3, op>, FMXImpl_AOAOA<int4, uchar4, double4, char4, op>  },
                    { FMXImpl_AOAOA<int, uchar, double, ushort, op>, FMXImpl_AOAOA<int2, uchar2, double2, ushort2, op>, FMXImpl_AOAOA<int3, uchar3, double3, ushort3, op>, FMXImpl_AOAOA<int4, uchar4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, uchar, double, short, op>, FMXImpl_AOAOA<int2, uchar2, double2, short2, op>, FMXImpl_AOAOA<int3, uchar3, double3, short3, op>, FMXImpl_AOAOA<int4, uchar4, double4, short4, op>  },
                    { FMXImpl_AOAOA<int, uchar, double, int, op>, FMXImpl_AOAOA<int2, uchar2, double2, int2, op>, FMXImpl_AOAOA<int3, uchar3, double3, int3, op>, FMXImpl_AOAOA<int4, uchar4, double4, int4, op>  },
                    { FMXImpl_AOAOA<int, uchar, double, float, op>, FMXImpl_AOAOA<int2, uchar2, double2, float2, op>, FMXImpl_AOAOA<int3, uchar3, double3, float3, op>, FMXImpl_AOAOA<int4, uchar4, double4, float4, op>  },
                    { FMXImpl_AOAOA<int, uchar, double, double, op>, FMXImpl_AOAOA<int2, uchar2, double2, double2, op>, FMXImpl_AOAOA<int3, uchar3, double3, double3, op>, FMXImpl_AOAOA<int4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<int, schar, uchar, uchar, op>, FMXImpl_AOAOA<int2, char2, uchar2, uchar2, op>, FMXImpl_AOAOA<int3, char3, uchar3, uchar3, op>, FMXImpl_AOAOA<int4, char4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, schar, uchar, schar, op>, FMXImpl_AOAOA<int2, char2, uchar2, char2, op>, FMXImpl_AOAOA<int3, char3, uchar3, char3, op>, FMXImpl_AOAOA<int4, char4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<int, schar, uchar, ushort, op>, FMXImpl_AOAOA<int2, char2, uchar2, ushort2, op>, FMXImpl_AOAOA<int3, char3, uchar3, ushort3, op>, FMXImpl_AOAOA<int4, char4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, schar, uchar, short, op>, FMXImpl_AOAOA<int2, char2, uchar2, short2, op>, FMXImpl_AOAOA<int3, char3, uchar3, short3, op>, FMXImpl_AOAOA<int4, char4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<int, schar, uchar, int, op>, FMXImpl_AOAOA<int2, char2, uchar2, int2, op>, FMXImpl_AOAOA<int3, char3, uchar3, int3, op>, FMXImpl_AOAOA<int4, char4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<int, schar, uchar, float, op>, FMXImpl_AOAOA<int2, char2, uchar2, float2, op>, FMXImpl_AOAOA<int3, char3, uchar3, float3, op>, FMXImpl_AOAOA<int4, char4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<int, schar, uchar, double, op>, FMXImpl_AOAOA<int2, char2, uchar2, double2, op>, FMXImpl_AOAOA<int3, char3, uchar3, double3, op>, FMXImpl_AOAOA<int4, char4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, schar, schar, uchar, op>, FMXImpl_AOAOA<int2, char2, char2, uchar2, op>, FMXImpl_AOAOA<int3, char3, char3, uchar3, op>, FMXImpl_AOAOA<int4, char4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, schar, schar, schar, op>, FMXImpl_AOAOA<int2, char2, char2, char2, op>, FMXImpl_AOAOA<int3, char3, char3, char3, op>, FMXImpl_AOAOA<int4, char4, char4, char4, op>  },
                    { FMXImpl_AOAOA<int, schar, schar, ushort, op>, FMXImpl_AOAOA<int2, char2, char2, ushort2, op>, FMXImpl_AOAOA<int3, char3, char3, ushort3, op>, FMXImpl_AOAOA<int4, char4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, schar, schar, short, op>, FMXImpl_AOAOA<int2, char2, char2, short2, op>, FMXImpl_AOAOA<int3, char3, char3, short3, op>, FMXImpl_AOAOA<int4, char4, char4, short4, op>  },
                    { FMXImpl_AOAOA<int, schar, schar, int, op>, FMXImpl_AOAOA<int2, char2, char2, int2, op>, FMXImpl_AOAOA<int3, char3, char3, int3, op>, FMXImpl_AOAOA<int4, char4, char4, int4, op>  },
                    { FMXImpl_AOAOA<int, schar, schar, float, op>, FMXImpl_AOAOA<int2, char2, char2, float2, op>, FMXImpl_AOAOA<int3, char3, char3, float3, op>, FMXImpl_AOAOA<int4, char4, char4, float4, op>  },
                    { FMXImpl_AOAOA<int, schar, schar, double, op>, FMXImpl_AOAOA<int2, char2, char2, double2, op>, FMXImpl_AOAOA<int3, char3, char3, double3, op>, FMXImpl_AOAOA<int4, char4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, schar, ushort, uchar, op>, FMXImpl_AOAOA<int2, char2, ushort2, uchar2, op>, FMXImpl_AOAOA<int3, char3, ushort3, uchar3, op>, FMXImpl_AOAOA<int4, char4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, schar, ushort, schar, op>, FMXImpl_AOAOA<int2, char2, ushort2, char2, op>, FMXImpl_AOAOA<int3, char3, ushort3, char3, op>, FMXImpl_AOAOA<int4, char4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<int, schar, ushort, ushort, op>, FMXImpl_AOAOA<int2, char2, ushort2, ushort2, op>, FMXImpl_AOAOA<int3, char3, ushort3, ushort3, op>, FMXImpl_AOAOA<int4, char4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, schar, ushort, short, op>, FMXImpl_AOAOA<int2, char2, ushort2, short2, op>, FMXImpl_AOAOA<int3, char3, ushort3, short3, op>, FMXImpl_AOAOA<int4, char4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<int, schar, ushort, int, op>, FMXImpl_AOAOA<int2, char2, ushort2, int2, op>, FMXImpl_AOAOA<int3, char3, ushort3, int3, op>, FMXImpl_AOAOA<int4, char4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<int, schar, ushort, float, op>, FMXImpl_AOAOA<int2, char2, ushort2, float2, op>, FMXImpl_AOAOA<int3, char3, ushort3, float3, op>, FMXImpl_AOAOA<int4, char4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<int, schar, ushort, double, op>, FMXImpl_AOAOA<int2, char2, ushort2, double2, op>, FMXImpl_AOAOA<int3, char3, ushort3, double3, op>, FMXImpl_AOAOA<int4, char4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, schar, short, uchar, op>, FMXImpl_AOAOA<int2, char2, short2, uchar2, op>, FMXImpl_AOAOA<int3, char3, short3, uchar3, op>, FMXImpl_AOAOA<int4, char4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, schar, short, schar, op>, FMXImpl_AOAOA<int2, char2, short2, char2, op>, FMXImpl_AOAOA<int3, char3, short3, char3, op>, FMXImpl_AOAOA<int4, char4, short4, char4, op>  },
                    { FMXImpl_AOAOA<int, schar, short, ushort, op>, FMXImpl_AOAOA<int2, char2, short2, ushort2, op>, FMXImpl_AOAOA<int3, char3, short3, ushort3, op>, FMXImpl_AOAOA<int4, char4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, schar, short, short, op>, FMXImpl_AOAOA<int2, char2, short2, short2, op>, FMXImpl_AOAOA<int3, char3, short3, short3, op>, FMXImpl_AOAOA<int4, char4, short4, short4, op>  },
                    { FMXImpl_AOAOA<int, schar, short, int, op>, FMXImpl_AOAOA<int2, char2, short2, int2, op>, FMXImpl_AOAOA<int3, char3, short3, int3, op>, FMXImpl_AOAOA<int4, char4, short4, int4, op>  },
                    { FMXImpl_AOAOA<int, schar, short, float, op>, FMXImpl_AOAOA<int2, char2, short2, float2, op>, FMXImpl_AOAOA<int3, char3, short3, float3, op>, FMXImpl_AOAOA<int4, char4, short4, float4, op>  },
                    { FMXImpl_AOAOA<int, schar, short, double, op>, FMXImpl_AOAOA<int2, char2, short2, double2, op>, FMXImpl_AOAOA<int3, char3, short3, double3, op>, FMXImpl_AOAOA<int4, char4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, schar, int, uchar, op>, FMXImpl_AOAOA<int2, char2, int2, uchar2, op>, FMXImpl_AOAOA<int3, char3, int3, uchar3, op>, FMXImpl_AOAOA<int4, char4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, schar, int, schar, op>, FMXImpl_AOAOA<int2, char2, int2, char2, op>, FMXImpl_AOAOA<int3, char3, int3, char3, op>, FMXImpl_AOAOA<int4, char4, int4, char4, op>  },
                    { FMXImpl_AOAOA<int, schar, int, ushort, op>, FMXImpl_AOAOA<int2, char2, int2, ushort2, op>, FMXImpl_AOAOA<int3, char3, int3, ushort3, op>, FMXImpl_AOAOA<int4, char4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, schar, int, short, op>, FMXImpl_AOAOA<int2, char2, int2, short2, op>, FMXImpl_AOAOA<int3, char3, int3, short3, op>, FMXImpl_AOAOA<int4, char4, int4, short4, op>  },
                    { FMXImpl_AOAOA<int, schar, int, int, op>, FMXImpl_AOAOA<int2, char2, int2, int2, op>, FMXImpl_AOAOA<int3, char3, int3, int3, op>, FMXImpl_AOAOA<int4, char4, int4, int4, op>  },
                    { FMXImpl_AOAOA<int, schar, int, float, op>, FMXImpl_AOAOA<int2, char2, int2, float2, op>, FMXImpl_AOAOA<int3, char3, int3, float3, op>, FMXImpl_AOAOA<int4, char4, int4, float4, op>  },
                    { FMXImpl_AOAOA<int, schar, int, double, op>, FMXImpl_AOAOA<int2, char2, int2, double2, op>, FMXImpl_AOAOA<int3, char3, int3, double3, op>, FMXImpl_AOAOA<int4, char4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, schar, float, uchar, op>, FMXImpl_AOAOA<int2, char2, float2, uchar2, op>, FMXImpl_AOAOA<int3, char3, float3, uchar3, op>, FMXImpl_AOAOA<int4, char4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, schar, float, schar, op>, FMXImpl_AOAOA<int2, char2, float2, char2, op>, FMXImpl_AOAOA<int3, char3, float3, char3, op>, FMXImpl_AOAOA<int4, char4, float4, char4, op>  },
                    { FMXImpl_AOAOA<int, schar, float, ushort, op>, FMXImpl_AOAOA<int2, char2, float2, ushort2, op>, FMXImpl_AOAOA<int3, char3, float3, ushort3, op>, FMXImpl_AOAOA<int4, char4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, schar, float, short, op>, FMXImpl_AOAOA<int2, char2, float2, short2, op>, FMXImpl_AOAOA<int3, char3, float3, short3, op>, FMXImpl_AOAOA<int4, char4, float4, short4, op>  },
                    { FMXImpl_AOAOA<int, schar, float, int, op>, FMXImpl_AOAOA<int2, char2, float2, int2, op>, FMXImpl_AOAOA<int3, char3, float3, int3, op>, FMXImpl_AOAOA<int4, char4, float4, int4, op>  },
                    { FMXImpl_AOAOA<int, schar, float, float, op>, FMXImpl_AOAOA<int2, char2, float2, float2, op>, FMXImpl_AOAOA<int3, char3, float3, float3, op>, FMXImpl_AOAOA<int4, char4, float4, float4, op>  },
                    { FMXImpl_AOAOA<int, schar, float, double, op>, FMXImpl_AOAOA<int2, char2, float2, double2, op>, FMXImpl_AOAOA<int3, char3, float3, double3, op>, FMXImpl_AOAOA<int4, char4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, schar, double, uchar, op>, FMXImpl_AOAOA<int2, char2, double2, uchar2, op>, FMXImpl_AOAOA<int3, char3, double3, uchar3, op>, FMXImpl_AOAOA<int4, char4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, schar, double, schar, op>, FMXImpl_AOAOA<int2, char2, double2, char2, op>, FMXImpl_AOAOA<int3, char3, double3, char3, op>, FMXImpl_AOAOA<int4, char4, double4, char4, op>  },
                    { FMXImpl_AOAOA<int, schar, double, ushort, op>, FMXImpl_AOAOA<int2, char2, double2, ushort2, op>, FMXImpl_AOAOA<int3, char3, double3, ushort3, op>, FMXImpl_AOAOA<int4, char4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, schar, double, short, op>, FMXImpl_AOAOA<int2, char2, double2, short2, op>, FMXImpl_AOAOA<int3, char3, double3, short3, op>, FMXImpl_AOAOA<int4, char4, double4, short4, op>  },
                    { FMXImpl_AOAOA<int, schar, double, int, op>, FMXImpl_AOAOA<int2, char2, double2, int2, op>, FMXImpl_AOAOA<int3, char3, double3, int3, op>, FMXImpl_AOAOA<int4, char4, double4, int4, op>  },
                    { FMXImpl_AOAOA<int, schar, double, float, op>, FMXImpl_AOAOA<int2, char2, double2, float2, op>, FMXImpl_AOAOA<int3, char3, double3, float3, op>, FMXImpl_AOAOA<int4, char4, double4, float4, op>  },
                    { FMXImpl_AOAOA<int, schar, double, double, op>, FMXImpl_AOAOA<int2, char2, double2, double2, op>, FMXImpl_AOAOA<int3, char3, double3, double3, op>, FMXImpl_AOAOA<int4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<int, ushort, uchar, uchar, op>, FMXImpl_AOAOA<int2, ushort2, uchar2, uchar2, op>, FMXImpl_AOAOA<int3, ushort3, uchar3, uchar3, op>, FMXImpl_AOAOA<int4, ushort4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, ushort, uchar, schar, op>, FMXImpl_AOAOA<int2, ushort2, uchar2, char2, op>, FMXImpl_AOAOA<int3, ushort3, uchar3, char3, op>, FMXImpl_AOAOA<int4, ushort4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<int, ushort, uchar, ushort, op>, FMXImpl_AOAOA<int2, ushort2, uchar2, ushort2, op>, FMXImpl_AOAOA<int3, ushort3, uchar3, ushort3, op>, FMXImpl_AOAOA<int4, ushort4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, ushort, uchar, short, op>, FMXImpl_AOAOA<int2, ushort2, uchar2, short2, op>, FMXImpl_AOAOA<int3, ushort3, uchar3, short3, op>, FMXImpl_AOAOA<int4, ushort4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<int, ushort, uchar, int, op>, FMXImpl_AOAOA<int2, ushort2, uchar2, int2, op>, FMXImpl_AOAOA<int3, ushort3, uchar3, int3, op>, FMXImpl_AOAOA<int4, ushort4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<int, ushort, uchar, float, op>, FMXImpl_AOAOA<int2, ushort2, uchar2, float2, op>, FMXImpl_AOAOA<int3, ushort3, uchar3, float3, op>, FMXImpl_AOAOA<int4, ushort4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<int, ushort, uchar, double, op>, FMXImpl_AOAOA<int2, ushort2, uchar2, double2, op>, FMXImpl_AOAOA<int3, ushort3, uchar3, double3, op>, FMXImpl_AOAOA<int4, ushort4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, ushort, schar, uchar, op>, FMXImpl_AOAOA<int2, ushort2, char2, uchar2, op>, FMXImpl_AOAOA<int3, ushort3, char3, uchar3, op>, FMXImpl_AOAOA<int4, ushort4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, ushort, schar, schar, op>, FMXImpl_AOAOA<int2, ushort2, char2, char2, op>, FMXImpl_AOAOA<int3, ushort3, char3, char3, op>, FMXImpl_AOAOA<int4, ushort4, char4, char4, op>  },
                    { FMXImpl_AOAOA<int, ushort, schar, ushort, op>, FMXImpl_AOAOA<int2, ushort2, char2, ushort2, op>, FMXImpl_AOAOA<int3, ushort3, char3, ushort3, op>, FMXImpl_AOAOA<int4, ushort4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, ushort, schar, short, op>, FMXImpl_AOAOA<int2, ushort2, char2, short2, op>, FMXImpl_AOAOA<int3, ushort3, char3, short3, op>, FMXImpl_AOAOA<int4, ushort4, char4, short4, op>  },
                    { FMXImpl_AOAOA<int, ushort, schar, int, op>, FMXImpl_AOAOA<int2, ushort2, char2, int2, op>, FMXImpl_AOAOA<int3, ushort3, char3, int3, op>, FMXImpl_AOAOA<int4, ushort4, char4, int4, op>  },
                    { FMXImpl_AOAOA<int, ushort, schar, float, op>, FMXImpl_AOAOA<int2, ushort2, char2, float2, op>, FMXImpl_AOAOA<int3, ushort3, char3, float3, op>, FMXImpl_AOAOA<int4, ushort4, char4, float4, op>  },
                    { FMXImpl_AOAOA<int, ushort, schar, double, op>, FMXImpl_AOAOA<int2, ushort2, char2, double2, op>, FMXImpl_AOAOA<int3, ushort3, char3, double3, op>, FMXImpl_AOAOA<int4, ushort4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, ushort, ushort, uchar, op>, FMXImpl_AOAOA<int2, ushort2, ushort2, uchar2, op>, FMXImpl_AOAOA<int3, ushort3, ushort3, uchar3, op>, FMXImpl_AOAOA<int4, ushort4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, ushort, ushort, schar, op>, FMXImpl_AOAOA<int2, ushort2, ushort2, char2, op>, FMXImpl_AOAOA<int3, ushort3, ushort3, char3, op>, FMXImpl_AOAOA<int4, ushort4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<int, ushort, ushort, ushort, op>, FMXImpl_AOAOA<int2, ushort2, ushort2, ushort2, op>, FMXImpl_AOAOA<int3, ushort3, ushort3, ushort3, op>, FMXImpl_AOAOA<int4, ushort4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, ushort, ushort, short, op>, FMXImpl_AOAOA<int2, ushort2, ushort2, short2, op>, FMXImpl_AOAOA<int3, ushort3, ushort3, short3, op>, FMXImpl_AOAOA<int4, ushort4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<int, ushort, ushort, int, op>, FMXImpl_AOAOA<int2, ushort2, ushort2, int2, op>, FMXImpl_AOAOA<int3, ushort3, ushort3, int3, op>, FMXImpl_AOAOA<int4, ushort4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<int, ushort, ushort, float, op>, FMXImpl_AOAOA<int2, ushort2, ushort2, float2, op>, FMXImpl_AOAOA<int3, ushort3, ushort3, float3, op>, FMXImpl_AOAOA<int4, ushort4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<int, ushort, ushort, double, op>, FMXImpl_AOAOA<int2, ushort2, ushort2, double2, op>, FMXImpl_AOAOA<int3, ushort3, ushort3, double3, op>, FMXImpl_AOAOA<int4, ushort4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, ushort, short, uchar, op>, FMXImpl_AOAOA<int2, ushort2, short2, uchar2, op>, FMXImpl_AOAOA<int3, ushort3, short3, uchar3, op>, FMXImpl_AOAOA<int4, ushort4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, ushort, short, schar, op>, FMXImpl_AOAOA<int2, ushort2, short2, char2, op>, FMXImpl_AOAOA<int3, ushort3, short3, char3, op>, FMXImpl_AOAOA<int4, ushort4, short4, char4, op>  },
                    { FMXImpl_AOAOA<int, ushort, short, ushort, op>, FMXImpl_AOAOA<int2, ushort2, short2, ushort2, op>, FMXImpl_AOAOA<int3, ushort3, short3, ushort3, op>, FMXImpl_AOAOA<int4, ushort4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, ushort, short, short, op>, FMXImpl_AOAOA<int2, ushort2, short2, short2, op>, FMXImpl_AOAOA<int3, ushort3, short3, short3, op>, FMXImpl_AOAOA<int4, ushort4, short4, short4, op>  },
                    { FMXImpl_AOAOA<int, ushort, short, int, op>, FMXImpl_AOAOA<int2, ushort2, short2, int2, op>, FMXImpl_AOAOA<int3, ushort3, short3, int3, op>, FMXImpl_AOAOA<int4, ushort4, short4, int4, op>  },
                    { FMXImpl_AOAOA<int, ushort, short, float, op>, FMXImpl_AOAOA<int2, ushort2, short2, float2, op>, FMXImpl_AOAOA<int3, ushort3, short3, float3, op>, FMXImpl_AOAOA<int4, ushort4, short4, float4, op>  },
                    { FMXImpl_AOAOA<int, ushort, short, double, op>, FMXImpl_AOAOA<int2, ushort2, short2, double2, op>, FMXImpl_AOAOA<int3, ushort3, short3, double3, op>, FMXImpl_AOAOA<int4, ushort4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, ushort, int, uchar, op>, FMXImpl_AOAOA<int2, ushort2, int2, uchar2, op>, FMXImpl_AOAOA<int3, ushort3, int3, uchar3, op>, FMXImpl_AOAOA<int4, ushort4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, ushort, int, schar, op>, FMXImpl_AOAOA<int2, ushort2, int2, char2, op>, FMXImpl_AOAOA<int3, ushort3, int3, char3, op>, FMXImpl_AOAOA<int4, ushort4, int4, char4, op>  },
                    { FMXImpl_AOAOA<int, ushort, int, ushort, op>, FMXImpl_AOAOA<int2, ushort2, int2, ushort2, op>, FMXImpl_AOAOA<int3, ushort3, int3, ushort3, op>, FMXImpl_AOAOA<int4, ushort4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, ushort, int, short, op>, FMXImpl_AOAOA<int2, ushort2, int2, short2, op>, FMXImpl_AOAOA<int3, ushort3, int3, short3, op>, FMXImpl_AOAOA<int4, ushort4, int4, short4, op>  },
                    { FMXImpl_AOAOA<int, ushort, int, int, op>, FMXImpl_AOAOA<int2, ushort2, int2, int2, op>, FMXImpl_AOAOA<int3, ushort3, int3, int3, op>, FMXImpl_AOAOA<int4, ushort4, int4, int4, op>  },
                    { FMXImpl_AOAOA<int, ushort, int, float, op>, FMXImpl_AOAOA<int2, ushort2, int2, float2, op>, FMXImpl_AOAOA<int3, ushort3, int3, float3, op>, FMXImpl_AOAOA<int4, ushort4, int4, float4, op>  },
                    { FMXImpl_AOAOA<int, ushort, int, double, op>, FMXImpl_AOAOA<int2, ushort2, int2, double2, op>, FMXImpl_AOAOA<int3, ushort3, int3, double3, op>, FMXImpl_AOAOA<int4, ushort4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, ushort, float, uchar, op>, FMXImpl_AOAOA<int2, ushort2, float2, uchar2, op>, FMXImpl_AOAOA<int3, ushort3, float3, uchar3, op>, FMXImpl_AOAOA<int4, ushort4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, ushort, float, schar, op>, FMXImpl_AOAOA<int2, ushort2, float2, char2, op>, FMXImpl_AOAOA<int3, ushort3, float3, char3, op>, FMXImpl_AOAOA<int4, ushort4, float4, char4, op>  },
                    { FMXImpl_AOAOA<int, ushort, float, ushort, op>, FMXImpl_AOAOA<int2, ushort2, float2, ushort2, op>, FMXImpl_AOAOA<int3, ushort3, float3, ushort3, op>, FMXImpl_AOAOA<int4, ushort4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, ushort, float, short, op>, FMXImpl_AOAOA<int2, ushort2, float2, short2, op>, FMXImpl_AOAOA<int3, ushort3, float3, short3, op>, FMXImpl_AOAOA<int4, ushort4, float4, short4, op>  },
                    { FMXImpl_AOAOA<int, ushort, float, int, op>, FMXImpl_AOAOA<int2, ushort2, float2, int2, op>, FMXImpl_AOAOA<int3, ushort3, float3, int3, op>, FMXImpl_AOAOA<int4, ushort4, float4, int4, op>  },
                    { FMXImpl_AOAOA<int, ushort, float, float, op>, FMXImpl_AOAOA<int2, ushort2, float2, float2, op>, FMXImpl_AOAOA<int3, ushort3, float3, float3, op>, FMXImpl_AOAOA<int4, ushort4, float4, float4, op>  },
                    { FMXImpl_AOAOA<int, ushort, float, double, op>, FMXImpl_AOAOA<int2, ushort2, float2, double2, op>, FMXImpl_AOAOA<int3, ushort3, float3, double3, op>, FMXImpl_AOAOA<int4, ushort4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, ushort, double, uchar, op>, FMXImpl_AOAOA<int2, ushort2, double2, uchar2, op>, FMXImpl_AOAOA<int3, ushort3, double3, uchar3, op>, FMXImpl_AOAOA<int4, ushort4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, ushort, double, schar, op>, FMXImpl_AOAOA<int2, ushort2, double2, char2, op>, FMXImpl_AOAOA<int3, ushort3, double3, char3, op>, FMXImpl_AOAOA<int4, ushort4, double4, char4, op>  },
                    { FMXImpl_AOAOA<int, ushort, double, ushort, op>, FMXImpl_AOAOA<int2, ushort2, double2, ushort2, op>, FMXImpl_AOAOA<int3, ushort3, double3, ushort3, op>, FMXImpl_AOAOA<int4, ushort4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, ushort, double, short, op>, FMXImpl_AOAOA<int2, ushort2, double2, short2, op>, FMXImpl_AOAOA<int3, ushort3, double3, short3, op>, FMXImpl_AOAOA<int4, ushort4, double4, short4, op>  },
                    { FMXImpl_AOAOA<int, ushort, double, int, op>, FMXImpl_AOAOA<int2, ushort2, double2, int2, op>, FMXImpl_AOAOA<int3, ushort3, double3, int3, op>, FMXImpl_AOAOA<int4, ushort4, double4, int4, op>  },
                    { FMXImpl_AOAOA<int, ushort, double, float, op>, FMXImpl_AOAOA<int2, ushort2, double2, float2, op>, FMXImpl_AOAOA<int3, ushort3, double3, float3, op>, FMXImpl_AOAOA<int4, ushort4, double4, float4, op>  },
                    { FMXImpl_AOAOA<int, ushort, double, double, op>, FMXImpl_AOAOA<int2, ushort2, double2, double2, op>, FMXImpl_AOAOA<int3, ushort3, double3, double3, op>, FMXImpl_AOAOA<int4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<int, short, uchar, uchar, op>, FMXImpl_AOAOA<int2, short2, uchar2, uchar2, op>, FMXImpl_AOAOA<int3, short3, uchar3, uchar3, op>, FMXImpl_AOAOA<int4, short4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, short, uchar, schar, op>, FMXImpl_AOAOA<int2, short2, uchar2, char2, op>, FMXImpl_AOAOA<int3, short3, uchar3, char3, op>, FMXImpl_AOAOA<int4, short4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<int, short, uchar, ushort, op>, FMXImpl_AOAOA<int2, short2, uchar2, ushort2, op>, FMXImpl_AOAOA<int3, short3, uchar3, ushort3, op>, FMXImpl_AOAOA<int4, short4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, short, uchar, short, op>, FMXImpl_AOAOA<int2, short2, uchar2, short2, op>, FMXImpl_AOAOA<int3, short3, uchar3, short3, op>, FMXImpl_AOAOA<int4, short4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<int, short, uchar, int, op>, FMXImpl_AOAOA<int2, short2, uchar2, int2, op>, FMXImpl_AOAOA<int3, short3, uchar3, int3, op>, FMXImpl_AOAOA<int4, short4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<int, short, uchar, float, op>, FMXImpl_AOAOA<int2, short2, uchar2, float2, op>, FMXImpl_AOAOA<int3, short3, uchar3, float3, op>, FMXImpl_AOAOA<int4, short4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<int, short, uchar, double, op>, FMXImpl_AOAOA<int2, short2, uchar2, double2, op>, FMXImpl_AOAOA<int3, short3, uchar3, double3, op>, FMXImpl_AOAOA<int4, short4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, short, schar, uchar, op>, FMXImpl_AOAOA<int2, short2, char2, uchar2, op>, FMXImpl_AOAOA<int3, short3, char3, uchar3, op>, FMXImpl_AOAOA<int4, short4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, short, schar, schar, op>, FMXImpl_AOAOA<int2, short2, char2, char2, op>, FMXImpl_AOAOA<int3, short3, char3, char3, op>, FMXImpl_AOAOA<int4, short4, char4, char4, op>  },
                    { FMXImpl_AOAOA<int, short, schar, ushort, op>, FMXImpl_AOAOA<int2, short2, char2, ushort2, op>, FMXImpl_AOAOA<int3, short3, char3, ushort3, op>, FMXImpl_AOAOA<int4, short4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, short, schar, short, op>, FMXImpl_AOAOA<int2, short2, char2, short2, op>, FMXImpl_AOAOA<int3, short3, char3, short3, op>, FMXImpl_AOAOA<int4, short4, char4, short4, op>  },
                    { FMXImpl_AOAOA<int, short, schar, int, op>, FMXImpl_AOAOA<int2, short2, char2, int2, op>, FMXImpl_AOAOA<int3, short3, char3, int3, op>, FMXImpl_AOAOA<int4, short4, char4, int4, op>  },
                    { FMXImpl_AOAOA<int, short, schar, float, op>, FMXImpl_AOAOA<int2, short2, char2, float2, op>, FMXImpl_AOAOA<int3, short3, char3, float3, op>, FMXImpl_AOAOA<int4, short4, char4, float4, op>  },
                    { FMXImpl_AOAOA<int, short, schar, double, op>, FMXImpl_AOAOA<int2, short2, char2, double2, op>, FMXImpl_AOAOA<int3, short3, char3, double3, op>, FMXImpl_AOAOA<int4, short4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, short, ushort, uchar, op>, FMXImpl_AOAOA<int2, short2, ushort2, uchar2, op>, FMXImpl_AOAOA<int3, short3, ushort3, uchar3, op>, FMXImpl_AOAOA<int4, short4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, short, ushort, schar, op>, FMXImpl_AOAOA<int2, short2, ushort2, char2, op>, FMXImpl_AOAOA<int3, short3, ushort3, char3, op>, FMXImpl_AOAOA<int4, short4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<int, short, ushort, ushort, op>, FMXImpl_AOAOA<int2, short2, ushort2, ushort2, op>, FMXImpl_AOAOA<int3, short3, ushort3, ushort3, op>, FMXImpl_AOAOA<int4, short4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, short, ushort, short, op>, FMXImpl_AOAOA<int2, short2, ushort2, short2, op>, FMXImpl_AOAOA<int3, short3, ushort3, short3, op>, FMXImpl_AOAOA<int4, short4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<int, short, ushort, int, op>, FMXImpl_AOAOA<int2, short2, ushort2, int2, op>, FMXImpl_AOAOA<int3, short3, ushort3, int3, op>, FMXImpl_AOAOA<int4, short4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<int, short, ushort, float, op>, FMXImpl_AOAOA<int2, short2, ushort2, float2, op>, FMXImpl_AOAOA<int3, short3, ushort3, float3, op>, FMXImpl_AOAOA<int4, short4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<int, short, ushort, double, op>, FMXImpl_AOAOA<int2, short2, ushort2, double2, op>, FMXImpl_AOAOA<int3, short3, ushort3, double3, op>, FMXImpl_AOAOA<int4, short4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, short, short, uchar, op>, FMXImpl_AOAOA<int2, short2, short2, uchar2, op>, FMXImpl_AOAOA<int3, short3, short3, uchar3, op>, FMXImpl_AOAOA<int4, short4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, short, short, schar, op>, FMXImpl_AOAOA<int2, short2, short2, char2, op>, FMXImpl_AOAOA<int3, short3, short3, char3, op>, FMXImpl_AOAOA<int4, short4, short4, char4, op>  },
                    { FMXImpl_AOAOA<int, short, short, ushort, op>, FMXImpl_AOAOA<int2, short2, short2, ushort2, op>, FMXImpl_AOAOA<int3, short3, short3, ushort3, op>, FMXImpl_AOAOA<int4, short4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, short, short, short, op>, FMXImpl_AOAOA<int2, short2, short2, short2, op>, FMXImpl_AOAOA<int3, short3, short3, short3, op>, FMXImpl_AOAOA<int4, short4, short4, short4, op>  },
                    { FMXImpl_AOAOA<int, short, short, int, op>, FMXImpl_AOAOA<int2, short2, short2, int2, op>, FMXImpl_AOAOA<int3, short3, short3, int3, op>, FMXImpl_AOAOA<int4, short4, short4, int4, op>  },
                    { FMXImpl_AOAOA<int, short, short, float, op>, FMXImpl_AOAOA<int2, short2, short2, float2, op>, FMXImpl_AOAOA<int3, short3, short3, float3, op>, FMXImpl_AOAOA<int4, short4, short4, float4, op>  },
                    { FMXImpl_AOAOA<int, short, short, double, op>, FMXImpl_AOAOA<int2, short2, short2, double2, op>, FMXImpl_AOAOA<int3, short3, short3, double3, op>, FMXImpl_AOAOA<int4, short4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, short, int, uchar, op>, FMXImpl_AOAOA<int2, short2, int2, uchar2, op>, FMXImpl_AOAOA<int3, short3, int3, uchar3, op>, FMXImpl_AOAOA<int4, short4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, short, int, schar, op>, FMXImpl_AOAOA<int2, short2, int2, char2, op>, FMXImpl_AOAOA<int3, short3, int3, char3, op>, FMXImpl_AOAOA<int4, short4, int4, char4, op>  },
                    { FMXImpl_AOAOA<int, short, int, ushort, op>, FMXImpl_AOAOA<int2, short2, int2, ushort2, op>, FMXImpl_AOAOA<int3, short3, int3, ushort3, op>, FMXImpl_AOAOA<int4, short4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, short, int, short, op>, FMXImpl_AOAOA<int2, short2, int2, short2, op>, FMXImpl_AOAOA<int3, short3, int3, short3, op>, FMXImpl_AOAOA<int4, short4, int4, short4, op>  },
                    { FMXImpl_AOAOA<int, short, int, int, op>, FMXImpl_AOAOA<int2, short2, int2, int2, op>, FMXImpl_AOAOA<int3, short3, int3, int3, op>, FMXImpl_AOAOA<int4, short4, int4, int4, op>  },
                    { FMXImpl_AOAOA<int, short, int, float, op>, FMXImpl_AOAOA<int2, short2, int2, float2, op>, FMXImpl_AOAOA<int3, short3, int3, float3, op>, FMXImpl_AOAOA<int4, short4, int4, float4, op>  },
                    { FMXImpl_AOAOA<int, short, int, double, op>, FMXImpl_AOAOA<int2, short2, int2, double2, op>, FMXImpl_AOAOA<int3, short3, int3, double3, op>, FMXImpl_AOAOA<int4, short4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, short, float, uchar, op>, FMXImpl_AOAOA<int2, short2, float2, uchar2, op>, FMXImpl_AOAOA<int3, short3, float3, uchar3, op>, FMXImpl_AOAOA<int4, short4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, short, float, schar, op>, FMXImpl_AOAOA<int2, short2, float2, char2, op>, FMXImpl_AOAOA<int3, short3, float3, char3, op>, FMXImpl_AOAOA<int4, short4, float4, char4, op>  },
                    { FMXImpl_AOAOA<int, short, float, ushort, op>, FMXImpl_AOAOA<int2, short2, float2, ushort2, op>, FMXImpl_AOAOA<int3, short3, float3, ushort3, op>, FMXImpl_AOAOA<int4, short4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, short, float, short, op>, FMXImpl_AOAOA<int2, short2, float2, short2, op>, FMXImpl_AOAOA<int3, short3, float3, short3, op>, FMXImpl_AOAOA<int4, short4, float4, short4, op>  },
                    { FMXImpl_AOAOA<int, short, float, int, op>, FMXImpl_AOAOA<int2, short2, float2, int2, op>, FMXImpl_AOAOA<int3, short3, float3, int3, op>, FMXImpl_AOAOA<int4, short4, float4, int4, op>  },
                    { FMXImpl_AOAOA<int, short, float, float, op>, FMXImpl_AOAOA<int2, short2, float2, float2, op>, FMXImpl_AOAOA<int3, short3, float3, float3, op>, FMXImpl_AOAOA<int4, short4, float4, float4, op>  },
                    { FMXImpl_AOAOA<int, short, float, double, op>, FMXImpl_AOAOA<int2, short2, float2, double2, op>, FMXImpl_AOAOA<int3, short3, float3, double3, op>, FMXImpl_AOAOA<int4, short4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, short, double, uchar, op>, FMXImpl_AOAOA<int2, short2, double2, uchar2, op>, FMXImpl_AOAOA<int3, short3, double3, uchar3, op>, FMXImpl_AOAOA<int4, short4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, short, double, schar, op>, FMXImpl_AOAOA<int2, short2, double2, char2, op>, FMXImpl_AOAOA<int3, short3, double3, char3, op>, FMXImpl_AOAOA<int4, short4, double4, char4, op>  },
                    { FMXImpl_AOAOA<int, short, double, ushort, op>, FMXImpl_AOAOA<int2, short2, double2, ushort2, op>, FMXImpl_AOAOA<int3, short3, double3, ushort3, op>, FMXImpl_AOAOA<int4, short4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, short, double, short, op>, FMXImpl_AOAOA<int2, short2, double2, short2, op>, FMXImpl_AOAOA<int3, short3, double3, short3, op>, FMXImpl_AOAOA<int4, short4, double4, short4, op>  },
                    { FMXImpl_AOAOA<int, short, double, int, op>, FMXImpl_AOAOA<int2, short2, double2, int2, op>, FMXImpl_AOAOA<int3, short3, double3, int3, op>, FMXImpl_AOAOA<int4, short4, double4, int4, op>  },
                    { FMXImpl_AOAOA<int, short, double, float, op>, FMXImpl_AOAOA<int2, short2, double2, float2, op>, FMXImpl_AOAOA<int3, short3, double3, float3, op>, FMXImpl_AOAOA<int4, short4, double4, float4, op>  },
                    { FMXImpl_AOAOA<int, short, double, double, op>, FMXImpl_AOAOA<int2, short2, double2, double2, op>, FMXImpl_AOAOA<int3, short3, double3, double3, op>, FMXImpl_AOAOA<int4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<int, int, uchar, uchar, op>, FMXImpl_AOAOA<int2, int2, uchar2, uchar2, op>, FMXImpl_AOAOA<int3, int3, uchar3, uchar3, op>, FMXImpl_AOAOA<int4, int4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, int, uchar, schar, op>, FMXImpl_AOAOA<int2, int2, uchar2, char2, op>, FMXImpl_AOAOA<int3, int3, uchar3, char3, op>, FMXImpl_AOAOA<int4, int4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<int, int, uchar, ushort, op>, FMXImpl_AOAOA<int2, int2, uchar2, ushort2, op>, FMXImpl_AOAOA<int3, int3, uchar3, ushort3, op>, FMXImpl_AOAOA<int4, int4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, int, uchar, short, op>, FMXImpl_AOAOA<int2, int2, uchar2, short2, op>, FMXImpl_AOAOA<int3, int3, uchar3, short3, op>, FMXImpl_AOAOA<int4, int4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<int, int, uchar, int, op>, FMXImpl_AOAOA<int2, int2, uchar2, int2, op>, FMXImpl_AOAOA<int3, int3, uchar3, int3, op>, FMXImpl_AOAOA<int4, int4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<int, int, uchar, float, op>, FMXImpl_AOAOA<int2, int2, uchar2, float2, op>, FMXImpl_AOAOA<int3, int3, uchar3, float3, op>, FMXImpl_AOAOA<int4, int4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<int, int, uchar, double, op>, FMXImpl_AOAOA<int2, int2, uchar2, double2, op>, FMXImpl_AOAOA<int3, int3, uchar3, double3, op>, FMXImpl_AOAOA<int4, int4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, int, schar, uchar, op>, FMXImpl_AOAOA<int2, int2, char2, uchar2, op>, FMXImpl_AOAOA<int3, int3, char3, uchar3, op>, FMXImpl_AOAOA<int4, int4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, int, schar, schar, op>, FMXImpl_AOAOA<int2, int2, char2, char2, op>, FMXImpl_AOAOA<int3, int3, char3, char3, op>, FMXImpl_AOAOA<int4, int4, char4, char4, op>  },
                    { FMXImpl_AOAOA<int, int, schar, ushort, op>, FMXImpl_AOAOA<int2, int2, char2, ushort2, op>, FMXImpl_AOAOA<int3, int3, char3, ushort3, op>, FMXImpl_AOAOA<int4, int4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, int, schar, short, op>, FMXImpl_AOAOA<int2, int2, char2, short2, op>, FMXImpl_AOAOA<int3, int3, char3, short3, op>, FMXImpl_AOAOA<int4, int4, char4, short4, op>  },
                    { FMXImpl_AOAOA<int, int, schar, int, op>, FMXImpl_AOAOA<int2, int2, char2, int2, op>, FMXImpl_AOAOA<int3, int3, char3, int3, op>, FMXImpl_AOAOA<int4, int4, char4, int4, op>  },
                    { FMXImpl_AOAOA<int, int, schar, float, op>, FMXImpl_AOAOA<int2, int2, char2, float2, op>, FMXImpl_AOAOA<int3, int3, char3, float3, op>, FMXImpl_AOAOA<int4, int4, char4, float4, op>  },
                    { FMXImpl_AOAOA<int, int, schar, double, op>, FMXImpl_AOAOA<int2, int2, char2, double2, op>, FMXImpl_AOAOA<int3, int3, char3, double3, op>, FMXImpl_AOAOA<int4, int4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, int, ushort, uchar, op>, FMXImpl_AOAOA<int2, int2, ushort2, uchar2, op>, FMXImpl_AOAOA<int3, int3, ushort3, uchar3, op>, FMXImpl_AOAOA<int4, int4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, int, ushort, schar, op>, FMXImpl_AOAOA<int2, int2, ushort2, char2, op>, FMXImpl_AOAOA<int3, int3, ushort3, char3, op>, FMXImpl_AOAOA<int4, int4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<int, int, ushort, ushort, op>, FMXImpl_AOAOA<int2, int2, ushort2, ushort2, op>, FMXImpl_AOAOA<int3, int3, ushort3, ushort3, op>, FMXImpl_AOAOA<int4, int4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, int, ushort, short, op>, FMXImpl_AOAOA<int2, int2, ushort2, short2, op>, FMXImpl_AOAOA<int3, int3, ushort3, short3, op>, FMXImpl_AOAOA<int4, int4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<int, int, ushort, int, op>, FMXImpl_AOAOA<int2, int2, ushort2, int2, op>, FMXImpl_AOAOA<int3, int3, ushort3, int3, op>, FMXImpl_AOAOA<int4, int4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<int, int, ushort, float, op>, FMXImpl_AOAOA<int2, int2, ushort2, float2, op>, FMXImpl_AOAOA<int3, int3, ushort3, float3, op>, FMXImpl_AOAOA<int4, int4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<int, int, ushort, double, op>, FMXImpl_AOAOA<int2, int2, ushort2, double2, op>, FMXImpl_AOAOA<int3, int3, ushort3, double3, op>, FMXImpl_AOAOA<int4, int4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, int, short, uchar, op>, FMXImpl_AOAOA<int2, int2, short2, uchar2, op>, FMXImpl_AOAOA<int3, int3, short3, uchar3, op>, FMXImpl_AOAOA<int4, int4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, int, short, schar, op>, FMXImpl_AOAOA<int2, int2, short2, char2, op>, FMXImpl_AOAOA<int3, int3, short3, char3, op>, FMXImpl_AOAOA<int4, int4, short4, char4, op>  },
                    { FMXImpl_AOAOA<int, int, short, ushort, op>, FMXImpl_AOAOA<int2, int2, short2, ushort2, op>, FMXImpl_AOAOA<int3, int3, short3, ushort3, op>, FMXImpl_AOAOA<int4, int4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, int, short, short, op>, FMXImpl_AOAOA<int2, int2, short2, short2, op>, FMXImpl_AOAOA<int3, int3, short3, short3, op>, FMXImpl_AOAOA<int4, int4, short4, short4, op>  },
                    { FMXImpl_AOAOA<int, int, short, int, op>, FMXImpl_AOAOA<int2, int2, short2, int2, op>, FMXImpl_AOAOA<int3, int3, short3, int3, op>, FMXImpl_AOAOA<int4, int4, short4, int4, op>  },
                    { FMXImpl_AOAOA<int, int, short, float, op>, FMXImpl_AOAOA<int2, int2, short2, float2, op>, FMXImpl_AOAOA<int3, int3, short3, float3, op>, FMXImpl_AOAOA<int4, int4, short4, float4, op>  },
                    { FMXImpl_AOAOA<int, int, short, double, op>, FMXImpl_AOAOA<int2, int2, short2, double2, op>, FMXImpl_AOAOA<int3, int3, short3, double3, op>, FMXImpl_AOAOA<int4, int4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, int, int, uchar, op>, FMXImpl_AOAOA<int2, int2, int2, uchar2, op>, FMXImpl_AOAOA<int3, int3, int3, uchar3, op>, FMXImpl_AOAOA<int4, int4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, int, int, schar, op>, FMXImpl_AOAOA<int2, int2, int2, char2, op>, FMXImpl_AOAOA<int3, int3, int3, char3, op>, FMXImpl_AOAOA<int4, int4, int4, char4, op>  },
                    { FMXImpl_AOAOA<int, int, int, ushort, op>, FMXImpl_AOAOA<int2, int2, int2, ushort2, op>, FMXImpl_AOAOA<int3, int3, int3, ushort3, op>, FMXImpl_AOAOA<int4, int4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, int, int, short, op>, FMXImpl_AOAOA<int2, int2, int2, short2, op>, FMXImpl_AOAOA<int3, int3, int3, short3, op>, FMXImpl_AOAOA<int4, int4, int4, short4, op>  },
                    { FMXImpl_AOAOA<int, int, int, int, op>, FMXImpl_AOAOA<int2, int2, int2, int2, op>, FMXImpl_AOAOA<int3, int3, int3, int3, op>, FMXImpl_AOAOA<int4, int4, int4, int4, op>  },
                    { FMXImpl_AOAOA<int, int, int, float, op>, FMXImpl_AOAOA<int2, int2, int2, float2, op>, FMXImpl_AOAOA<int3, int3, int3, float3, op>, FMXImpl_AOAOA<int4, int4, int4, float4, op>  },
                    { FMXImpl_AOAOA<int, int, int, double, op>, FMXImpl_AOAOA<int2, int2, int2, double2, op>, FMXImpl_AOAOA<int3, int3, int3, double3, op>, FMXImpl_AOAOA<int4, int4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, int, float, uchar, op>, FMXImpl_AOAOA<int2, int2, float2, uchar2, op>, FMXImpl_AOAOA<int3, int3, float3, uchar3, op>, FMXImpl_AOAOA<int4, int4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, int, float, schar, op>, FMXImpl_AOAOA<int2, int2, float2, char2, op>, FMXImpl_AOAOA<int3, int3, float3, char3, op>, FMXImpl_AOAOA<int4, int4, float4, char4, op>  },
                    { FMXImpl_AOAOA<int, int, float, ushort, op>, FMXImpl_AOAOA<int2, int2, float2, ushort2, op>, FMXImpl_AOAOA<int3, int3, float3, ushort3, op>, FMXImpl_AOAOA<int4, int4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, int, float, short, op>, FMXImpl_AOAOA<int2, int2, float2, short2, op>, FMXImpl_AOAOA<int3, int3, float3, short3, op>, FMXImpl_AOAOA<int4, int4, float4, short4, op>  },
                    { FMXImpl_AOAOA<int, int, float, int, op>, FMXImpl_AOAOA<int2, int2, float2, int2, op>, FMXImpl_AOAOA<int3, int3, float3, int3, op>, FMXImpl_AOAOA<int4, int4, float4, int4, op>  },
                    { FMXImpl_AOAOA<int, int, float, float, op>, FMXImpl_AOAOA<int2, int2, float2, float2, op>, FMXImpl_AOAOA<int3, int3, float3, float3, op>, FMXImpl_AOAOA<int4, int4, float4, float4, op>  },
                    { FMXImpl_AOAOA<int, int, float, double, op>, FMXImpl_AOAOA<int2, int2, float2, double2, op>, FMXImpl_AOAOA<int3, int3, float3, double3, op>, FMXImpl_AOAOA<int4, int4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, int, double, uchar, op>, FMXImpl_AOAOA<int2, int2, double2, uchar2, op>, FMXImpl_AOAOA<int3, int3, double3, uchar3, op>, FMXImpl_AOAOA<int4, int4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, int, double, schar, op>, FMXImpl_AOAOA<int2, int2, double2, char2, op>, FMXImpl_AOAOA<int3, int3, double3, char3, op>, FMXImpl_AOAOA<int4, int4, double4, char4, op>  },
                    { FMXImpl_AOAOA<int, int, double, ushort, op>, FMXImpl_AOAOA<int2, int2, double2, ushort2, op>, FMXImpl_AOAOA<int3, int3, double3, ushort3, op>, FMXImpl_AOAOA<int4, int4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, int, double, short, op>, FMXImpl_AOAOA<int2, int2, double2, short2, op>, FMXImpl_AOAOA<int3, int3, double3, short3, op>, FMXImpl_AOAOA<int4, int4, double4, short4, op>  },
                    { FMXImpl_AOAOA<int, int, double, int, op>, FMXImpl_AOAOA<int2, int2, double2, int2, op>, FMXImpl_AOAOA<int3, int3, double3, int3, op>, FMXImpl_AOAOA<int4, int4, double4, int4, op>  },
                    { FMXImpl_AOAOA<int, int, double, float, op>, FMXImpl_AOAOA<int2, int2, double2, float2, op>, FMXImpl_AOAOA<int3, int3, double3, float3, op>, FMXImpl_AOAOA<int4, int4, double4, float4, op>  },
                    { FMXImpl_AOAOA<int, int, double, double, op>, FMXImpl_AOAOA<int2, int2, double2, double2, op>, FMXImpl_AOAOA<int3, int3, double3, double3, op>, FMXImpl_AOAOA<int4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<int, float, uchar, uchar, op>, FMXImpl_AOAOA<int2, float2, uchar2, uchar2, op>, FMXImpl_AOAOA<int3, float3, uchar3, uchar3, op>, FMXImpl_AOAOA<int4, float4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, float, uchar, schar, op>, FMXImpl_AOAOA<int2, float2, uchar2, char2, op>, FMXImpl_AOAOA<int3, float3, uchar3, char3, op>, FMXImpl_AOAOA<int4, float4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<int, float, uchar, ushort, op>, FMXImpl_AOAOA<int2, float2, uchar2, ushort2, op>, FMXImpl_AOAOA<int3, float3, uchar3, ushort3, op>, FMXImpl_AOAOA<int4, float4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, float, uchar, short, op>, FMXImpl_AOAOA<int2, float2, uchar2, short2, op>, FMXImpl_AOAOA<int3, float3, uchar3, short3, op>, FMXImpl_AOAOA<int4, float4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<int, float, uchar, int, op>, FMXImpl_AOAOA<int2, float2, uchar2, int2, op>, FMXImpl_AOAOA<int3, float3, uchar3, int3, op>, FMXImpl_AOAOA<int4, float4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<int, float, uchar, float, op>, FMXImpl_AOAOA<int2, float2, uchar2, float2, op>, FMXImpl_AOAOA<int3, float3, uchar3, float3, op>, FMXImpl_AOAOA<int4, float4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<int, float, uchar, double, op>, FMXImpl_AOAOA<int2, float2, uchar2, double2, op>, FMXImpl_AOAOA<int3, float3, uchar3, double3, op>, FMXImpl_AOAOA<int4, float4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, float, schar, uchar, op>, FMXImpl_AOAOA<int2, float2, char2, uchar2, op>, FMXImpl_AOAOA<int3, float3, char3, uchar3, op>, FMXImpl_AOAOA<int4, float4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, float, schar, schar, op>, FMXImpl_AOAOA<int2, float2, char2, char2, op>, FMXImpl_AOAOA<int3, float3, char3, char3, op>, FMXImpl_AOAOA<int4, float4, char4, char4, op>  },
                    { FMXImpl_AOAOA<int, float, schar, ushort, op>, FMXImpl_AOAOA<int2, float2, char2, ushort2, op>, FMXImpl_AOAOA<int3, float3, char3, ushort3, op>, FMXImpl_AOAOA<int4, float4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, float, schar, short, op>, FMXImpl_AOAOA<int2, float2, char2, short2, op>, FMXImpl_AOAOA<int3, float3, char3, short3, op>, FMXImpl_AOAOA<int4, float4, char4, short4, op>  },
                    { FMXImpl_AOAOA<int, float, schar, int, op>, FMXImpl_AOAOA<int2, float2, char2, int2, op>, FMXImpl_AOAOA<int3, float3, char3, int3, op>, FMXImpl_AOAOA<int4, float4, char4, int4, op>  },
                    { FMXImpl_AOAOA<int, float, schar, float, op>, FMXImpl_AOAOA<int2, float2, char2, float2, op>, FMXImpl_AOAOA<int3, float3, char3, float3, op>, FMXImpl_AOAOA<int4, float4, char4, float4, op>  },
                    { FMXImpl_AOAOA<int, float, schar, double, op>, FMXImpl_AOAOA<int2, float2, char2, double2, op>, FMXImpl_AOAOA<int3, float3, char3, double3, op>, FMXImpl_AOAOA<int4, float4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, float, ushort, uchar, op>, FMXImpl_AOAOA<int2, float2, ushort2, uchar2, op>, FMXImpl_AOAOA<int3, float3, ushort3, uchar3, op>, FMXImpl_AOAOA<int4, float4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, float, ushort, schar, op>, FMXImpl_AOAOA<int2, float2, ushort2, char2, op>, FMXImpl_AOAOA<int3, float3, ushort3, char3, op>, FMXImpl_AOAOA<int4, float4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<int, float, ushort, ushort, op>, FMXImpl_AOAOA<int2, float2, ushort2, ushort2, op>, FMXImpl_AOAOA<int3, float3, ushort3, ushort3, op>, FMXImpl_AOAOA<int4, float4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, float, ushort, short, op>, FMXImpl_AOAOA<int2, float2, ushort2, short2, op>, FMXImpl_AOAOA<int3, float3, ushort3, short3, op>, FMXImpl_AOAOA<int4, float4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<int, float, ushort, int, op>, FMXImpl_AOAOA<int2, float2, ushort2, int2, op>, FMXImpl_AOAOA<int3, float3, ushort3, int3, op>, FMXImpl_AOAOA<int4, float4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<int, float, ushort, float, op>, FMXImpl_AOAOA<int2, float2, ushort2, float2, op>, FMXImpl_AOAOA<int3, float3, ushort3, float3, op>, FMXImpl_AOAOA<int4, float4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<int, float, ushort, double, op>, FMXImpl_AOAOA<int2, float2, ushort2, double2, op>, FMXImpl_AOAOA<int3, float3, ushort3, double3, op>, FMXImpl_AOAOA<int4, float4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, float, short, uchar, op>, FMXImpl_AOAOA<int2, float2, short2, uchar2, op>, FMXImpl_AOAOA<int3, float3, short3, uchar3, op>, FMXImpl_AOAOA<int4, float4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, float, short, schar, op>, FMXImpl_AOAOA<int2, float2, short2, char2, op>, FMXImpl_AOAOA<int3, float3, short3, char3, op>, FMXImpl_AOAOA<int4, float4, short4, char4, op>  },
                    { FMXImpl_AOAOA<int, float, short, ushort, op>, FMXImpl_AOAOA<int2, float2, short2, ushort2, op>, FMXImpl_AOAOA<int3, float3, short3, ushort3, op>, FMXImpl_AOAOA<int4, float4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, float, short, short, op>, FMXImpl_AOAOA<int2, float2, short2, short2, op>, FMXImpl_AOAOA<int3, float3, short3, short3, op>, FMXImpl_AOAOA<int4, float4, short4, short4, op>  },
                    { FMXImpl_AOAOA<int, float, short, int, op>, FMXImpl_AOAOA<int2, float2, short2, int2, op>, FMXImpl_AOAOA<int3, float3, short3, int3, op>, FMXImpl_AOAOA<int4, float4, short4, int4, op>  },
                    { FMXImpl_AOAOA<int, float, short, float, op>, FMXImpl_AOAOA<int2, float2, short2, float2, op>, FMXImpl_AOAOA<int3, float3, short3, float3, op>, FMXImpl_AOAOA<int4, float4, short4, float4, op>  },
                    { FMXImpl_AOAOA<int, float, short, double, op>, FMXImpl_AOAOA<int2, float2, short2, double2, op>, FMXImpl_AOAOA<int3, float3, short3, double3, op>, FMXImpl_AOAOA<int4, float4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, float, int, uchar, op>, FMXImpl_AOAOA<int2, float2, int2, uchar2, op>, FMXImpl_AOAOA<int3, float3, int3, uchar3, op>, FMXImpl_AOAOA<int4, float4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, float, int, schar, op>, FMXImpl_AOAOA<int2, float2, int2, char2, op>, FMXImpl_AOAOA<int3, float3, int3, char3, op>, FMXImpl_AOAOA<int4, float4, int4, char4, op>  },
                    { FMXImpl_AOAOA<int, float, int, ushort, op>, FMXImpl_AOAOA<int2, float2, int2, ushort2, op>, FMXImpl_AOAOA<int3, float3, int3, ushort3, op>, FMXImpl_AOAOA<int4, float4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, float, int, short, op>, FMXImpl_AOAOA<int2, float2, int2, short2, op>, FMXImpl_AOAOA<int3, float3, int3, short3, op>, FMXImpl_AOAOA<int4, float4, int4, short4, op>  },
                    { FMXImpl_AOAOA<int, float, int, int, op>, FMXImpl_AOAOA<int2, float2, int2, int2, op>, FMXImpl_AOAOA<int3, float3, int3, int3, op>, FMXImpl_AOAOA<int4, float4, int4, int4, op>  },
                    { FMXImpl_AOAOA<int, float, int, float, op>, FMXImpl_AOAOA<int2, float2, int2, float2, op>, FMXImpl_AOAOA<int3, float3, int3, float3, op>, FMXImpl_AOAOA<int4, float4, int4, float4, op>  },
                    { FMXImpl_AOAOA<int, float, int, double, op>, FMXImpl_AOAOA<int2, float2, int2, double2, op>, FMXImpl_AOAOA<int3, float3, int3, double3, op>, FMXImpl_AOAOA<int4, float4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, float, float, uchar, op>, FMXImpl_AOAOA<int2, float2, float2, uchar2, op>, FMXImpl_AOAOA<int3, float3, float3, uchar3, op>, FMXImpl_AOAOA<int4, float4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, float, float, schar, op>, FMXImpl_AOAOA<int2, float2, float2, char2, op>, FMXImpl_AOAOA<int3, float3, float3, char3, op>, FMXImpl_AOAOA<int4, float4, float4, char4, op>  },
                    { FMXImpl_AOAOA<int, float, float, ushort, op>, FMXImpl_AOAOA<int2, float2, float2, ushort2, op>, FMXImpl_AOAOA<int3, float3, float3, ushort3, op>, FMXImpl_AOAOA<int4, float4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, float, float, short, op>, FMXImpl_AOAOA<int2, float2, float2, short2, op>, FMXImpl_AOAOA<int3, float3, float3, short3, op>, FMXImpl_AOAOA<int4, float4, float4, short4, op>  },
                    { FMXImpl_AOAOA<int, float, float, int, op>, FMXImpl_AOAOA<int2, float2, float2, int2, op>, FMXImpl_AOAOA<int3, float3, float3, int3, op>, FMXImpl_AOAOA<int4, float4, float4, int4, op>  },
                    { FMXImpl_AOAOA<int, float, float, float, op>, FMXImpl_AOAOA<int2, float2, float2, float2, op>, FMXImpl_AOAOA<int3, float3, float3, float3, op>, FMXImpl_AOAOA<int4, float4, float4, float4, op>  },
                    { FMXImpl_AOAOA<int, float, float, double, op>, FMXImpl_AOAOA<int2, float2, float2, double2, op>, FMXImpl_AOAOA<int3, float3, float3, double3, op>, FMXImpl_AOAOA<int4, float4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, float, double, uchar, op>, FMXImpl_AOAOA<int2, float2, double2, uchar2, op>, FMXImpl_AOAOA<int3, float3, double3, uchar3, op>, FMXImpl_AOAOA<int4, float4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, float, double, schar, op>, FMXImpl_AOAOA<int2, float2, double2, char2, op>, FMXImpl_AOAOA<int3, float3, double3, char3, op>, FMXImpl_AOAOA<int4, float4, double4, char4, op>  },
                    { FMXImpl_AOAOA<int, float, double, ushort, op>, FMXImpl_AOAOA<int2, float2, double2, ushort2, op>, FMXImpl_AOAOA<int3, float3, double3, ushort3, op>, FMXImpl_AOAOA<int4, float4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, float, double, short, op>, FMXImpl_AOAOA<int2, float2, double2, short2, op>, FMXImpl_AOAOA<int3, float3, double3, short3, op>, FMXImpl_AOAOA<int4, float4, double4, short4, op>  },
                    { FMXImpl_AOAOA<int, float, double, int, op>, FMXImpl_AOAOA<int2, float2, double2, int2, op>, FMXImpl_AOAOA<int3, float3, double3, int3, op>, FMXImpl_AOAOA<int4, float4, double4, int4, op>  },
                    { FMXImpl_AOAOA<int, float, double, float, op>, FMXImpl_AOAOA<int2, float2, double2, float2, op>, FMXImpl_AOAOA<int3, float3, double3, float3, op>, FMXImpl_AOAOA<int4, float4, double4, float4, op>  },
                    { FMXImpl_AOAOA<int, float, double, double, op>, FMXImpl_AOAOA<int2, float2, double2, double2, op>, FMXImpl_AOAOA<int3, float3, double3, double3, op>, FMXImpl_AOAOA<int4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<int, double, uchar, uchar, op>, FMXImpl_AOAOA<int2, double2, uchar2, uchar2, op>, FMXImpl_AOAOA<int3, double3, uchar3, uchar3, op>, FMXImpl_AOAOA<int4, double4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, double, uchar, schar, op>, FMXImpl_AOAOA<int2, double2, uchar2, char2, op>, FMXImpl_AOAOA<int3, double3, uchar3, char3, op>, FMXImpl_AOAOA<int4, double4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<int, double, uchar, ushort, op>, FMXImpl_AOAOA<int2, double2, uchar2, ushort2, op>, FMXImpl_AOAOA<int3, double3, uchar3, ushort3, op>, FMXImpl_AOAOA<int4, double4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, double, uchar, short, op>, FMXImpl_AOAOA<int2, double2, uchar2, short2, op>, FMXImpl_AOAOA<int3, double3, uchar3, short3, op>, FMXImpl_AOAOA<int4, double4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<int, double, uchar, int, op>, FMXImpl_AOAOA<int2, double2, uchar2, int2, op>, FMXImpl_AOAOA<int3, double3, uchar3, int3, op>, FMXImpl_AOAOA<int4, double4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<int, double, uchar, float, op>, FMXImpl_AOAOA<int2, double2, uchar2, float2, op>, FMXImpl_AOAOA<int3, double3, uchar3, float3, op>, FMXImpl_AOAOA<int4, double4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<int, double, uchar, double, op>, FMXImpl_AOAOA<int2, double2, uchar2, double2, op>, FMXImpl_AOAOA<int3, double3, uchar3, double3, op>, FMXImpl_AOAOA<int4, double4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, double, schar, uchar, op>, FMXImpl_AOAOA<int2, double2, char2, uchar2, op>, FMXImpl_AOAOA<int3, double3, char3, uchar3, op>, FMXImpl_AOAOA<int4, double4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, double, schar, schar, op>, FMXImpl_AOAOA<int2, double2, char2, char2, op>, FMXImpl_AOAOA<int3, double3, char3, char3, op>, FMXImpl_AOAOA<int4, double4, char4, char4, op>  },
                    { FMXImpl_AOAOA<int, double, schar, ushort, op>, FMXImpl_AOAOA<int2, double2, char2, ushort2, op>, FMXImpl_AOAOA<int3, double3, char3, ushort3, op>, FMXImpl_AOAOA<int4, double4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, double, schar, short, op>, FMXImpl_AOAOA<int2, double2, char2, short2, op>, FMXImpl_AOAOA<int3, double3, char3, short3, op>, FMXImpl_AOAOA<int4, double4, char4, short4, op>  },
                    { FMXImpl_AOAOA<int, double, schar, int, op>, FMXImpl_AOAOA<int2, double2, char2, int2, op>, FMXImpl_AOAOA<int3, double3, char3, int3, op>, FMXImpl_AOAOA<int4, double4, char4, int4, op>  },
                    { FMXImpl_AOAOA<int, double, schar, float, op>, FMXImpl_AOAOA<int2, double2, char2, float2, op>, FMXImpl_AOAOA<int3, double3, char3, float3, op>, FMXImpl_AOAOA<int4, double4, char4, float4, op>  },
                    { FMXImpl_AOAOA<int, double, schar, double, op>, FMXImpl_AOAOA<int2, double2, char2, double2, op>, FMXImpl_AOAOA<int3, double3, char3, double3, op>, FMXImpl_AOAOA<int4, double4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, double, ushort, uchar, op>, FMXImpl_AOAOA<int2, double2, ushort2, uchar2, op>, FMXImpl_AOAOA<int3, double3, ushort3, uchar3, op>, FMXImpl_AOAOA<int4, double4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, double, ushort, schar, op>, FMXImpl_AOAOA<int2, double2, ushort2, char2, op>, FMXImpl_AOAOA<int3, double3, ushort3, char3, op>, FMXImpl_AOAOA<int4, double4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<int, double, ushort, ushort, op>, FMXImpl_AOAOA<int2, double2, ushort2, ushort2, op>, FMXImpl_AOAOA<int3, double3, ushort3, ushort3, op>, FMXImpl_AOAOA<int4, double4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, double, ushort, short, op>, FMXImpl_AOAOA<int2, double2, ushort2, short2, op>, FMXImpl_AOAOA<int3, double3, ushort3, short3, op>, FMXImpl_AOAOA<int4, double4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<int, double, ushort, int, op>, FMXImpl_AOAOA<int2, double2, ushort2, int2, op>, FMXImpl_AOAOA<int3, double3, ushort3, int3, op>, FMXImpl_AOAOA<int4, double4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<int, double, ushort, float, op>, FMXImpl_AOAOA<int2, double2, ushort2, float2, op>, FMXImpl_AOAOA<int3, double3, ushort3, float3, op>, FMXImpl_AOAOA<int4, double4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<int, double, ushort, double, op>, FMXImpl_AOAOA<int2, double2, ushort2, double2, op>, FMXImpl_AOAOA<int3, double3, ushort3, double3, op>, FMXImpl_AOAOA<int4, double4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, double, short, uchar, op>, FMXImpl_AOAOA<int2, double2, short2, uchar2, op>, FMXImpl_AOAOA<int3, double3, short3, uchar3, op>, FMXImpl_AOAOA<int4, double4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, double, short, schar, op>, FMXImpl_AOAOA<int2, double2, short2, char2, op>, FMXImpl_AOAOA<int3, double3, short3, char3, op>, FMXImpl_AOAOA<int4, double4, short4, char4, op>  },
                    { FMXImpl_AOAOA<int, double, short, ushort, op>, FMXImpl_AOAOA<int2, double2, short2, ushort2, op>, FMXImpl_AOAOA<int3, double3, short3, ushort3, op>, FMXImpl_AOAOA<int4, double4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, double, short, short, op>, FMXImpl_AOAOA<int2, double2, short2, short2, op>, FMXImpl_AOAOA<int3, double3, short3, short3, op>, FMXImpl_AOAOA<int4, double4, short4, short4, op>  },
                    { FMXImpl_AOAOA<int, double, short, int, op>, FMXImpl_AOAOA<int2, double2, short2, int2, op>, FMXImpl_AOAOA<int3, double3, short3, int3, op>, FMXImpl_AOAOA<int4, double4, short4, int4, op>  },
                    { FMXImpl_AOAOA<int, double, short, float, op>, FMXImpl_AOAOA<int2, double2, short2, float2, op>, FMXImpl_AOAOA<int3, double3, short3, float3, op>, FMXImpl_AOAOA<int4, double4, short4, float4, op>  },
                    { FMXImpl_AOAOA<int, double, short, double, op>, FMXImpl_AOAOA<int2, double2, short2, double2, op>, FMXImpl_AOAOA<int3, double3, short3, double3, op>, FMXImpl_AOAOA<int4, double4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, double, int, uchar, op>, FMXImpl_AOAOA<int2, double2, int2, uchar2, op>, FMXImpl_AOAOA<int3, double3, int3, uchar3, op>, FMXImpl_AOAOA<int4, double4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, double, int, schar, op>, FMXImpl_AOAOA<int2, double2, int2, char2, op>, FMXImpl_AOAOA<int3, double3, int3, char3, op>, FMXImpl_AOAOA<int4, double4, int4, char4, op>  },
                    { FMXImpl_AOAOA<int, double, int, ushort, op>, FMXImpl_AOAOA<int2, double2, int2, ushort2, op>, FMXImpl_AOAOA<int3, double3, int3, ushort3, op>, FMXImpl_AOAOA<int4, double4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, double, int, short, op>, FMXImpl_AOAOA<int2, double2, int2, short2, op>, FMXImpl_AOAOA<int3, double3, int3, short3, op>, FMXImpl_AOAOA<int4, double4, int4, short4, op>  },
                    { FMXImpl_AOAOA<int, double, int, int, op>, FMXImpl_AOAOA<int2, double2, int2, int2, op>, FMXImpl_AOAOA<int3, double3, int3, int3, op>, FMXImpl_AOAOA<int4, double4, int4, int4, op>  },
                    { FMXImpl_AOAOA<int, double, int, float, op>, FMXImpl_AOAOA<int2, double2, int2, float2, op>, FMXImpl_AOAOA<int3, double3, int3, float3, op>, FMXImpl_AOAOA<int4, double4, int4, float4, op>  },
                    { FMXImpl_AOAOA<int, double, int, double, op>, FMXImpl_AOAOA<int2, double2, int2, double2, op>, FMXImpl_AOAOA<int3, double3, int3, double3, op>, FMXImpl_AOAOA<int4, double4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, double, float, uchar, op>, FMXImpl_AOAOA<int2, double2, float2, uchar2, op>, FMXImpl_AOAOA<int3, double3, float3, uchar3, op>, FMXImpl_AOAOA<int4, double4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, double, float, schar, op>, FMXImpl_AOAOA<int2, double2, float2, char2, op>, FMXImpl_AOAOA<int3, double3, float3, char3, op>, FMXImpl_AOAOA<int4, double4, float4, char4, op>  },
                    { FMXImpl_AOAOA<int, double, float, ushort, op>, FMXImpl_AOAOA<int2, double2, float2, ushort2, op>, FMXImpl_AOAOA<int3, double3, float3, ushort3, op>, FMXImpl_AOAOA<int4, double4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, double, float, short, op>, FMXImpl_AOAOA<int2, double2, float2, short2, op>, FMXImpl_AOAOA<int3, double3, float3, short3, op>, FMXImpl_AOAOA<int4, double4, float4, short4, op>  },
                    { FMXImpl_AOAOA<int, double, float, int, op>, FMXImpl_AOAOA<int2, double2, float2, int2, op>, FMXImpl_AOAOA<int3, double3, float3, int3, op>, FMXImpl_AOAOA<int4, double4, float4, int4, op>  },
                    { FMXImpl_AOAOA<int, double, float, float, op>, FMXImpl_AOAOA<int2, double2, float2, float2, op>, FMXImpl_AOAOA<int3, double3, float3, float3, op>, FMXImpl_AOAOA<int4, double4, float4, float4, op>  },
                    { FMXImpl_AOAOA<int, double, float, double, op>, FMXImpl_AOAOA<int2, double2, float2, double2, op>, FMXImpl_AOAOA<int3, double3, float3, double3, op>, FMXImpl_AOAOA<int4, double4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<int, double, double, uchar, op>, FMXImpl_AOAOA<int2, double2, double2, uchar2, op>, FMXImpl_AOAOA<int3, double3, double3, uchar3, op>, FMXImpl_AOAOA<int4, double4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<int, double, double, schar, op>, FMXImpl_AOAOA<int2, double2, double2, char2, op>, FMXImpl_AOAOA<int3, double3, double3, char3, op>, FMXImpl_AOAOA<int4, double4, double4, char4, op>  },
                    { FMXImpl_AOAOA<int, double, double, ushort, op>, FMXImpl_AOAOA<int2, double2, double2, ushort2, op>, FMXImpl_AOAOA<int3, double3, double3, ushort3, op>, FMXImpl_AOAOA<int4, double4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<int, double, double, short, op>, FMXImpl_AOAOA<int2, double2, double2, short2, op>, FMXImpl_AOAOA<int3, double3, double3, short3, op>, FMXImpl_AOAOA<int4, double4, double4, short4, op>  },
                    { FMXImpl_AOAOA<int, double, double, int, op>, FMXImpl_AOAOA<int2, double2, double2, int2, op>, FMXImpl_AOAOA<int3, double3, double3, int3, op>, FMXImpl_AOAOA<int4, double4, double4, int4, op>  },
                    { FMXImpl_AOAOA<int, double, double, float, op>, FMXImpl_AOAOA<int2, double2, double2, float2, op>, FMXImpl_AOAOA<int3, double3, double3, float3, op>, FMXImpl_AOAOA<int4, double4, double4, float4, op>  },
                    { FMXImpl_AOAOA<int, double, double, double, op>, FMXImpl_AOAOA<int2, double2, double2, double2, op>, FMXImpl_AOAOA<int3, double3, double3, double3, op>, FMXImpl_AOAOA<int4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { FMXImpl_AOAOA<float, uchar, uchar, uchar, op>, FMXImpl_AOAOA<float2, uchar2, uchar2, uchar2, op>, FMXImpl_AOAOA<float3, uchar3, uchar3, uchar3, op>, FMXImpl_AOAOA<float4, uchar4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, uchar, uchar, schar, op>, FMXImpl_AOAOA<float2, uchar2, uchar2, char2, op>, FMXImpl_AOAOA<float3, uchar3, uchar3, char3, op>, FMXImpl_AOAOA<float4, uchar4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<float, uchar, uchar, ushort, op>, FMXImpl_AOAOA<float2, uchar2, uchar2, ushort2, op>, FMXImpl_AOAOA<float3, uchar3, uchar3, ushort3, op>, FMXImpl_AOAOA<float4, uchar4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, uchar, uchar, short, op>, FMXImpl_AOAOA<float2, uchar2, uchar2, short2, op>, FMXImpl_AOAOA<float3, uchar3, uchar3, short3, op>, FMXImpl_AOAOA<float4, uchar4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<float, uchar, uchar, int, op>, FMXImpl_AOAOA<float2, uchar2, uchar2, int2, op>, FMXImpl_AOAOA<float3, uchar3, uchar3, int3, op>, FMXImpl_AOAOA<float4, uchar4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<float, uchar, uchar, float, op>, FMXImpl_AOAOA<float2, uchar2, uchar2, float2, op>, FMXImpl_AOAOA<float3, uchar3, uchar3, float3, op>, FMXImpl_AOAOA<float4, uchar4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<float, uchar, uchar, double, op>, FMXImpl_AOAOA<float2, uchar2, uchar2, double2, op>, FMXImpl_AOAOA<float3, uchar3, uchar3, double3, op>, FMXImpl_AOAOA<float4, uchar4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, uchar, schar, uchar, op>, FMXImpl_AOAOA<float2, uchar2, char2, uchar2, op>, FMXImpl_AOAOA<float3, uchar3, char3, uchar3, op>, FMXImpl_AOAOA<float4, uchar4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, uchar, schar, schar, op>, FMXImpl_AOAOA<float2, uchar2, char2, char2, op>, FMXImpl_AOAOA<float3, uchar3, char3, char3, op>, FMXImpl_AOAOA<float4, uchar4, char4, char4, op>  },
                    { FMXImpl_AOAOA<float, uchar, schar, ushort, op>, FMXImpl_AOAOA<float2, uchar2, char2, ushort2, op>, FMXImpl_AOAOA<float3, uchar3, char3, ushort3, op>, FMXImpl_AOAOA<float4, uchar4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, uchar, schar, short, op>, FMXImpl_AOAOA<float2, uchar2, char2, short2, op>, FMXImpl_AOAOA<float3, uchar3, char3, short3, op>, FMXImpl_AOAOA<float4, uchar4, char4, short4, op>  },
                    { FMXImpl_AOAOA<float, uchar, schar, int, op>, FMXImpl_AOAOA<float2, uchar2, char2, int2, op>, FMXImpl_AOAOA<float3, uchar3, char3, int3, op>, FMXImpl_AOAOA<float4, uchar4, char4, int4, op>  },
                    { FMXImpl_AOAOA<float, uchar, schar, float, op>, FMXImpl_AOAOA<float2, uchar2, char2, float2, op>, FMXImpl_AOAOA<float3, uchar3, char3, float3, op>, FMXImpl_AOAOA<float4, uchar4, char4, float4, op>  },
                    { FMXImpl_AOAOA<float, uchar, schar, double, op>, FMXImpl_AOAOA<float2, uchar2, char2, double2, op>, FMXImpl_AOAOA<float3, uchar3, char3, double3, op>, FMXImpl_AOAOA<float4, uchar4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, uchar, ushort, uchar, op>, FMXImpl_AOAOA<float2, uchar2, ushort2, uchar2, op>, FMXImpl_AOAOA<float3, uchar3, ushort3, uchar3, op>, FMXImpl_AOAOA<float4, uchar4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, uchar, ushort, schar, op>, FMXImpl_AOAOA<float2, uchar2, ushort2, char2, op>, FMXImpl_AOAOA<float3, uchar3, ushort3, char3, op>, FMXImpl_AOAOA<float4, uchar4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<float, uchar, ushort, ushort, op>, FMXImpl_AOAOA<float2, uchar2, ushort2, ushort2, op>, FMXImpl_AOAOA<float3, uchar3, ushort3, ushort3, op>, FMXImpl_AOAOA<float4, uchar4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, uchar, ushort, short, op>, FMXImpl_AOAOA<float2, uchar2, ushort2, short2, op>, FMXImpl_AOAOA<float3, uchar3, ushort3, short3, op>, FMXImpl_AOAOA<float4, uchar4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<float, uchar, ushort, int, op>, FMXImpl_AOAOA<float2, uchar2, ushort2, int2, op>, FMXImpl_AOAOA<float3, uchar3, ushort3, int3, op>, FMXImpl_AOAOA<float4, uchar4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<float, uchar, ushort, float, op>, FMXImpl_AOAOA<float2, uchar2, ushort2, float2, op>, FMXImpl_AOAOA<float3, uchar3, ushort3, float3, op>, FMXImpl_AOAOA<float4, uchar4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<float, uchar, ushort, double, op>, FMXImpl_AOAOA<float2, uchar2, ushort2, double2, op>, FMXImpl_AOAOA<float3, uchar3, ushort3, double3, op>, FMXImpl_AOAOA<float4, uchar4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, uchar, short, uchar, op>, FMXImpl_AOAOA<float2, uchar2, short2, uchar2, op>, FMXImpl_AOAOA<float3, uchar3, short3, uchar3, op>, FMXImpl_AOAOA<float4, uchar4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, uchar, short, schar, op>, FMXImpl_AOAOA<float2, uchar2, short2, char2, op>, FMXImpl_AOAOA<float3, uchar3, short3, char3, op>, FMXImpl_AOAOA<float4, uchar4, short4, char4, op>  },
                    { FMXImpl_AOAOA<float, uchar, short, ushort, op>, FMXImpl_AOAOA<float2, uchar2, short2, ushort2, op>, FMXImpl_AOAOA<float3, uchar3, short3, ushort3, op>, FMXImpl_AOAOA<float4, uchar4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, uchar, short, short, op>, FMXImpl_AOAOA<float2, uchar2, short2, short2, op>, FMXImpl_AOAOA<float3, uchar3, short3, short3, op>, FMXImpl_AOAOA<float4, uchar4, short4, short4, op>  },
                    { FMXImpl_AOAOA<float, uchar, short, int, op>, FMXImpl_AOAOA<float2, uchar2, short2, int2, op>, FMXImpl_AOAOA<float3, uchar3, short3, int3, op>, FMXImpl_AOAOA<float4, uchar4, short4, int4, op>  },
                    { FMXImpl_AOAOA<float, uchar, short, float, op>, FMXImpl_AOAOA<float2, uchar2, short2, float2, op>, FMXImpl_AOAOA<float3, uchar3, short3, float3, op>, FMXImpl_AOAOA<float4, uchar4, short4, float4, op>  },
                    { FMXImpl_AOAOA<float, uchar, short, double, op>, FMXImpl_AOAOA<float2, uchar2, short2, double2, op>, FMXImpl_AOAOA<float3, uchar3, short3, double3, op>, FMXImpl_AOAOA<float4, uchar4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, uchar, int, uchar, op>, FMXImpl_AOAOA<float2, uchar2, int2, uchar2, op>, FMXImpl_AOAOA<float3, uchar3, int3, uchar3, op>, FMXImpl_AOAOA<float4, uchar4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, uchar, int, schar, op>, FMXImpl_AOAOA<float2, uchar2, int2, char2, op>, FMXImpl_AOAOA<float3, uchar3, int3, char3, op>, FMXImpl_AOAOA<float4, uchar4, int4, char4, op>  },
                    { FMXImpl_AOAOA<float, uchar, int, ushort, op>, FMXImpl_AOAOA<float2, uchar2, int2, ushort2, op>, FMXImpl_AOAOA<float3, uchar3, int3, ushort3, op>, FMXImpl_AOAOA<float4, uchar4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, uchar, int, short, op>, FMXImpl_AOAOA<float2, uchar2, int2, short2, op>, FMXImpl_AOAOA<float3, uchar3, int3, short3, op>, FMXImpl_AOAOA<float4, uchar4, int4, short4, op>  },
                    { FMXImpl_AOAOA<float, uchar, int, int, op>, FMXImpl_AOAOA<float2, uchar2, int2, int2, op>, FMXImpl_AOAOA<float3, uchar3, int3, int3, op>, FMXImpl_AOAOA<float4, uchar4, int4, int4, op>  },
                    { FMXImpl_AOAOA<float, uchar, int, float, op>, FMXImpl_AOAOA<float2, uchar2, int2, float2, op>, FMXImpl_AOAOA<float3, uchar3, int3, float3, op>, FMXImpl_AOAOA<float4, uchar4, int4, float4, op>  },
                    { FMXImpl_AOAOA<float, uchar, int, double, op>, FMXImpl_AOAOA<float2, uchar2, int2, double2, op>, FMXImpl_AOAOA<float3, uchar3, int3, double3, op>, FMXImpl_AOAOA<float4, uchar4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, uchar, float, uchar, op>, FMXImpl_AOAOA<float2, uchar2, float2, uchar2, op>, FMXImpl_AOAOA<float3, uchar3, float3, uchar3, op>, FMXImpl_AOAOA<float4, uchar4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, uchar, float, schar, op>, FMXImpl_AOAOA<float2, uchar2, float2, char2, op>, FMXImpl_AOAOA<float3, uchar3, float3, char3, op>, FMXImpl_AOAOA<float4, uchar4, float4, char4, op>  },
                    { FMXImpl_AOAOA<float, uchar, float, ushort, op>, FMXImpl_AOAOA<float2, uchar2, float2, ushort2, op>, FMXImpl_AOAOA<float3, uchar3, float3, ushort3, op>, FMXImpl_AOAOA<float4, uchar4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, uchar, float, short, op>, FMXImpl_AOAOA<float2, uchar2, float2, short2, op>, FMXImpl_AOAOA<float3, uchar3, float3, short3, op>, FMXImpl_AOAOA<float4, uchar4, float4, short4, op>  },
                    { FMXImpl_AOAOA<float, uchar, float, int, op>, FMXImpl_AOAOA<float2, uchar2, float2, int2, op>, FMXImpl_AOAOA<float3, uchar3, float3, int3, op>, FMXImpl_AOAOA<float4, uchar4, float4, int4, op>  },
                    { FMXImpl_AOAOA<float, uchar, float, float, op>, FMXImpl_AOAOA<float2, uchar2, float2, float2, op>, FMXImpl_AOAOA<float3, uchar3, float3, float3, op>, FMXImpl_AOAOA<float4, uchar4, float4, float4, op>  },
                    { FMXImpl_AOAOA<float, uchar, float, double, op>, FMXImpl_AOAOA<float2, uchar2, float2, double2, op>, FMXImpl_AOAOA<float3, uchar3, float3, double3, op>, FMXImpl_AOAOA<float4, uchar4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, uchar, double, uchar, op>, FMXImpl_AOAOA<float2, uchar2, double2, uchar2, op>, FMXImpl_AOAOA<float3, uchar3, double3, uchar3, op>, FMXImpl_AOAOA<float4, uchar4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, uchar, double, schar, op>, FMXImpl_AOAOA<float2, uchar2, double2, char2, op>, FMXImpl_AOAOA<float3, uchar3, double3, char3, op>, FMXImpl_AOAOA<float4, uchar4, double4, char4, op>  },
                    { FMXImpl_AOAOA<float, uchar, double, ushort, op>, FMXImpl_AOAOA<float2, uchar2, double2, ushort2, op>, FMXImpl_AOAOA<float3, uchar3, double3, ushort3, op>, FMXImpl_AOAOA<float4, uchar4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, uchar, double, short, op>, FMXImpl_AOAOA<float2, uchar2, double2, short2, op>, FMXImpl_AOAOA<float3, uchar3, double3, short3, op>, FMXImpl_AOAOA<float4, uchar4, double4, short4, op>  },
                    { FMXImpl_AOAOA<float, uchar, double, int, op>, FMXImpl_AOAOA<float2, uchar2, double2, int2, op>, FMXImpl_AOAOA<float3, uchar3, double3, int3, op>, FMXImpl_AOAOA<float4, uchar4, double4, int4, op>  },
                    { FMXImpl_AOAOA<float, uchar, double, float, op>, FMXImpl_AOAOA<float2, uchar2, double2, float2, op>, FMXImpl_AOAOA<float3, uchar3, double3, float3, op>, FMXImpl_AOAOA<float4, uchar4, double4, float4, op>  },
                    { FMXImpl_AOAOA<float, uchar, double, double, op>, FMXImpl_AOAOA<float2, uchar2, double2, double2, op>, FMXImpl_AOAOA<float3, uchar3, double3, double3, op>, FMXImpl_AOAOA<float4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<float, schar, uchar, uchar, op>, FMXImpl_AOAOA<float2, char2, uchar2, uchar2, op>, FMXImpl_AOAOA<float3, char3, uchar3, uchar3, op>, FMXImpl_AOAOA<float4, char4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, schar, uchar, schar, op>, FMXImpl_AOAOA<float2, char2, uchar2, char2, op>, FMXImpl_AOAOA<float3, char3, uchar3, char3, op>, FMXImpl_AOAOA<float4, char4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<float, schar, uchar, ushort, op>, FMXImpl_AOAOA<float2, char2, uchar2, ushort2, op>, FMXImpl_AOAOA<float3, char3, uchar3, ushort3, op>, FMXImpl_AOAOA<float4, char4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, schar, uchar, short, op>, FMXImpl_AOAOA<float2, char2, uchar2, short2, op>, FMXImpl_AOAOA<float3, char3, uchar3, short3, op>, FMXImpl_AOAOA<float4, char4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<float, schar, uchar, int, op>, FMXImpl_AOAOA<float2, char2, uchar2, int2, op>, FMXImpl_AOAOA<float3, char3, uchar3, int3, op>, FMXImpl_AOAOA<float4, char4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<float, schar, uchar, float, op>, FMXImpl_AOAOA<float2, char2, uchar2, float2, op>, FMXImpl_AOAOA<float3, char3, uchar3, float3, op>, FMXImpl_AOAOA<float4, char4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<float, schar, uchar, double, op>, FMXImpl_AOAOA<float2, char2, uchar2, double2, op>, FMXImpl_AOAOA<float3, char3, uchar3, double3, op>, FMXImpl_AOAOA<float4, char4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, schar, schar, uchar, op>, FMXImpl_AOAOA<float2, char2, char2, uchar2, op>, FMXImpl_AOAOA<float3, char3, char3, uchar3, op>, FMXImpl_AOAOA<float4, char4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, schar, schar, schar, op>, FMXImpl_AOAOA<float2, char2, char2, char2, op>, FMXImpl_AOAOA<float3, char3, char3, char3, op>, FMXImpl_AOAOA<float4, char4, char4, char4, op>  },
                    { FMXImpl_AOAOA<float, schar, schar, ushort, op>, FMXImpl_AOAOA<float2, char2, char2, ushort2, op>, FMXImpl_AOAOA<float3, char3, char3, ushort3, op>, FMXImpl_AOAOA<float4, char4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, schar, schar, short, op>, FMXImpl_AOAOA<float2, char2, char2, short2, op>, FMXImpl_AOAOA<float3, char3, char3, short3, op>, FMXImpl_AOAOA<float4, char4, char4, short4, op>  },
                    { FMXImpl_AOAOA<float, schar, schar, int, op>, FMXImpl_AOAOA<float2, char2, char2, int2, op>, FMXImpl_AOAOA<float3, char3, char3, int3, op>, FMXImpl_AOAOA<float4, char4, char4, int4, op>  },
                    { FMXImpl_AOAOA<float, schar, schar, float, op>, FMXImpl_AOAOA<float2, char2, char2, float2, op>, FMXImpl_AOAOA<float3, char3, char3, float3, op>, FMXImpl_AOAOA<float4, char4, char4, float4, op>  },
                    { FMXImpl_AOAOA<float, schar, schar, double, op>, FMXImpl_AOAOA<float2, char2, char2, double2, op>, FMXImpl_AOAOA<float3, char3, char3, double3, op>, FMXImpl_AOAOA<float4, char4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, schar, ushort, uchar, op>, FMXImpl_AOAOA<float2, char2, ushort2, uchar2, op>, FMXImpl_AOAOA<float3, char3, ushort3, uchar3, op>, FMXImpl_AOAOA<float4, char4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, schar, ushort, schar, op>, FMXImpl_AOAOA<float2, char2, ushort2, char2, op>, FMXImpl_AOAOA<float3, char3, ushort3, char3, op>, FMXImpl_AOAOA<float4, char4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<float, schar, ushort, ushort, op>, FMXImpl_AOAOA<float2, char2, ushort2, ushort2, op>, FMXImpl_AOAOA<float3, char3, ushort3, ushort3, op>, FMXImpl_AOAOA<float4, char4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, schar, ushort, short, op>, FMXImpl_AOAOA<float2, char2, ushort2, short2, op>, FMXImpl_AOAOA<float3, char3, ushort3, short3, op>, FMXImpl_AOAOA<float4, char4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<float, schar, ushort, int, op>, FMXImpl_AOAOA<float2, char2, ushort2, int2, op>, FMXImpl_AOAOA<float3, char3, ushort3, int3, op>, FMXImpl_AOAOA<float4, char4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<float, schar, ushort, float, op>, FMXImpl_AOAOA<float2, char2, ushort2, float2, op>, FMXImpl_AOAOA<float3, char3, ushort3, float3, op>, FMXImpl_AOAOA<float4, char4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<float, schar, ushort, double, op>, FMXImpl_AOAOA<float2, char2, ushort2, double2, op>, FMXImpl_AOAOA<float3, char3, ushort3, double3, op>, FMXImpl_AOAOA<float4, char4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, schar, short, uchar, op>, FMXImpl_AOAOA<float2, char2, short2, uchar2, op>, FMXImpl_AOAOA<float3, char3, short3, uchar3, op>, FMXImpl_AOAOA<float4, char4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, schar, short, schar, op>, FMXImpl_AOAOA<float2, char2, short2, char2, op>, FMXImpl_AOAOA<float3, char3, short3, char3, op>, FMXImpl_AOAOA<float4, char4, short4, char4, op>  },
                    { FMXImpl_AOAOA<float, schar, short, ushort, op>, FMXImpl_AOAOA<float2, char2, short2, ushort2, op>, FMXImpl_AOAOA<float3, char3, short3, ushort3, op>, FMXImpl_AOAOA<float4, char4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, schar, short, short, op>, FMXImpl_AOAOA<float2, char2, short2, short2, op>, FMXImpl_AOAOA<float3, char3, short3, short3, op>, FMXImpl_AOAOA<float4, char4, short4, short4, op>  },
                    { FMXImpl_AOAOA<float, schar, short, int, op>, FMXImpl_AOAOA<float2, char2, short2, int2, op>, FMXImpl_AOAOA<float3, char3, short3, int3, op>, FMXImpl_AOAOA<float4, char4, short4, int4, op>  },
                    { FMXImpl_AOAOA<float, schar, short, float, op>, FMXImpl_AOAOA<float2, char2, short2, float2, op>, FMXImpl_AOAOA<float3, char3, short3, float3, op>, FMXImpl_AOAOA<float4, char4, short4, float4, op>  },
                    { FMXImpl_AOAOA<float, schar, short, double, op>, FMXImpl_AOAOA<float2, char2, short2, double2, op>, FMXImpl_AOAOA<float3, char3, short3, double3, op>, FMXImpl_AOAOA<float4, char4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, schar, int, uchar, op>, FMXImpl_AOAOA<float2, char2, int2, uchar2, op>, FMXImpl_AOAOA<float3, char3, int3, uchar3, op>, FMXImpl_AOAOA<float4, char4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, schar, int, schar, op>, FMXImpl_AOAOA<float2, char2, int2, char2, op>, FMXImpl_AOAOA<float3, char3, int3, char3, op>, FMXImpl_AOAOA<float4, char4, int4, char4, op>  },
                    { FMXImpl_AOAOA<float, schar, int, ushort, op>, FMXImpl_AOAOA<float2, char2, int2, ushort2, op>, FMXImpl_AOAOA<float3, char3, int3, ushort3, op>, FMXImpl_AOAOA<float4, char4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, schar, int, short, op>, FMXImpl_AOAOA<float2, char2, int2, short2, op>, FMXImpl_AOAOA<float3, char3, int3, short3, op>, FMXImpl_AOAOA<float4, char4, int4, short4, op>  },
                    { FMXImpl_AOAOA<float, schar, int, int, op>, FMXImpl_AOAOA<float2, char2, int2, int2, op>, FMXImpl_AOAOA<float3, char3, int3, int3, op>, FMXImpl_AOAOA<float4, char4, int4, int4, op>  },
                    { FMXImpl_AOAOA<float, schar, int, float, op>, FMXImpl_AOAOA<float2, char2, int2, float2, op>, FMXImpl_AOAOA<float3, char3, int3, float3, op>, FMXImpl_AOAOA<float4, char4, int4, float4, op>  },
                    { FMXImpl_AOAOA<float, schar, int, double, op>, FMXImpl_AOAOA<float2, char2, int2, double2, op>, FMXImpl_AOAOA<float3, char3, int3, double3, op>, FMXImpl_AOAOA<float4, char4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, schar, float, uchar, op>, FMXImpl_AOAOA<float2, char2, float2, uchar2, op>, FMXImpl_AOAOA<float3, char3, float3, uchar3, op>, FMXImpl_AOAOA<float4, char4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, schar, float, schar, op>, FMXImpl_AOAOA<float2, char2, float2, char2, op>, FMXImpl_AOAOA<float3, char3, float3, char3, op>, FMXImpl_AOAOA<float4, char4, float4, char4, op>  },
                    { FMXImpl_AOAOA<float, schar, float, ushort, op>, FMXImpl_AOAOA<float2, char2, float2, ushort2, op>, FMXImpl_AOAOA<float3, char3, float3, ushort3, op>, FMXImpl_AOAOA<float4, char4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, schar, float, short, op>, FMXImpl_AOAOA<float2, char2, float2, short2, op>, FMXImpl_AOAOA<float3, char3, float3, short3, op>, FMXImpl_AOAOA<float4, char4, float4, short4, op>  },
                    { FMXImpl_AOAOA<float, schar, float, int, op>, FMXImpl_AOAOA<float2, char2, float2, int2, op>, FMXImpl_AOAOA<float3, char3, float3, int3, op>, FMXImpl_AOAOA<float4, char4, float4, int4, op>  },
                    { FMXImpl_AOAOA<float, schar, float, float, op>, FMXImpl_AOAOA<float2, char2, float2, float2, op>, FMXImpl_AOAOA<float3, char3, float3, float3, op>, FMXImpl_AOAOA<float4, char4, float4, float4, op>  },
                    { FMXImpl_AOAOA<float, schar, float, double, op>, FMXImpl_AOAOA<float2, char2, float2, double2, op>, FMXImpl_AOAOA<float3, char3, float3, double3, op>, FMXImpl_AOAOA<float4, char4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, schar, double, uchar, op>, FMXImpl_AOAOA<float2, char2, double2, uchar2, op>, FMXImpl_AOAOA<float3, char3, double3, uchar3, op>, FMXImpl_AOAOA<float4, char4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, schar, double, schar, op>, FMXImpl_AOAOA<float2, char2, double2, char2, op>, FMXImpl_AOAOA<float3, char3, double3, char3, op>, FMXImpl_AOAOA<float4, char4, double4, char4, op>  },
                    { FMXImpl_AOAOA<float, schar, double, ushort, op>, FMXImpl_AOAOA<float2, char2, double2, ushort2, op>, FMXImpl_AOAOA<float3, char3, double3, ushort3, op>, FMXImpl_AOAOA<float4, char4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, schar, double, short, op>, FMXImpl_AOAOA<float2, char2, double2, short2, op>, FMXImpl_AOAOA<float3, char3, double3, short3, op>, FMXImpl_AOAOA<float4, char4, double4, short4, op>  },
                    { FMXImpl_AOAOA<float, schar, double, int, op>, FMXImpl_AOAOA<float2, char2, double2, int2, op>, FMXImpl_AOAOA<float3, char3, double3, int3, op>, FMXImpl_AOAOA<float4, char4, double4, int4, op>  },
                    { FMXImpl_AOAOA<float, schar, double, float, op>, FMXImpl_AOAOA<float2, char2, double2, float2, op>, FMXImpl_AOAOA<float3, char3, double3, float3, op>, FMXImpl_AOAOA<float4, char4, double4, float4, op>  },
                    { FMXImpl_AOAOA<float, schar, double, double, op>, FMXImpl_AOAOA<float2, char2, double2, double2, op>, FMXImpl_AOAOA<float3, char3, double3, double3, op>, FMXImpl_AOAOA<float4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<float, ushort, uchar, uchar, op>, FMXImpl_AOAOA<float2, ushort2, uchar2, uchar2, op>, FMXImpl_AOAOA<float3, ushort3, uchar3, uchar3, op>, FMXImpl_AOAOA<float4, ushort4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, ushort, uchar, schar, op>, FMXImpl_AOAOA<float2, ushort2, uchar2, char2, op>, FMXImpl_AOAOA<float3, ushort3, uchar3, char3, op>, FMXImpl_AOAOA<float4, ushort4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<float, ushort, uchar, ushort, op>, FMXImpl_AOAOA<float2, ushort2, uchar2, ushort2, op>, FMXImpl_AOAOA<float3, ushort3, uchar3, ushort3, op>, FMXImpl_AOAOA<float4, ushort4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, ushort, uchar, short, op>, FMXImpl_AOAOA<float2, ushort2, uchar2, short2, op>, FMXImpl_AOAOA<float3, ushort3, uchar3, short3, op>, FMXImpl_AOAOA<float4, ushort4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<float, ushort, uchar, int, op>, FMXImpl_AOAOA<float2, ushort2, uchar2, int2, op>, FMXImpl_AOAOA<float3, ushort3, uchar3, int3, op>, FMXImpl_AOAOA<float4, ushort4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<float, ushort, uchar, float, op>, FMXImpl_AOAOA<float2, ushort2, uchar2, float2, op>, FMXImpl_AOAOA<float3, ushort3, uchar3, float3, op>, FMXImpl_AOAOA<float4, ushort4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<float, ushort, uchar, double, op>, FMXImpl_AOAOA<float2, ushort2, uchar2, double2, op>, FMXImpl_AOAOA<float3, ushort3, uchar3, double3, op>, FMXImpl_AOAOA<float4, ushort4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, ushort, schar, uchar, op>, FMXImpl_AOAOA<float2, ushort2, char2, uchar2, op>, FMXImpl_AOAOA<float3, ushort3, char3, uchar3, op>, FMXImpl_AOAOA<float4, ushort4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, ushort, schar, schar, op>, FMXImpl_AOAOA<float2, ushort2, char2, char2, op>, FMXImpl_AOAOA<float3, ushort3, char3, char3, op>, FMXImpl_AOAOA<float4, ushort4, char4, char4, op>  },
                    { FMXImpl_AOAOA<float, ushort, schar, ushort, op>, FMXImpl_AOAOA<float2, ushort2, char2, ushort2, op>, FMXImpl_AOAOA<float3, ushort3, char3, ushort3, op>, FMXImpl_AOAOA<float4, ushort4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, ushort, schar, short, op>, FMXImpl_AOAOA<float2, ushort2, char2, short2, op>, FMXImpl_AOAOA<float3, ushort3, char3, short3, op>, FMXImpl_AOAOA<float4, ushort4, char4, short4, op>  },
                    { FMXImpl_AOAOA<float, ushort, schar, int, op>, FMXImpl_AOAOA<float2, ushort2, char2, int2, op>, FMXImpl_AOAOA<float3, ushort3, char3, int3, op>, FMXImpl_AOAOA<float4, ushort4, char4, int4, op>  },
                    { FMXImpl_AOAOA<float, ushort, schar, float, op>, FMXImpl_AOAOA<float2, ushort2, char2, float2, op>, FMXImpl_AOAOA<float3, ushort3, char3, float3, op>, FMXImpl_AOAOA<float4, ushort4, char4, float4, op>  },
                    { FMXImpl_AOAOA<float, ushort, schar, double, op>, FMXImpl_AOAOA<float2, ushort2, char2, double2, op>, FMXImpl_AOAOA<float3, ushort3, char3, double3, op>, FMXImpl_AOAOA<float4, ushort4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, ushort, ushort, uchar, op>, FMXImpl_AOAOA<float2, ushort2, ushort2, uchar2, op>, FMXImpl_AOAOA<float3, ushort3, ushort3, uchar3, op>, FMXImpl_AOAOA<float4, ushort4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, ushort, ushort, schar, op>, FMXImpl_AOAOA<float2, ushort2, ushort2, char2, op>, FMXImpl_AOAOA<float3, ushort3, ushort3, char3, op>, FMXImpl_AOAOA<float4, ushort4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<float, ushort, ushort, ushort, op>, FMXImpl_AOAOA<float2, ushort2, ushort2, ushort2, op>, FMXImpl_AOAOA<float3, ushort3, ushort3, ushort3, op>, FMXImpl_AOAOA<float4, ushort4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, ushort, ushort, short, op>, FMXImpl_AOAOA<float2, ushort2, ushort2, short2, op>, FMXImpl_AOAOA<float3, ushort3, ushort3, short3, op>, FMXImpl_AOAOA<float4, ushort4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<float, ushort, ushort, int, op>, FMXImpl_AOAOA<float2, ushort2, ushort2, int2, op>, FMXImpl_AOAOA<float3, ushort3, ushort3, int3, op>, FMXImpl_AOAOA<float4, ushort4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<float, ushort, ushort, float, op>, FMXImpl_AOAOA<float2, ushort2, ushort2, float2, op>, FMXImpl_AOAOA<float3, ushort3, ushort3, float3, op>, FMXImpl_AOAOA<float4, ushort4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<float, ushort, ushort, double, op>, FMXImpl_AOAOA<float2, ushort2, ushort2, double2, op>, FMXImpl_AOAOA<float3, ushort3, ushort3, double3, op>, FMXImpl_AOAOA<float4, ushort4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, ushort, short, uchar, op>, FMXImpl_AOAOA<float2, ushort2, short2, uchar2, op>, FMXImpl_AOAOA<float3, ushort3, short3, uchar3, op>, FMXImpl_AOAOA<float4, ushort4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, ushort, short, schar, op>, FMXImpl_AOAOA<float2, ushort2, short2, char2, op>, FMXImpl_AOAOA<float3, ushort3, short3, char3, op>, FMXImpl_AOAOA<float4, ushort4, short4, char4, op>  },
                    { FMXImpl_AOAOA<float, ushort, short, ushort, op>, FMXImpl_AOAOA<float2, ushort2, short2, ushort2, op>, FMXImpl_AOAOA<float3, ushort3, short3, ushort3, op>, FMXImpl_AOAOA<float4, ushort4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, ushort, short, short, op>, FMXImpl_AOAOA<float2, ushort2, short2, short2, op>, FMXImpl_AOAOA<float3, ushort3, short3, short3, op>, FMXImpl_AOAOA<float4, ushort4, short4, short4, op>  },
                    { FMXImpl_AOAOA<float, ushort, short, int, op>, FMXImpl_AOAOA<float2, ushort2, short2, int2, op>, FMXImpl_AOAOA<float3, ushort3, short3, int3, op>, FMXImpl_AOAOA<float4, ushort4, short4, int4, op>  },
                    { FMXImpl_AOAOA<float, ushort, short, float, op>, FMXImpl_AOAOA<float2, ushort2, short2, float2, op>, FMXImpl_AOAOA<float3, ushort3, short3, float3, op>, FMXImpl_AOAOA<float4, ushort4, short4, float4, op>  },
                    { FMXImpl_AOAOA<float, ushort, short, double, op>, FMXImpl_AOAOA<float2, ushort2, short2, double2, op>, FMXImpl_AOAOA<float3, ushort3, short3, double3, op>, FMXImpl_AOAOA<float4, ushort4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, ushort, int, uchar, op>, FMXImpl_AOAOA<float2, ushort2, int2, uchar2, op>, FMXImpl_AOAOA<float3, ushort3, int3, uchar3, op>, FMXImpl_AOAOA<float4, ushort4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, ushort, int, schar, op>, FMXImpl_AOAOA<float2, ushort2, int2, char2, op>, FMXImpl_AOAOA<float3, ushort3, int3, char3, op>, FMXImpl_AOAOA<float4, ushort4, int4, char4, op>  },
                    { FMXImpl_AOAOA<float, ushort, int, ushort, op>, FMXImpl_AOAOA<float2, ushort2, int2, ushort2, op>, FMXImpl_AOAOA<float3, ushort3, int3, ushort3, op>, FMXImpl_AOAOA<float4, ushort4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, ushort, int, short, op>, FMXImpl_AOAOA<float2, ushort2, int2, short2, op>, FMXImpl_AOAOA<float3, ushort3, int3, short3, op>, FMXImpl_AOAOA<float4, ushort4, int4, short4, op>  },
                    { FMXImpl_AOAOA<float, ushort, int, int, op>, FMXImpl_AOAOA<float2, ushort2, int2, int2, op>, FMXImpl_AOAOA<float3, ushort3, int3, int3, op>, FMXImpl_AOAOA<float4, ushort4, int4, int4, op>  },
                    { FMXImpl_AOAOA<float, ushort, int, float, op>, FMXImpl_AOAOA<float2, ushort2, int2, float2, op>, FMXImpl_AOAOA<float3, ushort3, int3, float3, op>, FMXImpl_AOAOA<float4, ushort4, int4, float4, op>  },
                    { FMXImpl_AOAOA<float, ushort, int, double, op>, FMXImpl_AOAOA<float2, ushort2, int2, double2, op>, FMXImpl_AOAOA<float3, ushort3, int3, double3, op>, FMXImpl_AOAOA<float4, ushort4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, ushort, float, uchar, op>, FMXImpl_AOAOA<float2, ushort2, float2, uchar2, op>, FMXImpl_AOAOA<float3, ushort3, float3, uchar3, op>, FMXImpl_AOAOA<float4, ushort4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, ushort, float, schar, op>, FMXImpl_AOAOA<float2, ushort2, float2, char2, op>, FMXImpl_AOAOA<float3, ushort3, float3, char3, op>, FMXImpl_AOAOA<float4, ushort4, float4, char4, op>  },
                    { FMXImpl_AOAOA<float, ushort, float, ushort, op>, FMXImpl_AOAOA<float2, ushort2, float2, ushort2, op>, FMXImpl_AOAOA<float3, ushort3, float3, ushort3, op>, FMXImpl_AOAOA<float4, ushort4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, ushort, float, short, op>, FMXImpl_AOAOA<float2, ushort2, float2, short2, op>, FMXImpl_AOAOA<float3, ushort3, float3, short3, op>, FMXImpl_AOAOA<float4, ushort4, float4, short4, op>  },
                    { FMXImpl_AOAOA<float, ushort, float, int, op>, FMXImpl_AOAOA<float2, ushort2, float2, int2, op>, FMXImpl_AOAOA<float3, ushort3, float3, int3, op>, FMXImpl_AOAOA<float4, ushort4, float4, int4, op>  },
                    { FMXImpl_AOAOA<float, ushort, float, float, op>, FMXImpl_AOAOA<float2, ushort2, float2, float2, op>, FMXImpl_AOAOA<float3, ushort3, float3, float3, op>, FMXImpl_AOAOA<float4, ushort4, float4, float4, op>  },
                    { FMXImpl_AOAOA<float, ushort, float, double, op>, FMXImpl_AOAOA<float2, ushort2, float2, double2, op>, FMXImpl_AOAOA<float3, ushort3, float3, double3, op>, FMXImpl_AOAOA<float4, ushort4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, ushort, double, uchar, op>, FMXImpl_AOAOA<float2, ushort2, double2, uchar2, op>, FMXImpl_AOAOA<float3, ushort3, double3, uchar3, op>, FMXImpl_AOAOA<float4, ushort4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, ushort, double, schar, op>, FMXImpl_AOAOA<float2, ushort2, double2, char2, op>, FMXImpl_AOAOA<float3, ushort3, double3, char3, op>, FMXImpl_AOAOA<float4, ushort4, double4, char4, op>  },
                    { FMXImpl_AOAOA<float, ushort, double, ushort, op>, FMXImpl_AOAOA<float2, ushort2, double2, ushort2, op>, FMXImpl_AOAOA<float3, ushort3, double3, ushort3, op>, FMXImpl_AOAOA<float4, ushort4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, ushort, double, short, op>, FMXImpl_AOAOA<float2, ushort2, double2, short2, op>, FMXImpl_AOAOA<float3, ushort3, double3, short3, op>, FMXImpl_AOAOA<float4, ushort4, double4, short4, op>  },
                    { FMXImpl_AOAOA<float, ushort, double, int, op>, FMXImpl_AOAOA<float2, ushort2, double2, int2, op>, FMXImpl_AOAOA<float3, ushort3, double3, int3, op>, FMXImpl_AOAOA<float4, ushort4, double4, int4, op>  },
                    { FMXImpl_AOAOA<float, ushort, double, float, op>, FMXImpl_AOAOA<float2, ushort2, double2, float2, op>, FMXImpl_AOAOA<float3, ushort3, double3, float3, op>, FMXImpl_AOAOA<float4, ushort4, double4, float4, op>  },
                    { FMXImpl_AOAOA<float, ushort, double, double, op>, FMXImpl_AOAOA<float2, ushort2, double2, double2, op>, FMXImpl_AOAOA<float3, ushort3, double3, double3, op>, FMXImpl_AOAOA<float4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<float, short, uchar, uchar, op>, FMXImpl_AOAOA<float2, short2, uchar2, uchar2, op>, FMXImpl_AOAOA<float3, short3, uchar3, uchar3, op>, FMXImpl_AOAOA<float4, short4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, short, uchar, schar, op>, FMXImpl_AOAOA<float2, short2, uchar2, char2, op>, FMXImpl_AOAOA<float3, short3, uchar3, char3, op>, FMXImpl_AOAOA<float4, short4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<float, short, uchar, ushort, op>, FMXImpl_AOAOA<float2, short2, uchar2, ushort2, op>, FMXImpl_AOAOA<float3, short3, uchar3, ushort3, op>, FMXImpl_AOAOA<float4, short4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, short, uchar, short, op>, FMXImpl_AOAOA<float2, short2, uchar2, short2, op>, FMXImpl_AOAOA<float3, short3, uchar3, short3, op>, FMXImpl_AOAOA<float4, short4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<float, short, uchar, int, op>, FMXImpl_AOAOA<float2, short2, uchar2, int2, op>, FMXImpl_AOAOA<float3, short3, uchar3, int3, op>, FMXImpl_AOAOA<float4, short4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<float, short, uchar, float, op>, FMXImpl_AOAOA<float2, short2, uchar2, float2, op>, FMXImpl_AOAOA<float3, short3, uchar3, float3, op>, FMXImpl_AOAOA<float4, short4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<float, short, uchar, double, op>, FMXImpl_AOAOA<float2, short2, uchar2, double2, op>, FMXImpl_AOAOA<float3, short3, uchar3, double3, op>, FMXImpl_AOAOA<float4, short4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, short, schar, uchar, op>, FMXImpl_AOAOA<float2, short2, char2, uchar2, op>, FMXImpl_AOAOA<float3, short3, char3, uchar3, op>, FMXImpl_AOAOA<float4, short4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, short, schar, schar, op>, FMXImpl_AOAOA<float2, short2, char2, char2, op>, FMXImpl_AOAOA<float3, short3, char3, char3, op>, FMXImpl_AOAOA<float4, short4, char4, char4, op>  },
                    { FMXImpl_AOAOA<float, short, schar, ushort, op>, FMXImpl_AOAOA<float2, short2, char2, ushort2, op>, FMXImpl_AOAOA<float3, short3, char3, ushort3, op>, FMXImpl_AOAOA<float4, short4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, short, schar, short, op>, FMXImpl_AOAOA<float2, short2, char2, short2, op>, FMXImpl_AOAOA<float3, short3, char3, short3, op>, FMXImpl_AOAOA<float4, short4, char4, short4, op>  },
                    { FMXImpl_AOAOA<float, short, schar, int, op>, FMXImpl_AOAOA<float2, short2, char2, int2, op>, FMXImpl_AOAOA<float3, short3, char3, int3, op>, FMXImpl_AOAOA<float4, short4, char4, int4, op>  },
                    { FMXImpl_AOAOA<float, short, schar, float, op>, FMXImpl_AOAOA<float2, short2, char2, float2, op>, FMXImpl_AOAOA<float3, short3, char3, float3, op>, FMXImpl_AOAOA<float4, short4, char4, float4, op>  },
                    { FMXImpl_AOAOA<float, short, schar, double, op>, FMXImpl_AOAOA<float2, short2, char2, double2, op>, FMXImpl_AOAOA<float3, short3, char3, double3, op>, FMXImpl_AOAOA<float4, short4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, short, ushort, uchar, op>, FMXImpl_AOAOA<float2, short2, ushort2, uchar2, op>, FMXImpl_AOAOA<float3, short3, ushort3, uchar3, op>, FMXImpl_AOAOA<float4, short4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, short, ushort, schar, op>, FMXImpl_AOAOA<float2, short2, ushort2, char2, op>, FMXImpl_AOAOA<float3, short3, ushort3, char3, op>, FMXImpl_AOAOA<float4, short4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<float, short, ushort, ushort, op>, FMXImpl_AOAOA<float2, short2, ushort2, ushort2, op>, FMXImpl_AOAOA<float3, short3, ushort3, ushort3, op>, FMXImpl_AOAOA<float4, short4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, short, ushort, short, op>, FMXImpl_AOAOA<float2, short2, ushort2, short2, op>, FMXImpl_AOAOA<float3, short3, ushort3, short3, op>, FMXImpl_AOAOA<float4, short4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<float, short, ushort, int, op>, FMXImpl_AOAOA<float2, short2, ushort2, int2, op>, FMXImpl_AOAOA<float3, short3, ushort3, int3, op>, FMXImpl_AOAOA<float4, short4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<float, short, ushort, float, op>, FMXImpl_AOAOA<float2, short2, ushort2, float2, op>, FMXImpl_AOAOA<float3, short3, ushort3, float3, op>, FMXImpl_AOAOA<float4, short4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<float, short, ushort, double, op>, FMXImpl_AOAOA<float2, short2, ushort2, double2, op>, FMXImpl_AOAOA<float3, short3, ushort3, double3, op>, FMXImpl_AOAOA<float4, short4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, short, short, uchar, op>, FMXImpl_AOAOA<float2, short2, short2, uchar2, op>, FMXImpl_AOAOA<float3, short3, short3, uchar3, op>, FMXImpl_AOAOA<float4, short4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, short, short, schar, op>, FMXImpl_AOAOA<float2, short2, short2, char2, op>, FMXImpl_AOAOA<float3, short3, short3, char3, op>, FMXImpl_AOAOA<float4, short4, short4, char4, op>  },
                    { FMXImpl_AOAOA<float, short, short, ushort, op>, FMXImpl_AOAOA<float2, short2, short2, ushort2, op>, FMXImpl_AOAOA<float3, short3, short3, ushort3, op>, FMXImpl_AOAOA<float4, short4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, short, short, short, op>, FMXImpl_AOAOA<float2, short2, short2, short2, op>, FMXImpl_AOAOA<float3, short3, short3, short3, op>, FMXImpl_AOAOA<float4, short4, short4, short4, op>  },
                    { FMXImpl_AOAOA<float, short, short, int, op>, FMXImpl_AOAOA<float2, short2, short2, int2, op>, FMXImpl_AOAOA<float3, short3, short3, int3, op>, FMXImpl_AOAOA<float4, short4, short4, int4, op>  },
                    { FMXImpl_AOAOA<float, short, short, float, op>, FMXImpl_AOAOA<float2, short2, short2, float2, op>, FMXImpl_AOAOA<float3, short3, short3, float3, op>, FMXImpl_AOAOA<float4, short4, short4, float4, op>  },
                    { FMXImpl_AOAOA<float, short, short, double, op>, FMXImpl_AOAOA<float2, short2, short2, double2, op>, FMXImpl_AOAOA<float3, short3, short3, double3, op>, FMXImpl_AOAOA<float4, short4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, short, int, uchar, op>, FMXImpl_AOAOA<float2, short2, int2, uchar2, op>, FMXImpl_AOAOA<float3, short3, int3, uchar3, op>, FMXImpl_AOAOA<float4, short4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, short, int, schar, op>, FMXImpl_AOAOA<float2, short2, int2, char2, op>, FMXImpl_AOAOA<float3, short3, int3, char3, op>, FMXImpl_AOAOA<float4, short4, int4, char4, op>  },
                    { FMXImpl_AOAOA<float, short, int, ushort, op>, FMXImpl_AOAOA<float2, short2, int2, ushort2, op>, FMXImpl_AOAOA<float3, short3, int3, ushort3, op>, FMXImpl_AOAOA<float4, short4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, short, int, short, op>, FMXImpl_AOAOA<float2, short2, int2, short2, op>, FMXImpl_AOAOA<float3, short3, int3, short3, op>, FMXImpl_AOAOA<float4, short4, int4, short4, op>  },
                    { FMXImpl_AOAOA<float, short, int, int, op>, FMXImpl_AOAOA<float2, short2, int2, int2, op>, FMXImpl_AOAOA<float3, short3, int3, int3, op>, FMXImpl_AOAOA<float4, short4, int4, int4, op>  },
                    { FMXImpl_AOAOA<float, short, int, float, op>, FMXImpl_AOAOA<float2, short2, int2, float2, op>, FMXImpl_AOAOA<float3, short3, int3, float3, op>, FMXImpl_AOAOA<float4, short4, int4, float4, op>  },
                    { FMXImpl_AOAOA<float, short, int, double, op>, FMXImpl_AOAOA<float2, short2, int2, double2, op>, FMXImpl_AOAOA<float3, short3, int3, double3, op>, FMXImpl_AOAOA<float4, short4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, short, float, uchar, op>, FMXImpl_AOAOA<float2, short2, float2, uchar2, op>, FMXImpl_AOAOA<float3, short3, float3, uchar3, op>, FMXImpl_AOAOA<float4, short4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, short, float, schar, op>, FMXImpl_AOAOA<float2, short2, float2, char2, op>, FMXImpl_AOAOA<float3, short3, float3, char3, op>, FMXImpl_AOAOA<float4, short4, float4, char4, op>  },
                    { FMXImpl_AOAOA<float, short, float, ushort, op>, FMXImpl_AOAOA<float2, short2, float2, ushort2, op>, FMXImpl_AOAOA<float3, short3, float3, ushort3, op>, FMXImpl_AOAOA<float4, short4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, short, float, short, op>, FMXImpl_AOAOA<float2, short2, float2, short2, op>, FMXImpl_AOAOA<float3, short3, float3, short3, op>, FMXImpl_AOAOA<float4, short4, float4, short4, op>  },
                    { FMXImpl_AOAOA<float, short, float, int, op>, FMXImpl_AOAOA<float2, short2, float2, int2, op>, FMXImpl_AOAOA<float3, short3, float3, int3, op>, FMXImpl_AOAOA<float4, short4, float4, int4, op>  },
                    { FMXImpl_AOAOA<float, short, float, float, op>, FMXImpl_AOAOA<float2, short2, float2, float2, op>, FMXImpl_AOAOA<float3, short3, float3, float3, op>, FMXImpl_AOAOA<float4, short4, float4, float4, op>  },
                    { FMXImpl_AOAOA<float, short, float, double, op>, FMXImpl_AOAOA<float2, short2, float2, double2, op>, FMXImpl_AOAOA<float3, short3, float3, double3, op>, FMXImpl_AOAOA<float4, short4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, short, double, uchar, op>, FMXImpl_AOAOA<float2, short2, double2, uchar2, op>, FMXImpl_AOAOA<float3, short3, double3, uchar3, op>, FMXImpl_AOAOA<float4, short4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, short, double, schar, op>, FMXImpl_AOAOA<float2, short2, double2, char2, op>, FMXImpl_AOAOA<float3, short3, double3, char3, op>, FMXImpl_AOAOA<float4, short4, double4, char4, op>  },
                    { FMXImpl_AOAOA<float, short, double, ushort, op>, FMXImpl_AOAOA<float2, short2, double2, ushort2, op>, FMXImpl_AOAOA<float3, short3, double3, ushort3, op>, FMXImpl_AOAOA<float4, short4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, short, double, short, op>, FMXImpl_AOAOA<float2, short2, double2, short2, op>, FMXImpl_AOAOA<float3, short3, double3, short3, op>, FMXImpl_AOAOA<float4, short4, double4, short4, op>  },
                    { FMXImpl_AOAOA<float, short, double, int, op>, FMXImpl_AOAOA<float2, short2, double2, int2, op>, FMXImpl_AOAOA<float3, short3, double3, int3, op>, FMXImpl_AOAOA<float4, short4, double4, int4, op>  },
                    { FMXImpl_AOAOA<float, short, double, float, op>, FMXImpl_AOAOA<float2, short2, double2, float2, op>, FMXImpl_AOAOA<float3, short3, double3, float3, op>, FMXImpl_AOAOA<float4, short4, double4, float4, op>  },
                    { FMXImpl_AOAOA<float, short, double, double, op>, FMXImpl_AOAOA<float2, short2, double2, double2, op>, FMXImpl_AOAOA<float3, short3, double3, double3, op>, FMXImpl_AOAOA<float4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<float, int, uchar, uchar, op>, FMXImpl_AOAOA<float2, int2, uchar2, uchar2, op>, FMXImpl_AOAOA<float3, int3, uchar3, uchar3, op>, FMXImpl_AOAOA<float4, int4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, int, uchar, schar, op>, FMXImpl_AOAOA<float2, int2, uchar2, char2, op>, FMXImpl_AOAOA<float3, int3, uchar3, char3, op>, FMXImpl_AOAOA<float4, int4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<float, int, uchar, ushort, op>, FMXImpl_AOAOA<float2, int2, uchar2, ushort2, op>, FMXImpl_AOAOA<float3, int3, uchar3, ushort3, op>, FMXImpl_AOAOA<float4, int4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, int, uchar, short, op>, FMXImpl_AOAOA<float2, int2, uchar2, short2, op>, FMXImpl_AOAOA<float3, int3, uchar3, short3, op>, FMXImpl_AOAOA<float4, int4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<float, int, uchar, int, op>, FMXImpl_AOAOA<float2, int2, uchar2, int2, op>, FMXImpl_AOAOA<float3, int3, uchar3, int3, op>, FMXImpl_AOAOA<float4, int4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<float, int, uchar, float, op>, FMXImpl_AOAOA<float2, int2, uchar2, float2, op>, FMXImpl_AOAOA<float3, int3, uchar3, float3, op>, FMXImpl_AOAOA<float4, int4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<float, int, uchar, double, op>, FMXImpl_AOAOA<float2, int2, uchar2, double2, op>, FMXImpl_AOAOA<float3, int3, uchar3, double3, op>, FMXImpl_AOAOA<float4, int4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, int, schar, uchar, op>, FMXImpl_AOAOA<float2, int2, char2, uchar2, op>, FMXImpl_AOAOA<float3, int3, char3, uchar3, op>, FMXImpl_AOAOA<float4, int4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, int, schar, schar, op>, FMXImpl_AOAOA<float2, int2, char2, char2, op>, FMXImpl_AOAOA<float3, int3, char3, char3, op>, FMXImpl_AOAOA<float4, int4, char4, char4, op>  },
                    { FMXImpl_AOAOA<float, int, schar, ushort, op>, FMXImpl_AOAOA<float2, int2, char2, ushort2, op>, FMXImpl_AOAOA<float3, int3, char3, ushort3, op>, FMXImpl_AOAOA<float4, int4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, int, schar, short, op>, FMXImpl_AOAOA<float2, int2, char2, short2, op>, FMXImpl_AOAOA<float3, int3, char3, short3, op>, FMXImpl_AOAOA<float4, int4, char4, short4, op>  },
                    { FMXImpl_AOAOA<float, int, schar, int, op>, FMXImpl_AOAOA<float2, int2, char2, int2, op>, FMXImpl_AOAOA<float3, int3, char3, int3, op>, FMXImpl_AOAOA<float4, int4, char4, int4, op>  },
                    { FMXImpl_AOAOA<float, int, schar, float, op>, FMXImpl_AOAOA<float2, int2, char2, float2, op>, FMXImpl_AOAOA<float3, int3, char3, float3, op>, FMXImpl_AOAOA<float4, int4, char4, float4, op>  },
                    { FMXImpl_AOAOA<float, int, schar, double, op>, FMXImpl_AOAOA<float2, int2, char2, double2, op>, FMXImpl_AOAOA<float3, int3, char3, double3, op>, FMXImpl_AOAOA<float4, int4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, int, ushort, uchar, op>, FMXImpl_AOAOA<float2, int2, ushort2, uchar2, op>, FMXImpl_AOAOA<float3, int3, ushort3, uchar3, op>, FMXImpl_AOAOA<float4, int4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, int, ushort, schar, op>, FMXImpl_AOAOA<float2, int2, ushort2, char2, op>, FMXImpl_AOAOA<float3, int3, ushort3, char3, op>, FMXImpl_AOAOA<float4, int4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<float, int, ushort, ushort, op>, FMXImpl_AOAOA<float2, int2, ushort2, ushort2, op>, FMXImpl_AOAOA<float3, int3, ushort3, ushort3, op>, FMXImpl_AOAOA<float4, int4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, int, ushort, short, op>, FMXImpl_AOAOA<float2, int2, ushort2, short2, op>, FMXImpl_AOAOA<float3, int3, ushort3, short3, op>, FMXImpl_AOAOA<float4, int4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<float, int, ushort, int, op>, FMXImpl_AOAOA<float2, int2, ushort2, int2, op>, FMXImpl_AOAOA<float3, int3, ushort3, int3, op>, FMXImpl_AOAOA<float4, int4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<float, int, ushort, float, op>, FMXImpl_AOAOA<float2, int2, ushort2, float2, op>, FMXImpl_AOAOA<float3, int3, ushort3, float3, op>, FMXImpl_AOAOA<float4, int4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<float, int, ushort, double, op>, FMXImpl_AOAOA<float2, int2, ushort2, double2, op>, FMXImpl_AOAOA<float3, int3, ushort3, double3, op>, FMXImpl_AOAOA<float4, int4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, int, short, uchar, op>, FMXImpl_AOAOA<float2, int2, short2, uchar2, op>, FMXImpl_AOAOA<float3, int3, short3, uchar3, op>, FMXImpl_AOAOA<float4, int4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, int, short, schar, op>, FMXImpl_AOAOA<float2, int2, short2, char2, op>, FMXImpl_AOAOA<float3, int3, short3, char3, op>, FMXImpl_AOAOA<float4, int4, short4, char4, op>  },
                    { FMXImpl_AOAOA<float, int, short, ushort, op>, FMXImpl_AOAOA<float2, int2, short2, ushort2, op>, FMXImpl_AOAOA<float3, int3, short3, ushort3, op>, FMXImpl_AOAOA<float4, int4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, int, short, short, op>, FMXImpl_AOAOA<float2, int2, short2, short2, op>, FMXImpl_AOAOA<float3, int3, short3, short3, op>, FMXImpl_AOAOA<float4, int4, short4, short4, op>  },
                    { FMXImpl_AOAOA<float, int, short, int, op>, FMXImpl_AOAOA<float2, int2, short2, int2, op>, FMXImpl_AOAOA<float3, int3, short3, int3, op>, FMXImpl_AOAOA<float4, int4, short4, int4, op>  },
                    { FMXImpl_AOAOA<float, int, short, float, op>, FMXImpl_AOAOA<float2, int2, short2, float2, op>, FMXImpl_AOAOA<float3, int3, short3, float3, op>, FMXImpl_AOAOA<float4, int4, short4, float4, op>  },
                    { FMXImpl_AOAOA<float, int, short, double, op>, FMXImpl_AOAOA<float2, int2, short2, double2, op>, FMXImpl_AOAOA<float3, int3, short3, double3, op>, FMXImpl_AOAOA<float4, int4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, int, int, uchar, op>, FMXImpl_AOAOA<float2, int2, int2, uchar2, op>, FMXImpl_AOAOA<float3, int3, int3, uchar3, op>, FMXImpl_AOAOA<float4, int4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, int, int, schar, op>, FMXImpl_AOAOA<float2, int2, int2, char2, op>, FMXImpl_AOAOA<float3, int3, int3, char3, op>, FMXImpl_AOAOA<float4, int4, int4, char4, op>  },
                    { FMXImpl_AOAOA<float, int, int, ushort, op>, FMXImpl_AOAOA<float2, int2, int2, ushort2, op>, FMXImpl_AOAOA<float3, int3, int3, ushort3, op>, FMXImpl_AOAOA<float4, int4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, int, int, short, op>, FMXImpl_AOAOA<float2, int2, int2, short2, op>, FMXImpl_AOAOA<float3, int3, int3, short3, op>, FMXImpl_AOAOA<float4, int4, int4, short4, op>  },
                    { FMXImpl_AOAOA<float, int, int, int, op>, FMXImpl_AOAOA<float2, int2, int2, int2, op>, FMXImpl_AOAOA<float3, int3, int3, int3, op>, FMXImpl_AOAOA<float4, int4, int4, int4, op>  },
                    { FMXImpl_AOAOA<float, int, int, float, op>, FMXImpl_AOAOA<float2, int2, int2, float2, op>, FMXImpl_AOAOA<float3, int3, int3, float3, op>, FMXImpl_AOAOA<float4, int4, int4, float4, op>  },
                    { FMXImpl_AOAOA<float, int, int, double, op>, FMXImpl_AOAOA<float2, int2, int2, double2, op>, FMXImpl_AOAOA<float3, int3, int3, double3, op>, FMXImpl_AOAOA<float4, int4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, int, float, uchar, op>, FMXImpl_AOAOA<float2, int2, float2, uchar2, op>, FMXImpl_AOAOA<float3, int3, float3, uchar3, op>, FMXImpl_AOAOA<float4, int4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, int, float, schar, op>, FMXImpl_AOAOA<float2, int2, float2, char2, op>, FMXImpl_AOAOA<float3, int3, float3, char3, op>, FMXImpl_AOAOA<float4, int4, float4, char4, op>  },
                    { FMXImpl_AOAOA<float, int, float, ushort, op>, FMXImpl_AOAOA<float2, int2, float2, ushort2, op>, FMXImpl_AOAOA<float3, int3, float3, ushort3, op>, FMXImpl_AOAOA<float4, int4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, int, float, short, op>, FMXImpl_AOAOA<float2, int2, float2, short2, op>, FMXImpl_AOAOA<float3, int3, float3, short3, op>, FMXImpl_AOAOA<float4, int4, float4, short4, op>  },
                    { FMXImpl_AOAOA<float, int, float, int, op>, FMXImpl_AOAOA<float2, int2, float2, int2, op>, FMXImpl_AOAOA<float3, int3, float3, int3, op>, FMXImpl_AOAOA<float4, int4, float4, int4, op>  },
                    { FMXImpl_AOAOA<float, int, float, float, op>, FMXImpl_AOAOA<float2, int2, float2, float2, op>, FMXImpl_AOAOA<float3, int3, float3, float3, op>, FMXImpl_AOAOA<float4, int4, float4, float4, op>  },
                    { FMXImpl_AOAOA<float, int, float, double, op>, FMXImpl_AOAOA<float2, int2, float2, double2, op>, FMXImpl_AOAOA<float3, int3, float3, double3, op>, FMXImpl_AOAOA<float4, int4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, int, double, uchar, op>, FMXImpl_AOAOA<float2, int2, double2, uchar2, op>, FMXImpl_AOAOA<float3, int3, double3, uchar3, op>, FMXImpl_AOAOA<float4, int4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, int, double, schar, op>, FMXImpl_AOAOA<float2, int2, double2, char2, op>, FMXImpl_AOAOA<float3, int3, double3, char3, op>, FMXImpl_AOAOA<float4, int4, double4, char4, op>  },
                    { FMXImpl_AOAOA<float, int, double, ushort, op>, FMXImpl_AOAOA<float2, int2, double2, ushort2, op>, FMXImpl_AOAOA<float3, int3, double3, ushort3, op>, FMXImpl_AOAOA<float4, int4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, int, double, short, op>, FMXImpl_AOAOA<float2, int2, double2, short2, op>, FMXImpl_AOAOA<float3, int3, double3, short3, op>, FMXImpl_AOAOA<float4, int4, double4, short4, op>  },
                    { FMXImpl_AOAOA<float, int, double, int, op>, FMXImpl_AOAOA<float2, int2, double2, int2, op>, FMXImpl_AOAOA<float3, int3, double3, int3, op>, FMXImpl_AOAOA<float4, int4, double4, int4, op>  },
                    { FMXImpl_AOAOA<float, int, double, float, op>, FMXImpl_AOAOA<float2, int2, double2, float2, op>, FMXImpl_AOAOA<float3, int3, double3, float3, op>, FMXImpl_AOAOA<float4, int4, double4, float4, op>  },
                    { FMXImpl_AOAOA<float, int, double, double, op>, FMXImpl_AOAOA<float2, int2, double2, double2, op>, FMXImpl_AOAOA<float3, int3, double3, double3, op>, FMXImpl_AOAOA<float4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<float, float, uchar, uchar, op>, FMXImpl_AOAOA<float2, float2, uchar2, uchar2, op>, FMXImpl_AOAOA<float3, float3, uchar3, uchar3, op>, FMXImpl_AOAOA<float4, float4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, float, uchar, schar, op>, FMXImpl_AOAOA<float2, float2, uchar2, char2, op>, FMXImpl_AOAOA<float3, float3, uchar3, char3, op>, FMXImpl_AOAOA<float4, float4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<float, float, uchar, ushort, op>, FMXImpl_AOAOA<float2, float2, uchar2, ushort2, op>, FMXImpl_AOAOA<float3, float3, uchar3, ushort3, op>, FMXImpl_AOAOA<float4, float4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, float, uchar, short, op>, FMXImpl_AOAOA<float2, float2, uchar2, short2, op>, FMXImpl_AOAOA<float3, float3, uchar3, short3, op>, FMXImpl_AOAOA<float4, float4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<float, float, uchar, int, op>, FMXImpl_AOAOA<float2, float2, uchar2, int2, op>, FMXImpl_AOAOA<float3, float3, uchar3, int3, op>, FMXImpl_AOAOA<float4, float4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<float, float, uchar, float, op>, FMXImpl_AOAOA<float2, float2, uchar2, float2, op>, FMXImpl_AOAOA<float3, float3, uchar3, float3, op>, FMXImpl_AOAOA<float4, float4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<float, float, uchar, double, op>, FMXImpl_AOAOA<float2, float2, uchar2, double2, op>, FMXImpl_AOAOA<float3, float3, uchar3, double3, op>, FMXImpl_AOAOA<float4, float4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, float, schar, uchar, op>, FMXImpl_AOAOA<float2, float2, char2, uchar2, op>, FMXImpl_AOAOA<float3, float3, char3, uchar3, op>, FMXImpl_AOAOA<float4, float4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, float, schar, schar, op>, FMXImpl_AOAOA<float2, float2, char2, char2, op>, FMXImpl_AOAOA<float3, float3, char3, char3, op>, FMXImpl_AOAOA<float4, float4, char4, char4, op>  },
                    { FMXImpl_AOAOA<float, float, schar, ushort, op>, FMXImpl_AOAOA<float2, float2, char2, ushort2, op>, FMXImpl_AOAOA<float3, float3, char3, ushort3, op>, FMXImpl_AOAOA<float4, float4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, float, schar, short, op>, FMXImpl_AOAOA<float2, float2, char2, short2, op>, FMXImpl_AOAOA<float3, float3, char3, short3, op>, FMXImpl_AOAOA<float4, float4, char4, short4, op>  },
                    { FMXImpl_AOAOA<float, float, schar, int, op>, FMXImpl_AOAOA<float2, float2, char2, int2, op>, FMXImpl_AOAOA<float3, float3, char3, int3, op>, FMXImpl_AOAOA<float4, float4, char4, int4, op>  },
                    { FMXImpl_AOAOA<float, float, schar, float, op>, FMXImpl_AOAOA<float2, float2, char2, float2, op>, FMXImpl_AOAOA<float3, float3, char3, float3, op>, FMXImpl_AOAOA<float4, float4, char4, float4, op>  },
                    { FMXImpl_AOAOA<float, float, schar, double, op>, FMXImpl_AOAOA<float2, float2, char2, double2, op>, FMXImpl_AOAOA<float3, float3, char3, double3, op>, FMXImpl_AOAOA<float4, float4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, float, ushort, uchar, op>, FMXImpl_AOAOA<float2, float2, ushort2, uchar2, op>, FMXImpl_AOAOA<float3, float3, ushort3, uchar3, op>, FMXImpl_AOAOA<float4, float4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, float, ushort, schar, op>, FMXImpl_AOAOA<float2, float2, ushort2, char2, op>, FMXImpl_AOAOA<float3, float3, ushort3, char3, op>, FMXImpl_AOAOA<float4, float4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<float, float, ushort, ushort, op>, FMXImpl_AOAOA<float2, float2, ushort2, ushort2, op>, FMXImpl_AOAOA<float3, float3, ushort3, ushort3, op>, FMXImpl_AOAOA<float4, float4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, float, ushort, short, op>, FMXImpl_AOAOA<float2, float2, ushort2, short2, op>, FMXImpl_AOAOA<float3, float3, ushort3, short3, op>, FMXImpl_AOAOA<float4, float4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<float, float, ushort, int, op>, FMXImpl_AOAOA<float2, float2, ushort2, int2, op>, FMXImpl_AOAOA<float3, float3, ushort3, int3, op>, FMXImpl_AOAOA<float4, float4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<float, float, ushort, float, op>, FMXImpl_AOAOA<float2, float2, ushort2, float2, op>, FMXImpl_AOAOA<float3, float3, ushort3, float3, op>, FMXImpl_AOAOA<float4, float4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<float, float, ushort, double, op>, FMXImpl_AOAOA<float2, float2, ushort2, double2, op>, FMXImpl_AOAOA<float3, float3, ushort3, double3, op>, FMXImpl_AOAOA<float4, float4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, float, short, uchar, op>, FMXImpl_AOAOA<float2, float2, short2, uchar2, op>, FMXImpl_AOAOA<float3, float3, short3, uchar3, op>, FMXImpl_AOAOA<float4, float4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, float, short, schar, op>, FMXImpl_AOAOA<float2, float2, short2, char2, op>, FMXImpl_AOAOA<float3, float3, short3, char3, op>, FMXImpl_AOAOA<float4, float4, short4, char4, op>  },
                    { FMXImpl_AOAOA<float, float, short, ushort, op>, FMXImpl_AOAOA<float2, float2, short2, ushort2, op>, FMXImpl_AOAOA<float3, float3, short3, ushort3, op>, FMXImpl_AOAOA<float4, float4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, float, short, short, op>, FMXImpl_AOAOA<float2, float2, short2, short2, op>, FMXImpl_AOAOA<float3, float3, short3, short3, op>, FMXImpl_AOAOA<float4, float4, short4, short4, op>  },
                    { FMXImpl_AOAOA<float, float, short, int, op>, FMXImpl_AOAOA<float2, float2, short2, int2, op>, FMXImpl_AOAOA<float3, float3, short3, int3, op>, FMXImpl_AOAOA<float4, float4, short4, int4, op>  },
                    { FMXImpl_AOAOA<float, float, short, float, op>, FMXImpl_AOAOA<float2, float2, short2, float2, op>, FMXImpl_AOAOA<float3, float3, short3, float3, op>, FMXImpl_AOAOA<float4, float4, short4, float4, op>  },
                    { FMXImpl_AOAOA<float, float, short, double, op>, FMXImpl_AOAOA<float2, float2, short2, double2, op>, FMXImpl_AOAOA<float3, float3, short3, double3, op>, FMXImpl_AOAOA<float4, float4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, float, int, uchar, op>, FMXImpl_AOAOA<float2, float2, int2, uchar2, op>, FMXImpl_AOAOA<float3, float3, int3, uchar3, op>, FMXImpl_AOAOA<float4, float4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, float, int, schar, op>, FMXImpl_AOAOA<float2, float2, int2, char2, op>, FMXImpl_AOAOA<float3, float3, int3, char3, op>, FMXImpl_AOAOA<float4, float4, int4, char4, op>  },
                    { FMXImpl_AOAOA<float, float, int, ushort, op>, FMXImpl_AOAOA<float2, float2, int2, ushort2, op>, FMXImpl_AOAOA<float3, float3, int3, ushort3, op>, FMXImpl_AOAOA<float4, float4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, float, int, short, op>, FMXImpl_AOAOA<float2, float2, int2, short2, op>, FMXImpl_AOAOA<float3, float3, int3, short3, op>, FMXImpl_AOAOA<float4, float4, int4, short4, op>  },
                    { FMXImpl_AOAOA<float, float, int, int, op>, FMXImpl_AOAOA<float2, float2, int2, int2, op>, FMXImpl_AOAOA<float3, float3, int3, int3, op>, FMXImpl_AOAOA<float4, float4, int4, int4, op>  },
                    { FMXImpl_AOAOA<float, float, int, float, op>, FMXImpl_AOAOA<float2, float2, int2, float2, op>, FMXImpl_AOAOA<float3, float3, int3, float3, op>, FMXImpl_AOAOA<float4, float4, int4, float4, op>  },
                    { FMXImpl_AOAOA<float, float, int, double, op>, FMXImpl_AOAOA<float2, float2, int2, double2, op>, FMXImpl_AOAOA<float3, float3, int3, double3, op>, FMXImpl_AOAOA<float4, float4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, float, float, uchar, op>, FMXImpl_AOAOA<float2, float2, float2, uchar2, op>, FMXImpl_AOAOA<float3, float3, float3, uchar3, op>, FMXImpl_AOAOA<float4, float4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, float, float, schar, op>, FMXImpl_AOAOA<float2, float2, float2, char2, op>, FMXImpl_AOAOA<float3, float3, float3, char3, op>, FMXImpl_AOAOA<float4, float4, float4, char4, op>  },
                    { FMXImpl_AOAOA<float, float, float, ushort, op>, FMXImpl_AOAOA<float2, float2, float2, ushort2, op>, FMXImpl_AOAOA<float3, float3, float3, ushort3, op>, FMXImpl_AOAOA<float4, float4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, float, float, short, op>, FMXImpl_AOAOA<float2, float2, float2, short2, op>, FMXImpl_AOAOA<float3, float3, float3, short3, op>, FMXImpl_AOAOA<float4, float4, float4, short4, op>  },
                    { FMXImpl_AOAOA<float, float, float, int, op>, FMXImpl_AOAOA<float2, float2, float2, int2, op>, FMXImpl_AOAOA<float3, float3, float3, int3, op>, FMXImpl_AOAOA<float4, float4, float4, int4, op>  },
                    { FMXImpl_AOAOA<float, float, float, float, op>, FMXImpl_AOAOA<float2, float2, float2, float2, op>, FMXImpl_AOAOA<float3, float3, float3, float3, op>, FMXImpl_AOAOA<float4, float4, float4, float4, op>  },
                    { FMXImpl_AOAOA<float, float, float, double, op>, FMXImpl_AOAOA<float2, float2, float2, double2, op>, FMXImpl_AOAOA<float3, float3, float3, double3, op>, FMXImpl_AOAOA<float4, float4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, float, double, uchar, op>, FMXImpl_AOAOA<float2, float2, double2, uchar2, op>, FMXImpl_AOAOA<float3, float3, double3, uchar3, op>, FMXImpl_AOAOA<float4, float4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, float, double, schar, op>, FMXImpl_AOAOA<float2, float2, double2, char2, op>, FMXImpl_AOAOA<float3, float3, double3, char3, op>, FMXImpl_AOAOA<float4, float4, double4, char4, op>  },
                    { FMXImpl_AOAOA<float, float, double, ushort, op>, FMXImpl_AOAOA<float2, float2, double2, ushort2, op>, FMXImpl_AOAOA<float3, float3, double3, ushort3, op>, FMXImpl_AOAOA<float4, float4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, float, double, short, op>, FMXImpl_AOAOA<float2, float2, double2, short2, op>, FMXImpl_AOAOA<float3, float3, double3, short3, op>, FMXImpl_AOAOA<float4, float4, double4, short4, op>  },
                    { FMXImpl_AOAOA<float, float, double, int, op>, FMXImpl_AOAOA<float2, float2, double2, int2, op>, FMXImpl_AOAOA<float3, float3, double3, int3, op>, FMXImpl_AOAOA<float4, float4, double4, int4, op>  },
                    { FMXImpl_AOAOA<float, float, double, float, op>, FMXImpl_AOAOA<float2, float2, double2, float2, op>, FMXImpl_AOAOA<float3, float3, double3, float3, op>, FMXImpl_AOAOA<float4, float4, double4, float4, op>  },
                    { FMXImpl_AOAOA<float, float, double, double, op>, FMXImpl_AOAOA<float2, float2, double2, double2, op>, FMXImpl_AOAOA<float3, float3, double3, double3, op>, FMXImpl_AOAOA<float4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<float, double, uchar, uchar, op>, FMXImpl_AOAOA<float2, double2, uchar2, uchar2, op>, FMXImpl_AOAOA<float3, double3, uchar3, uchar3, op>, FMXImpl_AOAOA<float4, double4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, double, uchar, schar, op>, FMXImpl_AOAOA<float2, double2, uchar2, char2, op>, FMXImpl_AOAOA<float3, double3, uchar3, char3, op>, FMXImpl_AOAOA<float4, double4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<float, double, uchar, ushort, op>, FMXImpl_AOAOA<float2, double2, uchar2, ushort2, op>, FMXImpl_AOAOA<float3, double3, uchar3, ushort3, op>, FMXImpl_AOAOA<float4, double4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, double, uchar, short, op>, FMXImpl_AOAOA<float2, double2, uchar2, short2, op>, FMXImpl_AOAOA<float3, double3, uchar3, short3, op>, FMXImpl_AOAOA<float4, double4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<float, double, uchar, int, op>, FMXImpl_AOAOA<float2, double2, uchar2, int2, op>, FMXImpl_AOAOA<float3, double3, uchar3, int3, op>, FMXImpl_AOAOA<float4, double4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<float, double, uchar, float, op>, FMXImpl_AOAOA<float2, double2, uchar2, float2, op>, FMXImpl_AOAOA<float3, double3, uchar3, float3, op>, FMXImpl_AOAOA<float4, double4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<float, double, uchar, double, op>, FMXImpl_AOAOA<float2, double2, uchar2, double2, op>, FMXImpl_AOAOA<float3, double3, uchar3, double3, op>, FMXImpl_AOAOA<float4, double4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, double, schar, uchar, op>, FMXImpl_AOAOA<float2, double2, char2, uchar2, op>, FMXImpl_AOAOA<float3, double3, char3, uchar3, op>, FMXImpl_AOAOA<float4, double4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, double, schar, schar, op>, FMXImpl_AOAOA<float2, double2, char2, char2, op>, FMXImpl_AOAOA<float3, double3, char3, char3, op>, FMXImpl_AOAOA<float4, double4, char4, char4, op>  },
                    { FMXImpl_AOAOA<float, double, schar, ushort, op>, FMXImpl_AOAOA<float2, double2, char2, ushort2, op>, FMXImpl_AOAOA<float3, double3, char3, ushort3, op>, FMXImpl_AOAOA<float4, double4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, double, schar, short, op>, FMXImpl_AOAOA<float2, double2, char2, short2, op>, FMXImpl_AOAOA<float3, double3, char3, short3, op>, FMXImpl_AOAOA<float4, double4, char4, short4, op>  },
                    { FMXImpl_AOAOA<float, double, schar, int, op>, FMXImpl_AOAOA<float2, double2, char2, int2, op>, FMXImpl_AOAOA<float3, double3, char3, int3, op>, FMXImpl_AOAOA<float4, double4, char4, int4, op>  },
                    { FMXImpl_AOAOA<float, double, schar, float, op>, FMXImpl_AOAOA<float2, double2, char2, float2, op>, FMXImpl_AOAOA<float3, double3, char3, float3, op>, FMXImpl_AOAOA<float4, double4, char4, float4, op>  },
                    { FMXImpl_AOAOA<float, double, schar, double, op>, FMXImpl_AOAOA<float2, double2, char2, double2, op>, FMXImpl_AOAOA<float3, double3, char3, double3, op>, FMXImpl_AOAOA<float4, double4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, double, ushort, uchar, op>, FMXImpl_AOAOA<float2, double2, ushort2, uchar2, op>, FMXImpl_AOAOA<float3, double3, ushort3, uchar3, op>, FMXImpl_AOAOA<float4, double4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, double, ushort, schar, op>, FMXImpl_AOAOA<float2, double2, ushort2, char2, op>, FMXImpl_AOAOA<float3, double3, ushort3, char3, op>, FMXImpl_AOAOA<float4, double4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<float, double, ushort, ushort, op>, FMXImpl_AOAOA<float2, double2, ushort2, ushort2, op>, FMXImpl_AOAOA<float3, double3, ushort3, ushort3, op>, FMXImpl_AOAOA<float4, double4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, double, ushort, short, op>, FMXImpl_AOAOA<float2, double2, ushort2, short2, op>, FMXImpl_AOAOA<float3, double3, ushort3, short3, op>, FMXImpl_AOAOA<float4, double4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<float, double, ushort, int, op>, FMXImpl_AOAOA<float2, double2, ushort2, int2, op>, FMXImpl_AOAOA<float3, double3, ushort3, int3, op>, FMXImpl_AOAOA<float4, double4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<float, double, ushort, float, op>, FMXImpl_AOAOA<float2, double2, ushort2, float2, op>, FMXImpl_AOAOA<float3, double3, ushort3, float3, op>, FMXImpl_AOAOA<float4, double4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<float, double, ushort, double, op>, FMXImpl_AOAOA<float2, double2, ushort2, double2, op>, FMXImpl_AOAOA<float3, double3, ushort3, double3, op>, FMXImpl_AOAOA<float4, double4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, double, short, uchar, op>, FMXImpl_AOAOA<float2, double2, short2, uchar2, op>, FMXImpl_AOAOA<float3, double3, short3, uchar3, op>, FMXImpl_AOAOA<float4, double4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, double, short, schar, op>, FMXImpl_AOAOA<float2, double2, short2, char2, op>, FMXImpl_AOAOA<float3, double3, short3, char3, op>, FMXImpl_AOAOA<float4, double4, short4, char4, op>  },
                    { FMXImpl_AOAOA<float, double, short, ushort, op>, FMXImpl_AOAOA<float2, double2, short2, ushort2, op>, FMXImpl_AOAOA<float3, double3, short3, ushort3, op>, FMXImpl_AOAOA<float4, double4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, double, short, short, op>, FMXImpl_AOAOA<float2, double2, short2, short2, op>, FMXImpl_AOAOA<float3, double3, short3, short3, op>, FMXImpl_AOAOA<float4, double4, short4, short4, op>  },
                    { FMXImpl_AOAOA<float, double, short, int, op>, FMXImpl_AOAOA<float2, double2, short2, int2, op>, FMXImpl_AOAOA<float3, double3, short3, int3, op>, FMXImpl_AOAOA<float4, double4, short4, int4, op>  },
                    { FMXImpl_AOAOA<float, double, short, float, op>, FMXImpl_AOAOA<float2, double2, short2, float2, op>, FMXImpl_AOAOA<float3, double3, short3, float3, op>, FMXImpl_AOAOA<float4, double4, short4, float4, op>  },
                    { FMXImpl_AOAOA<float, double, short, double, op>, FMXImpl_AOAOA<float2, double2, short2, double2, op>, FMXImpl_AOAOA<float3, double3, short3, double3, op>, FMXImpl_AOAOA<float4, double4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, double, int, uchar, op>, FMXImpl_AOAOA<float2, double2, int2, uchar2, op>, FMXImpl_AOAOA<float3, double3, int3, uchar3, op>, FMXImpl_AOAOA<float4, double4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, double, int, schar, op>, FMXImpl_AOAOA<float2, double2, int2, char2, op>, FMXImpl_AOAOA<float3, double3, int3, char3, op>, FMXImpl_AOAOA<float4, double4, int4, char4, op>  },
                    { FMXImpl_AOAOA<float, double, int, ushort, op>, FMXImpl_AOAOA<float2, double2, int2, ushort2, op>, FMXImpl_AOAOA<float3, double3, int3, ushort3, op>, FMXImpl_AOAOA<float4, double4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, double, int, short, op>, FMXImpl_AOAOA<float2, double2, int2, short2, op>, FMXImpl_AOAOA<float3, double3, int3, short3, op>, FMXImpl_AOAOA<float4, double4, int4, short4, op>  },
                    { FMXImpl_AOAOA<float, double, int, int, op>, FMXImpl_AOAOA<float2, double2, int2, int2, op>, FMXImpl_AOAOA<float3, double3, int3, int3, op>, FMXImpl_AOAOA<float4, double4, int4, int4, op>  },
                    { FMXImpl_AOAOA<float, double, int, float, op>, FMXImpl_AOAOA<float2, double2, int2, float2, op>, FMXImpl_AOAOA<float3, double3, int3, float3, op>, FMXImpl_AOAOA<float4, double4, int4, float4, op>  },
                    { FMXImpl_AOAOA<float, double, int, double, op>, FMXImpl_AOAOA<float2, double2, int2, double2, op>, FMXImpl_AOAOA<float3, double3, int3, double3, op>, FMXImpl_AOAOA<float4, double4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, double, float, uchar, op>, FMXImpl_AOAOA<float2, double2, float2, uchar2, op>, FMXImpl_AOAOA<float3, double3, float3, uchar3, op>, FMXImpl_AOAOA<float4, double4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, double, float, schar, op>, FMXImpl_AOAOA<float2, double2, float2, char2, op>, FMXImpl_AOAOA<float3, double3, float3, char3, op>, FMXImpl_AOAOA<float4, double4, float4, char4, op>  },
                    { FMXImpl_AOAOA<float, double, float, ushort, op>, FMXImpl_AOAOA<float2, double2, float2, ushort2, op>, FMXImpl_AOAOA<float3, double3, float3, ushort3, op>, FMXImpl_AOAOA<float4, double4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, double, float, short, op>, FMXImpl_AOAOA<float2, double2, float2, short2, op>, FMXImpl_AOAOA<float3, double3, float3, short3, op>, FMXImpl_AOAOA<float4, double4, float4, short4, op>  },
                    { FMXImpl_AOAOA<float, double, float, int, op>, FMXImpl_AOAOA<float2, double2, float2, int2, op>, FMXImpl_AOAOA<float3, double3, float3, int3, op>, FMXImpl_AOAOA<float4, double4, float4, int4, op>  },
                    { FMXImpl_AOAOA<float, double, float, float, op>, FMXImpl_AOAOA<float2, double2, float2, float2, op>, FMXImpl_AOAOA<float3, double3, float3, float3, op>, FMXImpl_AOAOA<float4, double4, float4, float4, op>  },
                    { FMXImpl_AOAOA<float, double, float, double, op>, FMXImpl_AOAOA<float2, double2, float2, double2, op>, FMXImpl_AOAOA<float3, double3, float3, double3, op>, FMXImpl_AOAOA<float4, double4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<float, double, double, uchar, op>, FMXImpl_AOAOA<float2, double2, double2, uchar2, op>, FMXImpl_AOAOA<float3, double3, double3, uchar3, op>, FMXImpl_AOAOA<float4, double4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<float, double, double, schar, op>, FMXImpl_AOAOA<float2, double2, double2, char2, op>, FMXImpl_AOAOA<float3, double3, double3, char3, op>, FMXImpl_AOAOA<float4, double4, double4, char4, op>  },
                    { FMXImpl_AOAOA<float, double, double, ushort, op>, FMXImpl_AOAOA<float2, double2, double2, ushort2, op>, FMXImpl_AOAOA<float3, double3, double3, ushort3, op>, FMXImpl_AOAOA<float4, double4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<float, double, double, short, op>, FMXImpl_AOAOA<float2, double2, double2, short2, op>, FMXImpl_AOAOA<float3, double3, double3, short3, op>, FMXImpl_AOAOA<float4, double4, double4, short4, op>  },
                    { FMXImpl_AOAOA<float, double, double, int, op>, FMXImpl_AOAOA<float2, double2, double2, int2, op>, FMXImpl_AOAOA<float3, double3, double3, int3, op>, FMXImpl_AOAOA<float4, double4, double4, int4, op>  },
                    { FMXImpl_AOAOA<float, double, double, float, op>, FMXImpl_AOAOA<float2, double2, double2, float2, op>, FMXImpl_AOAOA<float3, double3, double3, float3, op>, FMXImpl_AOAOA<float4, double4, double4, float4, op>  },
                    { FMXImpl_AOAOA<float, double, double, double, op>, FMXImpl_AOAOA<float2, double2, double2, double2, op>, FMXImpl_AOAOA<float3, double3, double3, double3, op>, FMXImpl_AOAOA<float4, double4, double4, double4, op>  },
                },
            },
        },
        {
            {
                {
                    { FMXImpl_AOAOA<double, uchar, uchar, uchar, op>, FMXImpl_AOAOA<double2, uchar2, uchar2, uchar2, op>, FMXImpl_AOAOA<double3, uchar3, uchar3, uchar3, op>, FMXImpl_AOAOA<double4, uchar4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, uchar, uchar, schar, op>, FMXImpl_AOAOA<double2, uchar2, uchar2, char2, op>, FMXImpl_AOAOA<double3, uchar3, uchar3, char3, op>, FMXImpl_AOAOA<double4, uchar4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<double, uchar, uchar, ushort, op>, FMXImpl_AOAOA<double2, uchar2, uchar2, ushort2, op>, FMXImpl_AOAOA<double3, uchar3, uchar3, ushort3, op>, FMXImpl_AOAOA<double4, uchar4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, uchar, uchar, short, op>, FMXImpl_AOAOA<double2, uchar2, uchar2, short2, op>, FMXImpl_AOAOA<double3, uchar3, uchar3, short3, op>, FMXImpl_AOAOA<double4, uchar4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<double, uchar, uchar, int, op>, FMXImpl_AOAOA<double2, uchar2, uchar2, int2, op>, FMXImpl_AOAOA<double3, uchar3, uchar3, int3, op>, FMXImpl_AOAOA<double4, uchar4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<double, uchar, uchar, float, op>, FMXImpl_AOAOA<double2, uchar2, uchar2, float2, op>, FMXImpl_AOAOA<double3, uchar3, uchar3, float3, op>, FMXImpl_AOAOA<double4, uchar4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<double, uchar, uchar, double, op>, FMXImpl_AOAOA<double2, uchar2, uchar2, double2, op>, FMXImpl_AOAOA<double3, uchar3, uchar3, double3, op>, FMXImpl_AOAOA<double4, uchar4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, uchar, schar, uchar, op>, FMXImpl_AOAOA<double2, uchar2, char2, uchar2, op>, FMXImpl_AOAOA<double3, uchar3, char3, uchar3, op>, FMXImpl_AOAOA<double4, uchar4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, uchar, schar, schar, op>, FMXImpl_AOAOA<double2, uchar2, char2, char2, op>, FMXImpl_AOAOA<double3, uchar3, char3, char3, op>, FMXImpl_AOAOA<double4, uchar4, char4, char4, op>  },
                    { FMXImpl_AOAOA<double, uchar, schar, ushort, op>, FMXImpl_AOAOA<double2, uchar2, char2, ushort2, op>, FMXImpl_AOAOA<double3, uchar3, char3, ushort3, op>, FMXImpl_AOAOA<double4, uchar4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, uchar, schar, short, op>, FMXImpl_AOAOA<double2, uchar2, char2, short2, op>, FMXImpl_AOAOA<double3, uchar3, char3, short3, op>, FMXImpl_AOAOA<double4, uchar4, char4, short4, op>  },
                    { FMXImpl_AOAOA<double, uchar, schar, int, op>, FMXImpl_AOAOA<double2, uchar2, char2, int2, op>, FMXImpl_AOAOA<double3, uchar3, char3, int3, op>, FMXImpl_AOAOA<double4, uchar4, char4, int4, op>  },
                    { FMXImpl_AOAOA<double, uchar, schar, float, op>, FMXImpl_AOAOA<double2, uchar2, char2, float2, op>, FMXImpl_AOAOA<double3, uchar3, char3, float3, op>, FMXImpl_AOAOA<double4, uchar4, char4, float4, op>  },
                    { FMXImpl_AOAOA<double, uchar, schar, double, op>, FMXImpl_AOAOA<double2, uchar2, char2, double2, op>, FMXImpl_AOAOA<double3, uchar3, char3, double3, op>, FMXImpl_AOAOA<double4, uchar4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, uchar, ushort, uchar, op>, FMXImpl_AOAOA<double2, uchar2, ushort2, uchar2, op>, FMXImpl_AOAOA<double3, uchar3, ushort3, uchar3, op>, FMXImpl_AOAOA<double4, uchar4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, uchar, ushort, schar, op>, FMXImpl_AOAOA<double2, uchar2, ushort2, char2, op>, FMXImpl_AOAOA<double3, uchar3, ushort3, char3, op>, FMXImpl_AOAOA<double4, uchar4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<double, uchar, ushort, ushort, op>, FMXImpl_AOAOA<double2, uchar2, ushort2, ushort2, op>, FMXImpl_AOAOA<double3, uchar3, ushort3, ushort3, op>, FMXImpl_AOAOA<double4, uchar4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, uchar, ushort, short, op>, FMXImpl_AOAOA<double2, uchar2, ushort2, short2, op>, FMXImpl_AOAOA<double3, uchar3, ushort3, short3, op>, FMXImpl_AOAOA<double4, uchar4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<double, uchar, ushort, int, op>, FMXImpl_AOAOA<double2, uchar2, ushort2, int2, op>, FMXImpl_AOAOA<double3, uchar3, ushort3, int3, op>, FMXImpl_AOAOA<double4, uchar4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<double, uchar, ushort, float, op>, FMXImpl_AOAOA<double2, uchar2, ushort2, float2, op>, FMXImpl_AOAOA<double3, uchar3, ushort3, float3, op>, FMXImpl_AOAOA<double4, uchar4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<double, uchar, ushort, double, op>, FMXImpl_AOAOA<double2, uchar2, ushort2, double2, op>, FMXImpl_AOAOA<double3, uchar3, ushort3, double3, op>, FMXImpl_AOAOA<double4, uchar4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, uchar, short, uchar, op>, FMXImpl_AOAOA<double2, uchar2, short2, uchar2, op>, FMXImpl_AOAOA<double3, uchar3, short3, uchar3, op>, FMXImpl_AOAOA<double4, uchar4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, uchar, short, schar, op>, FMXImpl_AOAOA<double2, uchar2, short2, char2, op>, FMXImpl_AOAOA<double3, uchar3, short3, char3, op>, FMXImpl_AOAOA<double4, uchar4, short4, char4, op>  },
                    { FMXImpl_AOAOA<double, uchar, short, ushort, op>, FMXImpl_AOAOA<double2, uchar2, short2, ushort2, op>, FMXImpl_AOAOA<double3, uchar3, short3, ushort3, op>, FMXImpl_AOAOA<double4, uchar4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, uchar, short, short, op>, FMXImpl_AOAOA<double2, uchar2, short2, short2, op>, FMXImpl_AOAOA<double3, uchar3, short3, short3, op>, FMXImpl_AOAOA<double4, uchar4, short4, short4, op>  },
                    { FMXImpl_AOAOA<double, uchar, short, int, op>, FMXImpl_AOAOA<double2, uchar2, short2, int2, op>, FMXImpl_AOAOA<double3, uchar3, short3, int3, op>, FMXImpl_AOAOA<double4, uchar4, short4, int4, op>  },
                    { FMXImpl_AOAOA<double, uchar, short, float, op>, FMXImpl_AOAOA<double2, uchar2, short2, float2, op>, FMXImpl_AOAOA<double3, uchar3, short3, float3, op>, FMXImpl_AOAOA<double4, uchar4, short4, float4, op>  },
                    { FMXImpl_AOAOA<double, uchar, short, double, op>, FMXImpl_AOAOA<double2, uchar2, short2, double2, op>, FMXImpl_AOAOA<double3, uchar3, short3, double3, op>, FMXImpl_AOAOA<double4, uchar4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, uchar, int, uchar, op>, FMXImpl_AOAOA<double2, uchar2, int2, uchar2, op>, FMXImpl_AOAOA<double3, uchar3, int3, uchar3, op>, FMXImpl_AOAOA<double4, uchar4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, uchar, int, schar, op>, FMXImpl_AOAOA<double2, uchar2, int2, char2, op>, FMXImpl_AOAOA<double3, uchar3, int3, char3, op>, FMXImpl_AOAOA<double4, uchar4, int4, char4, op>  },
                    { FMXImpl_AOAOA<double, uchar, int, ushort, op>, FMXImpl_AOAOA<double2, uchar2, int2, ushort2, op>, FMXImpl_AOAOA<double3, uchar3, int3, ushort3, op>, FMXImpl_AOAOA<double4, uchar4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, uchar, int, short, op>, FMXImpl_AOAOA<double2, uchar2, int2, short2, op>, FMXImpl_AOAOA<double3, uchar3, int3, short3, op>, FMXImpl_AOAOA<double4, uchar4, int4, short4, op>  },
                    { FMXImpl_AOAOA<double, uchar, int, int, op>, FMXImpl_AOAOA<double2, uchar2, int2, int2, op>, FMXImpl_AOAOA<double3, uchar3, int3, int3, op>, FMXImpl_AOAOA<double4, uchar4, int4, int4, op>  },
                    { FMXImpl_AOAOA<double, uchar, int, float, op>, FMXImpl_AOAOA<double2, uchar2, int2, float2, op>, FMXImpl_AOAOA<double3, uchar3, int3, float3, op>, FMXImpl_AOAOA<double4, uchar4, int4, float4, op>  },
                    { FMXImpl_AOAOA<double, uchar, int, double, op>, FMXImpl_AOAOA<double2, uchar2, int2, double2, op>, FMXImpl_AOAOA<double3, uchar3, int3, double3, op>, FMXImpl_AOAOA<double4, uchar4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, uchar, float, uchar, op>, FMXImpl_AOAOA<double2, uchar2, float2, uchar2, op>, FMXImpl_AOAOA<double3, uchar3, float3, uchar3, op>, FMXImpl_AOAOA<double4, uchar4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, uchar, float, schar, op>, FMXImpl_AOAOA<double2, uchar2, float2, char2, op>, FMXImpl_AOAOA<double3, uchar3, float3, char3, op>, FMXImpl_AOAOA<double4, uchar4, float4, char4, op>  },
                    { FMXImpl_AOAOA<double, uchar, float, ushort, op>, FMXImpl_AOAOA<double2, uchar2, float2, ushort2, op>, FMXImpl_AOAOA<double3, uchar3, float3, ushort3, op>, FMXImpl_AOAOA<double4, uchar4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, uchar, float, short, op>, FMXImpl_AOAOA<double2, uchar2, float2, short2, op>, FMXImpl_AOAOA<double3, uchar3, float3, short3, op>, FMXImpl_AOAOA<double4, uchar4, float4, short4, op>  },
                    { FMXImpl_AOAOA<double, uchar, float, int, op>, FMXImpl_AOAOA<double2, uchar2, float2, int2, op>, FMXImpl_AOAOA<double3, uchar3, float3, int3, op>, FMXImpl_AOAOA<double4, uchar4, float4, int4, op>  },
                    { FMXImpl_AOAOA<double, uchar, float, float, op>, FMXImpl_AOAOA<double2, uchar2, float2, float2, op>, FMXImpl_AOAOA<double3, uchar3, float3, float3, op>, FMXImpl_AOAOA<double4, uchar4, float4, float4, op>  },
                    { FMXImpl_AOAOA<double, uchar, float, double, op>, FMXImpl_AOAOA<double2, uchar2, float2, double2, op>, FMXImpl_AOAOA<double3, uchar3, float3, double3, op>, FMXImpl_AOAOA<double4, uchar4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, uchar, double, uchar, op>, FMXImpl_AOAOA<double2, uchar2, double2, uchar2, op>, FMXImpl_AOAOA<double3, uchar3, double3, uchar3, op>, FMXImpl_AOAOA<double4, uchar4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, uchar, double, schar, op>, FMXImpl_AOAOA<double2, uchar2, double2, char2, op>, FMXImpl_AOAOA<double3, uchar3, double3, char3, op>, FMXImpl_AOAOA<double4, uchar4, double4, char4, op>  },
                    { FMXImpl_AOAOA<double, uchar, double, ushort, op>, FMXImpl_AOAOA<double2, uchar2, double2, ushort2, op>, FMXImpl_AOAOA<double3, uchar3, double3, ushort3, op>, FMXImpl_AOAOA<double4, uchar4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, uchar, double, short, op>, FMXImpl_AOAOA<double2, uchar2, double2, short2, op>, FMXImpl_AOAOA<double3, uchar3, double3, short3, op>, FMXImpl_AOAOA<double4, uchar4, double4, short4, op>  },
                    { FMXImpl_AOAOA<double, uchar, double, int, op>, FMXImpl_AOAOA<double2, uchar2, double2, int2, op>, FMXImpl_AOAOA<double3, uchar3, double3, int3, op>, FMXImpl_AOAOA<double4, uchar4, double4, int4, op>  },
                    { FMXImpl_AOAOA<double, uchar, double, float, op>, FMXImpl_AOAOA<double2, uchar2, double2, float2, op>, FMXImpl_AOAOA<double3, uchar3, double3, float3, op>, FMXImpl_AOAOA<double4, uchar4, double4, float4, op>  },
                    { FMXImpl_AOAOA<double, uchar, double, double, op>, FMXImpl_AOAOA<double2, uchar2, double2, double2, op>, FMXImpl_AOAOA<double3, uchar3, double3, double3, op>, FMXImpl_AOAOA<double4, uchar4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<double, schar, uchar, uchar, op>, FMXImpl_AOAOA<double2, char2, uchar2, uchar2, op>, FMXImpl_AOAOA<double3, char3, uchar3, uchar3, op>, FMXImpl_AOAOA<double4, char4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, schar, uchar, schar, op>, FMXImpl_AOAOA<double2, char2, uchar2, char2, op>, FMXImpl_AOAOA<double3, char3, uchar3, char3, op>, FMXImpl_AOAOA<double4, char4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<double, schar, uchar, ushort, op>, FMXImpl_AOAOA<double2, char2, uchar2, ushort2, op>, FMXImpl_AOAOA<double3, char3, uchar3, ushort3, op>, FMXImpl_AOAOA<double4, char4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, schar, uchar, short, op>, FMXImpl_AOAOA<double2, char2, uchar2, short2, op>, FMXImpl_AOAOA<double3, char3, uchar3, short3, op>, FMXImpl_AOAOA<double4, char4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<double, schar, uchar, int, op>, FMXImpl_AOAOA<double2, char2, uchar2, int2, op>, FMXImpl_AOAOA<double3, char3, uchar3, int3, op>, FMXImpl_AOAOA<double4, char4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<double, schar, uchar, float, op>, FMXImpl_AOAOA<double2, char2, uchar2, float2, op>, FMXImpl_AOAOA<double3, char3, uchar3, float3, op>, FMXImpl_AOAOA<double4, char4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<double, schar, uchar, double, op>, FMXImpl_AOAOA<double2, char2, uchar2, double2, op>, FMXImpl_AOAOA<double3, char3, uchar3, double3, op>, FMXImpl_AOAOA<double4, char4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, schar, schar, uchar, op>, FMXImpl_AOAOA<double2, char2, char2, uchar2, op>, FMXImpl_AOAOA<double3, char3, char3, uchar3, op>, FMXImpl_AOAOA<double4, char4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, schar, schar, schar, op>, FMXImpl_AOAOA<double2, char2, char2, char2, op>, FMXImpl_AOAOA<double3, char3, char3, char3, op>, FMXImpl_AOAOA<double4, char4, char4, char4, op>  },
                    { FMXImpl_AOAOA<double, schar, schar, ushort, op>, FMXImpl_AOAOA<double2, char2, char2, ushort2, op>, FMXImpl_AOAOA<double3, char3, char3, ushort3, op>, FMXImpl_AOAOA<double4, char4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, schar, schar, short, op>, FMXImpl_AOAOA<double2, char2, char2, short2, op>, FMXImpl_AOAOA<double3, char3, char3, short3, op>, FMXImpl_AOAOA<double4, char4, char4, short4, op>  },
                    { FMXImpl_AOAOA<double, schar, schar, int, op>, FMXImpl_AOAOA<double2, char2, char2, int2, op>, FMXImpl_AOAOA<double3, char3, char3, int3, op>, FMXImpl_AOAOA<double4, char4, char4, int4, op>  },
                    { FMXImpl_AOAOA<double, schar, schar, float, op>, FMXImpl_AOAOA<double2, char2, char2, float2, op>, FMXImpl_AOAOA<double3, char3, char3, float3, op>, FMXImpl_AOAOA<double4, char4, char4, float4, op>  },
                    { FMXImpl_AOAOA<double, schar, schar, double, op>, FMXImpl_AOAOA<double2, char2, char2, double2, op>, FMXImpl_AOAOA<double3, char3, char3, double3, op>, FMXImpl_AOAOA<double4, char4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, schar, ushort, uchar, op>, FMXImpl_AOAOA<double2, char2, ushort2, uchar2, op>, FMXImpl_AOAOA<double3, char3, ushort3, uchar3, op>, FMXImpl_AOAOA<double4, char4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, schar, ushort, schar, op>, FMXImpl_AOAOA<double2, char2, ushort2, char2, op>, FMXImpl_AOAOA<double3, char3, ushort3, char3, op>, FMXImpl_AOAOA<double4, char4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<double, schar, ushort, ushort, op>, FMXImpl_AOAOA<double2, char2, ushort2, ushort2, op>, FMXImpl_AOAOA<double3, char3, ushort3, ushort3, op>, FMXImpl_AOAOA<double4, char4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, schar, ushort, short, op>, FMXImpl_AOAOA<double2, char2, ushort2, short2, op>, FMXImpl_AOAOA<double3, char3, ushort3, short3, op>, FMXImpl_AOAOA<double4, char4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<double, schar, ushort, int, op>, FMXImpl_AOAOA<double2, char2, ushort2, int2, op>, FMXImpl_AOAOA<double3, char3, ushort3, int3, op>, FMXImpl_AOAOA<double4, char4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<double, schar, ushort, float, op>, FMXImpl_AOAOA<double2, char2, ushort2, float2, op>, FMXImpl_AOAOA<double3, char3, ushort3, float3, op>, FMXImpl_AOAOA<double4, char4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<double, schar, ushort, double, op>, FMXImpl_AOAOA<double2, char2, ushort2, double2, op>, FMXImpl_AOAOA<double3, char3, ushort3, double3, op>, FMXImpl_AOAOA<double4, char4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, schar, short, uchar, op>, FMXImpl_AOAOA<double2, char2, short2, uchar2, op>, FMXImpl_AOAOA<double3, char3, short3, uchar3, op>, FMXImpl_AOAOA<double4, char4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, schar, short, schar, op>, FMXImpl_AOAOA<double2, char2, short2, char2, op>, FMXImpl_AOAOA<double3, char3, short3, char3, op>, FMXImpl_AOAOA<double4, char4, short4, char4, op>  },
                    { FMXImpl_AOAOA<double, schar, short, ushort, op>, FMXImpl_AOAOA<double2, char2, short2, ushort2, op>, FMXImpl_AOAOA<double3, char3, short3, ushort3, op>, FMXImpl_AOAOA<double4, char4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, schar, short, short, op>, FMXImpl_AOAOA<double2, char2, short2, short2, op>, FMXImpl_AOAOA<double3, char3, short3, short3, op>, FMXImpl_AOAOA<double4, char4, short4, short4, op>  },
                    { FMXImpl_AOAOA<double, schar, short, int, op>, FMXImpl_AOAOA<double2, char2, short2, int2, op>, FMXImpl_AOAOA<double3, char3, short3, int3, op>, FMXImpl_AOAOA<double4, char4, short4, int4, op>  },
                    { FMXImpl_AOAOA<double, schar, short, float, op>, FMXImpl_AOAOA<double2, char2, short2, float2, op>, FMXImpl_AOAOA<double3, char3, short3, float3, op>, FMXImpl_AOAOA<double4, char4, short4, float4, op>  },
                    { FMXImpl_AOAOA<double, schar, short, double, op>, FMXImpl_AOAOA<double2, char2, short2, double2, op>, FMXImpl_AOAOA<double3, char3, short3, double3, op>, FMXImpl_AOAOA<double4, char4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, schar, int, uchar, op>, FMXImpl_AOAOA<double2, char2, int2, uchar2, op>, FMXImpl_AOAOA<double3, char3, int3, uchar3, op>, FMXImpl_AOAOA<double4, char4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, schar, int, schar, op>, FMXImpl_AOAOA<double2, char2, int2, char2, op>, FMXImpl_AOAOA<double3, char3, int3, char3, op>, FMXImpl_AOAOA<double4, char4, int4, char4, op>  },
                    { FMXImpl_AOAOA<double, schar, int, ushort, op>, FMXImpl_AOAOA<double2, char2, int2, ushort2, op>, FMXImpl_AOAOA<double3, char3, int3, ushort3, op>, FMXImpl_AOAOA<double4, char4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, schar, int, short, op>, FMXImpl_AOAOA<double2, char2, int2, short2, op>, FMXImpl_AOAOA<double3, char3, int3, short3, op>, FMXImpl_AOAOA<double4, char4, int4, short4, op>  },
                    { FMXImpl_AOAOA<double, schar, int, int, op>, FMXImpl_AOAOA<double2, char2, int2, int2, op>, FMXImpl_AOAOA<double3, char3, int3, int3, op>, FMXImpl_AOAOA<double4, char4, int4, int4, op>  },
                    { FMXImpl_AOAOA<double, schar, int, float, op>, FMXImpl_AOAOA<double2, char2, int2, float2, op>, FMXImpl_AOAOA<double3, char3, int3, float3, op>, FMXImpl_AOAOA<double4, char4, int4, float4, op>  },
                    { FMXImpl_AOAOA<double, schar, int, double, op>, FMXImpl_AOAOA<double2, char2, int2, double2, op>, FMXImpl_AOAOA<double3, char3, int3, double3, op>, FMXImpl_AOAOA<double4, char4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, schar, float, uchar, op>, FMXImpl_AOAOA<double2, char2, float2, uchar2, op>, FMXImpl_AOAOA<double3, char3, float3, uchar3, op>, FMXImpl_AOAOA<double4, char4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, schar, float, schar, op>, FMXImpl_AOAOA<double2, char2, float2, char2, op>, FMXImpl_AOAOA<double3, char3, float3, char3, op>, FMXImpl_AOAOA<double4, char4, float4, char4, op>  },
                    { FMXImpl_AOAOA<double, schar, float, ushort, op>, FMXImpl_AOAOA<double2, char2, float2, ushort2, op>, FMXImpl_AOAOA<double3, char3, float3, ushort3, op>, FMXImpl_AOAOA<double4, char4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, schar, float, short, op>, FMXImpl_AOAOA<double2, char2, float2, short2, op>, FMXImpl_AOAOA<double3, char3, float3, short3, op>, FMXImpl_AOAOA<double4, char4, float4, short4, op>  },
                    { FMXImpl_AOAOA<double, schar, float, int, op>, FMXImpl_AOAOA<double2, char2, float2, int2, op>, FMXImpl_AOAOA<double3, char3, float3, int3, op>, FMXImpl_AOAOA<double4, char4, float4, int4, op>  },
                    { FMXImpl_AOAOA<double, schar, float, float, op>, FMXImpl_AOAOA<double2, char2, float2, float2, op>, FMXImpl_AOAOA<double3, char3, float3, float3, op>, FMXImpl_AOAOA<double4, char4, float4, float4, op>  },
                    { FMXImpl_AOAOA<double, schar, float, double, op>, FMXImpl_AOAOA<double2, char2, float2, double2, op>, FMXImpl_AOAOA<double3, char3, float3, double3, op>, FMXImpl_AOAOA<double4, char4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, schar, double, uchar, op>, FMXImpl_AOAOA<double2, char2, double2, uchar2, op>, FMXImpl_AOAOA<double3, char3, double3, uchar3, op>, FMXImpl_AOAOA<double4, char4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, schar, double, schar, op>, FMXImpl_AOAOA<double2, char2, double2, char2, op>, FMXImpl_AOAOA<double3, char3, double3, char3, op>, FMXImpl_AOAOA<double4, char4, double4, char4, op>  },
                    { FMXImpl_AOAOA<double, schar, double, ushort, op>, FMXImpl_AOAOA<double2, char2, double2, ushort2, op>, FMXImpl_AOAOA<double3, char3, double3, ushort3, op>, FMXImpl_AOAOA<double4, char4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, schar, double, short, op>, FMXImpl_AOAOA<double2, char2, double2, short2, op>, FMXImpl_AOAOA<double3, char3, double3, short3, op>, FMXImpl_AOAOA<double4, char4, double4, short4, op>  },
                    { FMXImpl_AOAOA<double, schar, double, int, op>, FMXImpl_AOAOA<double2, char2, double2, int2, op>, FMXImpl_AOAOA<double3, char3, double3, int3, op>, FMXImpl_AOAOA<double4, char4, double4, int4, op>  },
                    { FMXImpl_AOAOA<double, schar, double, float, op>, FMXImpl_AOAOA<double2, char2, double2, float2, op>, FMXImpl_AOAOA<double3, char3, double3, float3, op>, FMXImpl_AOAOA<double4, char4, double4, float4, op>  },
                    { FMXImpl_AOAOA<double, schar, double, double, op>, FMXImpl_AOAOA<double2, char2, double2, double2, op>, FMXImpl_AOAOA<double3, char3, double3, double3, op>, FMXImpl_AOAOA<double4, char4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<double, ushort, uchar, uchar, op>, FMXImpl_AOAOA<double2, ushort2, uchar2, uchar2, op>, FMXImpl_AOAOA<double3, ushort3, uchar3, uchar3, op>, FMXImpl_AOAOA<double4, ushort4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, ushort, uchar, schar, op>, FMXImpl_AOAOA<double2, ushort2, uchar2, char2, op>, FMXImpl_AOAOA<double3, ushort3, uchar3, char3, op>, FMXImpl_AOAOA<double4, ushort4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<double, ushort, uchar, ushort, op>, FMXImpl_AOAOA<double2, ushort2, uchar2, ushort2, op>, FMXImpl_AOAOA<double3, ushort3, uchar3, ushort3, op>, FMXImpl_AOAOA<double4, ushort4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, ushort, uchar, short, op>, FMXImpl_AOAOA<double2, ushort2, uchar2, short2, op>, FMXImpl_AOAOA<double3, ushort3, uchar3, short3, op>, FMXImpl_AOAOA<double4, ushort4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<double, ushort, uchar, int, op>, FMXImpl_AOAOA<double2, ushort2, uchar2, int2, op>, FMXImpl_AOAOA<double3, ushort3, uchar3, int3, op>, FMXImpl_AOAOA<double4, ushort4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<double, ushort, uchar, float, op>, FMXImpl_AOAOA<double2, ushort2, uchar2, float2, op>, FMXImpl_AOAOA<double3, ushort3, uchar3, float3, op>, FMXImpl_AOAOA<double4, ushort4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<double, ushort, uchar, double, op>, FMXImpl_AOAOA<double2, ushort2, uchar2, double2, op>, FMXImpl_AOAOA<double3, ushort3, uchar3, double3, op>, FMXImpl_AOAOA<double4, ushort4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, ushort, schar, uchar, op>, FMXImpl_AOAOA<double2, ushort2, char2, uchar2, op>, FMXImpl_AOAOA<double3, ushort3, char3, uchar3, op>, FMXImpl_AOAOA<double4, ushort4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, ushort, schar, schar, op>, FMXImpl_AOAOA<double2, ushort2, char2, char2, op>, FMXImpl_AOAOA<double3, ushort3, char3, char3, op>, FMXImpl_AOAOA<double4, ushort4, char4, char4, op>  },
                    { FMXImpl_AOAOA<double, ushort, schar, ushort, op>, FMXImpl_AOAOA<double2, ushort2, char2, ushort2, op>, FMXImpl_AOAOA<double3, ushort3, char3, ushort3, op>, FMXImpl_AOAOA<double4, ushort4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, ushort, schar, short, op>, FMXImpl_AOAOA<double2, ushort2, char2, short2, op>, FMXImpl_AOAOA<double3, ushort3, char3, short3, op>, FMXImpl_AOAOA<double4, ushort4, char4, short4, op>  },
                    { FMXImpl_AOAOA<double, ushort, schar, int, op>, FMXImpl_AOAOA<double2, ushort2, char2, int2, op>, FMXImpl_AOAOA<double3, ushort3, char3, int3, op>, FMXImpl_AOAOA<double4, ushort4, char4, int4, op>  },
                    { FMXImpl_AOAOA<double, ushort, schar, float, op>, FMXImpl_AOAOA<double2, ushort2, char2, float2, op>, FMXImpl_AOAOA<double3, ushort3, char3, float3, op>, FMXImpl_AOAOA<double4, ushort4, char4, float4, op>  },
                    { FMXImpl_AOAOA<double, ushort, schar, double, op>, FMXImpl_AOAOA<double2, ushort2, char2, double2, op>, FMXImpl_AOAOA<double3, ushort3, char3, double3, op>, FMXImpl_AOAOA<double4, ushort4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, ushort, ushort, uchar, op>, FMXImpl_AOAOA<double2, ushort2, ushort2, uchar2, op>, FMXImpl_AOAOA<double3, ushort3, ushort3, uchar3, op>, FMXImpl_AOAOA<double4, ushort4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, ushort, ushort, schar, op>, FMXImpl_AOAOA<double2, ushort2, ushort2, char2, op>, FMXImpl_AOAOA<double3, ushort3, ushort3, char3, op>, FMXImpl_AOAOA<double4, ushort4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<double, ushort, ushort, ushort, op>, FMXImpl_AOAOA<double2, ushort2, ushort2, ushort2, op>, FMXImpl_AOAOA<double3, ushort3, ushort3, ushort3, op>, FMXImpl_AOAOA<double4, ushort4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, ushort, ushort, short, op>, FMXImpl_AOAOA<double2, ushort2, ushort2, short2, op>, FMXImpl_AOAOA<double3, ushort3, ushort3, short3, op>, FMXImpl_AOAOA<double4, ushort4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<double, ushort, ushort, int, op>, FMXImpl_AOAOA<double2, ushort2, ushort2, int2, op>, FMXImpl_AOAOA<double3, ushort3, ushort3, int3, op>, FMXImpl_AOAOA<double4, ushort4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<double, ushort, ushort, float, op>, FMXImpl_AOAOA<double2, ushort2, ushort2, float2, op>, FMXImpl_AOAOA<double3, ushort3, ushort3, float3, op>, FMXImpl_AOAOA<double4, ushort4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<double, ushort, ushort, double, op>, FMXImpl_AOAOA<double2, ushort2, ushort2, double2, op>, FMXImpl_AOAOA<double3, ushort3, ushort3, double3, op>, FMXImpl_AOAOA<double4, ushort4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, ushort, short, uchar, op>, FMXImpl_AOAOA<double2, ushort2, short2, uchar2, op>, FMXImpl_AOAOA<double3, ushort3, short3, uchar3, op>, FMXImpl_AOAOA<double4, ushort4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, ushort, short, schar, op>, FMXImpl_AOAOA<double2, ushort2, short2, char2, op>, FMXImpl_AOAOA<double3, ushort3, short3, char3, op>, FMXImpl_AOAOA<double4, ushort4, short4, char4, op>  },
                    { FMXImpl_AOAOA<double, ushort, short, ushort, op>, FMXImpl_AOAOA<double2, ushort2, short2, ushort2, op>, FMXImpl_AOAOA<double3, ushort3, short3, ushort3, op>, FMXImpl_AOAOA<double4, ushort4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, ushort, short, short, op>, FMXImpl_AOAOA<double2, ushort2, short2, short2, op>, FMXImpl_AOAOA<double3, ushort3, short3, short3, op>, FMXImpl_AOAOA<double4, ushort4, short4, short4, op>  },
                    { FMXImpl_AOAOA<double, ushort, short, int, op>, FMXImpl_AOAOA<double2, ushort2, short2, int2, op>, FMXImpl_AOAOA<double3, ushort3, short3, int3, op>, FMXImpl_AOAOA<double4, ushort4, short4, int4, op>  },
                    { FMXImpl_AOAOA<double, ushort, short, float, op>, FMXImpl_AOAOA<double2, ushort2, short2, float2, op>, FMXImpl_AOAOA<double3, ushort3, short3, float3, op>, FMXImpl_AOAOA<double4, ushort4, short4, float4, op>  },
                    { FMXImpl_AOAOA<double, ushort, short, double, op>, FMXImpl_AOAOA<double2, ushort2, short2, double2, op>, FMXImpl_AOAOA<double3, ushort3, short3, double3, op>, FMXImpl_AOAOA<double4, ushort4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, ushort, int, uchar, op>, FMXImpl_AOAOA<double2, ushort2, int2, uchar2, op>, FMXImpl_AOAOA<double3, ushort3, int3, uchar3, op>, FMXImpl_AOAOA<double4, ushort4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, ushort, int, schar, op>, FMXImpl_AOAOA<double2, ushort2, int2, char2, op>, FMXImpl_AOAOA<double3, ushort3, int3, char3, op>, FMXImpl_AOAOA<double4, ushort4, int4, char4, op>  },
                    { FMXImpl_AOAOA<double, ushort, int, ushort, op>, FMXImpl_AOAOA<double2, ushort2, int2, ushort2, op>, FMXImpl_AOAOA<double3, ushort3, int3, ushort3, op>, FMXImpl_AOAOA<double4, ushort4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, ushort, int, short, op>, FMXImpl_AOAOA<double2, ushort2, int2, short2, op>, FMXImpl_AOAOA<double3, ushort3, int3, short3, op>, FMXImpl_AOAOA<double4, ushort4, int4, short4, op>  },
                    { FMXImpl_AOAOA<double, ushort, int, int, op>, FMXImpl_AOAOA<double2, ushort2, int2, int2, op>, FMXImpl_AOAOA<double3, ushort3, int3, int3, op>, FMXImpl_AOAOA<double4, ushort4, int4, int4, op>  },
                    { FMXImpl_AOAOA<double, ushort, int, float, op>, FMXImpl_AOAOA<double2, ushort2, int2, float2, op>, FMXImpl_AOAOA<double3, ushort3, int3, float3, op>, FMXImpl_AOAOA<double4, ushort4, int4, float4, op>  },
                    { FMXImpl_AOAOA<double, ushort, int, double, op>, FMXImpl_AOAOA<double2, ushort2, int2, double2, op>, FMXImpl_AOAOA<double3, ushort3, int3, double3, op>, FMXImpl_AOAOA<double4, ushort4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, ushort, float, uchar, op>, FMXImpl_AOAOA<double2, ushort2, float2, uchar2, op>, FMXImpl_AOAOA<double3, ushort3, float3, uchar3, op>, FMXImpl_AOAOA<double4, ushort4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, ushort, float, schar, op>, FMXImpl_AOAOA<double2, ushort2, float2, char2, op>, FMXImpl_AOAOA<double3, ushort3, float3, char3, op>, FMXImpl_AOAOA<double4, ushort4, float4, char4, op>  },
                    { FMXImpl_AOAOA<double, ushort, float, ushort, op>, FMXImpl_AOAOA<double2, ushort2, float2, ushort2, op>, FMXImpl_AOAOA<double3, ushort3, float3, ushort3, op>, FMXImpl_AOAOA<double4, ushort4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, ushort, float, short, op>, FMXImpl_AOAOA<double2, ushort2, float2, short2, op>, FMXImpl_AOAOA<double3, ushort3, float3, short3, op>, FMXImpl_AOAOA<double4, ushort4, float4, short4, op>  },
                    { FMXImpl_AOAOA<double, ushort, float, int, op>, FMXImpl_AOAOA<double2, ushort2, float2, int2, op>, FMXImpl_AOAOA<double3, ushort3, float3, int3, op>, FMXImpl_AOAOA<double4, ushort4, float4, int4, op>  },
                    { FMXImpl_AOAOA<double, ushort, float, float, op>, FMXImpl_AOAOA<double2, ushort2, float2, float2, op>, FMXImpl_AOAOA<double3, ushort3, float3, float3, op>, FMXImpl_AOAOA<double4, ushort4, float4, float4, op>  },
                    { FMXImpl_AOAOA<double, ushort, float, double, op>, FMXImpl_AOAOA<double2, ushort2, float2, double2, op>, FMXImpl_AOAOA<double3, ushort3, float3, double3, op>, FMXImpl_AOAOA<double4, ushort4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, ushort, double, uchar, op>, FMXImpl_AOAOA<double2, ushort2, double2, uchar2, op>, FMXImpl_AOAOA<double3, ushort3, double3, uchar3, op>, FMXImpl_AOAOA<double4, ushort4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, ushort, double, schar, op>, FMXImpl_AOAOA<double2, ushort2, double2, char2, op>, FMXImpl_AOAOA<double3, ushort3, double3, char3, op>, FMXImpl_AOAOA<double4, ushort4, double4, char4, op>  },
                    { FMXImpl_AOAOA<double, ushort, double, ushort, op>, FMXImpl_AOAOA<double2, ushort2, double2, ushort2, op>, FMXImpl_AOAOA<double3, ushort3, double3, ushort3, op>, FMXImpl_AOAOA<double4, ushort4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, ushort, double, short, op>, FMXImpl_AOAOA<double2, ushort2, double2, short2, op>, FMXImpl_AOAOA<double3, ushort3, double3, short3, op>, FMXImpl_AOAOA<double4, ushort4, double4, short4, op>  },
                    { FMXImpl_AOAOA<double, ushort, double, int, op>, FMXImpl_AOAOA<double2, ushort2, double2, int2, op>, FMXImpl_AOAOA<double3, ushort3, double3, int3, op>, FMXImpl_AOAOA<double4, ushort4, double4, int4, op>  },
                    { FMXImpl_AOAOA<double, ushort, double, float, op>, FMXImpl_AOAOA<double2, ushort2, double2, float2, op>, FMXImpl_AOAOA<double3, ushort3, double3, float3, op>, FMXImpl_AOAOA<double4, ushort4, double4, float4, op>  },
                    { FMXImpl_AOAOA<double, ushort, double, double, op>, FMXImpl_AOAOA<double2, ushort2, double2, double2, op>, FMXImpl_AOAOA<double3, ushort3, double3, double3, op>, FMXImpl_AOAOA<double4, ushort4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<double, short, uchar, uchar, op>, FMXImpl_AOAOA<double2, short2, uchar2, uchar2, op>, FMXImpl_AOAOA<double3, short3, uchar3, uchar3, op>, FMXImpl_AOAOA<double4, short4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, short, uchar, schar, op>, FMXImpl_AOAOA<double2, short2, uchar2, char2, op>, FMXImpl_AOAOA<double3, short3, uchar3, char3, op>, FMXImpl_AOAOA<double4, short4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<double, short, uchar, ushort, op>, FMXImpl_AOAOA<double2, short2, uchar2, ushort2, op>, FMXImpl_AOAOA<double3, short3, uchar3, ushort3, op>, FMXImpl_AOAOA<double4, short4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, short, uchar, short, op>, FMXImpl_AOAOA<double2, short2, uchar2, short2, op>, FMXImpl_AOAOA<double3, short3, uchar3, short3, op>, FMXImpl_AOAOA<double4, short4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<double, short, uchar, int, op>, FMXImpl_AOAOA<double2, short2, uchar2, int2, op>, FMXImpl_AOAOA<double3, short3, uchar3, int3, op>, FMXImpl_AOAOA<double4, short4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<double, short, uchar, float, op>, FMXImpl_AOAOA<double2, short2, uchar2, float2, op>, FMXImpl_AOAOA<double3, short3, uchar3, float3, op>, FMXImpl_AOAOA<double4, short4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<double, short, uchar, double, op>, FMXImpl_AOAOA<double2, short2, uchar2, double2, op>, FMXImpl_AOAOA<double3, short3, uchar3, double3, op>, FMXImpl_AOAOA<double4, short4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, short, schar, uchar, op>, FMXImpl_AOAOA<double2, short2, char2, uchar2, op>, FMXImpl_AOAOA<double3, short3, char3, uchar3, op>, FMXImpl_AOAOA<double4, short4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, short, schar, schar, op>, FMXImpl_AOAOA<double2, short2, char2, char2, op>, FMXImpl_AOAOA<double3, short3, char3, char3, op>, FMXImpl_AOAOA<double4, short4, char4, char4, op>  },
                    { FMXImpl_AOAOA<double, short, schar, ushort, op>, FMXImpl_AOAOA<double2, short2, char2, ushort2, op>, FMXImpl_AOAOA<double3, short3, char3, ushort3, op>, FMXImpl_AOAOA<double4, short4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, short, schar, short, op>, FMXImpl_AOAOA<double2, short2, char2, short2, op>, FMXImpl_AOAOA<double3, short3, char3, short3, op>, FMXImpl_AOAOA<double4, short4, char4, short4, op>  },
                    { FMXImpl_AOAOA<double, short, schar, int, op>, FMXImpl_AOAOA<double2, short2, char2, int2, op>, FMXImpl_AOAOA<double3, short3, char3, int3, op>, FMXImpl_AOAOA<double4, short4, char4, int4, op>  },
                    { FMXImpl_AOAOA<double, short, schar, float, op>, FMXImpl_AOAOA<double2, short2, char2, float2, op>, FMXImpl_AOAOA<double3, short3, char3, float3, op>, FMXImpl_AOAOA<double4, short4, char4, float4, op>  },
                    { FMXImpl_AOAOA<double, short, schar, double, op>, FMXImpl_AOAOA<double2, short2, char2, double2, op>, FMXImpl_AOAOA<double3, short3, char3, double3, op>, FMXImpl_AOAOA<double4, short4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, short, ushort, uchar, op>, FMXImpl_AOAOA<double2, short2, ushort2, uchar2, op>, FMXImpl_AOAOA<double3, short3, ushort3, uchar3, op>, FMXImpl_AOAOA<double4, short4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, short, ushort, schar, op>, FMXImpl_AOAOA<double2, short2, ushort2, char2, op>, FMXImpl_AOAOA<double3, short3, ushort3, char3, op>, FMXImpl_AOAOA<double4, short4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<double, short, ushort, ushort, op>, FMXImpl_AOAOA<double2, short2, ushort2, ushort2, op>, FMXImpl_AOAOA<double3, short3, ushort3, ushort3, op>, FMXImpl_AOAOA<double4, short4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, short, ushort, short, op>, FMXImpl_AOAOA<double2, short2, ushort2, short2, op>, FMXImpl_AOAOA<double3, short3, ushort3, short3, op>, FMXImpl_AOAOA<double4, short4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<double, short, ushort, int, op>, FMXImpl_AOAOA<double2, short2, ushort2, int2, op>, FMXImpl_AOAOA<double3, short3, ushort3, int3, op>, FMXImpl_AOAOA<double4, short4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<double, short, ushort, float, op>, FMXImpl_AOAOA<double2, short2, ushort2, float2, op>, FMXImpl_AOAOA<double3, short3, ushort3, float3, op>, FMXImpl_AOAOA<double4, short4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<double, short, ushort, double, op>, FMXImpl_AOAOA<double2, short2, ushort2, double2, op>, FMXImpl_AOAOA<double3, short3, ushort3, double3, op>, FMXImpl_AOAOA<double4, short4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, short, short, uchar, op>, FMXImpl_AOAOA<double2, short2, short2, uchar2, op>, FMXImpl_AOAOA<double3, short3, short3, uchar3, op>, FMXImpl_AOAOA<double4, short4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, short, short, schar, op>, FMXImpl_AOAOA<double2, short2, short2, char2, op>, FMXImpl_AOAOA<double3, short3, short3, char3, op>, FMXImpl_AOAOA<double4, short4, short4, char4, op>  },
                    { FMXImpl_AOAOA<double, short, short, ushort, op>, FMXImpl_AOAOA<double2, short2, short2, ushort2, op>, FMXImpl_AOAOA<double3, short3, short3, ushort3, op>, FMXImpl_AOAOA<double4, short4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, short, short, short, op>, FMXImpl_AOAOA<double2, short2, short2, short2, op>, FMXImpl_AOAOA<double3, short3, short3, short3, op>, FMXImpl_AOAOA<double4, short4, short4, short4, op>  },
                    { FMXImpl_AOAOA<double, short, short, int, op>, FMXImpl_AOAOA<double2, short2, short2, int2, op>, FMXImpl_AOAOA<double3, short3, short3, int3, op>, FMXImpl_AOAOA<double4, short4, short4, int4, op>  },
                    { FMXImpl_AOAOA<double, short, short, float, op>, FMXImpl_AOAOA<double2, short2, short2, float2, op>, FMXImpl_AOAOA<double3, short3, short3, float3, op>, FMXImpl_AOAOA<double4, short4, short4, float4, op>  },
                    { FMXImpl_AOAOA<double, short, short, double, op>, FMXImpl_AOAOA<double2, short2, short2, double2, op>, FMXImpl_AOAOA<double3, short3, short3, double3, op>, FMXImpl_AOAOA<double4, short4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, short, int, uchar, op>, FMXImpl_AOAOA<double2, short2, int2, uchar2, op>, FMXImpl_AOAOA<double3, short3, int3, uchar3, op>, FMXImpl_AOAOA<double4, short4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, short, int, schar, op>, FMXImpl_AOAOA<double2, short2, int2, char2, op>, FMXImpl_AOAOA<double3, short3, int3, char3, op>, FMXImpl_AOAOA<double4, short4, int4, char4, op>  },
                    { FMXImpl_AOAOA<double, short, int, ushort, op>, FMXImpl_AOAOA<double2, short2, int2, ushort2, op>, FMXImpl_AOAOA<double3, short3, int3, ushort3, op>, FMXImpl_AOAOA<double4, short4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, short, int, short, op>, FMXImpl_AOAOA<double2, short2, int2, short2, op>, FMXImpl_AOAOA<double3, short3, int3, short3, op>, FMXImpl_AOAOA<double4, short4, int4, short4, op>  },
                    { FMXImpl_AOAOA<double, short, int, int, op>, FMXImpl_AOAOA<double2, short2, int2, int2, op>, FMXImpl_AOAOA<double3, short3, int3, int3, op>, FMXImpl_AOAOA<double4, short4, int4, int4, op>  },
                    { FMXImpl_AOAOA<double, short, int, float, op>, FMXImpl_AOAOA<double2, short2, int2, float2, op>, FMXImpl_AOAOA<double3, short3, int3, float3, op>, FMXImpl_AOAOA<double4, short4, int4, float4, op>  },
                    { FMXImpl_AOAOA<double, short, int, double, op>, FMXImpl_AOAOA<double2, short2, int2, double2, op>, FMXImpl_AOAOA<double3, short3, int3, double3, op>, FMXImpl_AOAOA<double4, short4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, short, float, uchar, op>, FMXImpl_AOAOA<double2, short2, float2, uchar2, op>, FMXImpl_AOAOA<double3, short3, float3, uchar3, op>, FMXImpl_AOAOA<double4, short4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, short, float, schar, op>, FMXImpl_AOAOA<double2, short2, float2, char2, op>, FMXImpl_AOAOA<double3, short3, float3, char3, op>, FMXImpl_AOAOA<double4, short4, float4, char4, op>  },
                    { FMXImpl_AOAOA<double, short, float, ushort, op>, FMXImpl_AOAOA<double2, short2, float2, ushort2, op>, FMXImpl_AOAOA<double3, short3, float3, ushort3, op>, FMXImpl_AOAOA<double4, short4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, short, float, short, op>, FMXImpl_AOAOA<double2, short2, float2, short2, op>, FMXImpl_AOAOA<double3, short3, float3, short3, op>, FMXImpl_AOAOA<double4, short4, float4, short4, op>  },
                    { FMXImpl_AOAOA<double, short, float, int, op>, FMXImpl_AOAOA<double2, short2, float2, int2, op>, FMXImpl_AOAOA<double3, short3, float3, int3, op>, FMXImpl_AOAOA<double4, short4, float4, int4, op>  },
                    { FMXImpl_AOAOA<double, short, float, float, op>, FMXImpl_AOAOA<double2, short2, float2, float2, op>, FMXImpl_AOAOA<double3, short3, float3, float3, op>, FMXImpl_AOAOA<double4, short4, float4, float4, op>  },
                    { FMXImpl_AOAOA<double, short, float, double, op>, FMXImpl_AOAOA<double2, short2, float2, double2, op>, FMXImpl_AOAOA<double3, short3, float3, double3, op>, FMXImpl_AOAOA<double4, short4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, short, double, uchar, op>, FMXImpl_AOAOA<double2, short2, double2, uchar2, op>, FMXImpl_AOAOA<double3, short3, double3, uchar3, op>, FMXImpl_AOAOA<double4, short4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, short, double, schar, op>, FMXImpl_AOAOA<double2, short2, double2, char2, op>, FMXImpl_AOAOA<double3, short3, double3, char3, op>, FMXImpl_AOAOA<double4, short4, double4, char4, op>  },
                    { FMXImpl_AOAOA<double, short, double, ushort, op>, FMXImpl_AOAOA<double2, short2, double2, ushort2, op>, FMXImpl_AOAOA<double3, short3, double3, ushort3, op>, FMXImpl_AOAOA<double4, short4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, short, double, short, op>, FMXImpl_AOAOA<double2, short2, double2, short2, op>, FMXImpl_AOAOA<double3, short3, double3, short3, op>, FMXImpl_AOAOA<double4, short4, double4, short4, op>  },
                    { FMXImpl_AOAOA<double, short, double, int, op>, FMXImpl_AOAOA<double2, short2, double2, int2, op>, FMXImpl_AOAOA<double3, short3, double3, int3, op>, FMXImpl_AOAOA<double4, short4, double4, int4, op>  },
                    { FMXImpl_AOAOA<double, short, double, float, op>, FMXImpl_AOAOA<double2, short2, double2, float2, op>, FMXImpl_AOAOA<double3, short3, double3, float3, op>, FMXImpl_AOAOA<double4, short4, double4, float4, op>  },
                    { FMXImpl_AOAOA<double, short, double, double, op>, FMXImpl_AOAOA<double2, short2, double2, double2, op>, FMXImpl_AOAOA<double3, short3, double3, double3, op>, FMXImpl_AOAOA<double4, short4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<double, int, uchar, uchar, op>, FMXImpl_AOAOA<double2, int2, uchar2, uchar2, op>, FMXImpl_AOAOA<double3, int3, uchar3, uchar3, op>, FMXImpl_AOAOA<double4, int4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, int, uchar, schar, op>, FMXImpl_AOAOA<double2, int2, uchar2, char2, op>, FMXImpl_AOAOA<double3, int3, uchar3, char3, op>, FMXImpl_AOAOA<double4, int4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<double, int, uchar, ushort, op>, FMXImpl_AOAOA<double2, int2, uchar2, ushort2, op>, FMXImpl_AOAOA<double3, int3, uchar3, ushort3, op>, FMXImpl_AOAOA<double4, int4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, int, uchar, short, op>, FMXImpl_AOAOA<double2, int2, uchar2, short2, op>, FMXImpl_AOAOA<double3, int3, uchar3, short3, op>, FMXImpl_AOAOA<double4, int4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<double, int, uchar, int, op>, FMXImpl_AOAOA<double2, int2, uchar2, int2, op>, FMXImpl_AOAOA<double3, int3, uchar3, int3, op>, FMXImpl_AOAOA<double4, int4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<double, int, uchar, float, op>, FMXImpl_AOAOA<double2, int2, uchar2, float2, op>, FMXImpl_AOAOA<double3, int3, uchar3, float3, op>, FMXImpl_AOAOA<double4, int4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<double, int, uchar, double, op>, FMXImpl_AOAOA<double2, int2, uchar2, double2, op>, FMXImpl_AOAOA<double3, int3, uchar3, double3, op>, FMXImpl_AOAOA<double4, int4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, int, schar, uchar, op>, FMXImpl_AOAOA<double2, int2, char2, uchar2, op>, FMXImpl_AOAOA<double3, int3, char3, uchar3, op>, FMXImpl_AOAOA<double4, int4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, int, schar, schar, op>, FMXImpl_AOAOA<double2, int2, char2, char2, op>, FMXImpl_AOAOA<double3, int3, char3, char3, op>, FMXImpl_AOAOA<double4, int4, char4, char4, op>  },
                    { FMXImpl_AOAOA<double, int, schar, ushort, op>, FMXImpl_AOAOA<double2, int2, char2, ushort2, op>, FMXImpl_AOAOA<double3, int3, char3, ushort3, op>, FMXImpl_AOAOA<double4, int4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, int, schar, short, op>, FMXImpl_AOAOA<double2, int2, char2, short2, op>, FMXImpl_AOAOA<double3, int3, char3, short3, op>, FMXImpl_AOAOA<double4, int4, char4, short4, op>  },
                    { FMXImpl_AOAOA<double, int, schar, int, op>, FMXImpl_AOAOA<double2, int2, char2, int2, op>, FMXImpl_AOAOA<double3, int3, char3, int3, op>, FMXImpl_AOAOA<double4, int4, char4, int4, op>  },
                    { FMXImpl_AOAOA<double, int, schar, float, op>, FMXImpl_AOAOA<double2, int2, char2, float2, op>, FMXImpl_AOAOA<double3, int3, char3, float3, op>, FMXImpl_AOAOA<double4, int4, char4, float4, op>  },
                    { FMXImpl_AOAOA<double, int, schar, double, op>, FMXImpl_AOAOA<double2, int2, char2, double2, op>, FMXImpl_AOAOA<double3, int3, char3, double3, op>, FMXImpl_AOAOA<double4, int4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, int, ushort, uchar, op>, FMXImpl_AOAOA<double2, int2, ushort2, uchar2, op>, FMXImpl_AOAOA<double3, int3, ushort3, uchar3, op>, FMXImpl_AOAOA<double4, int4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, int, ushort, schar, op>, FMXImpl_AOAOA<double2, int2, ushort2, char2, op>, FMXImpl_AOAOA<double3, int3, ushort3, char3, op>, FMXImpl_AOAOA<double4, int4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<double, int, ushort, ushort, op>, FMXImpl_AOAOA<double2, int2, ushort2, ushort2, op>, FMXImpl_AOAOA<double3, int3, ushort3, ushort3, op>, FMXImpl_AOAOA<double4, int4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, int, ushort, short, op>, FMXImpl_AOAOA<double2, int2, ushort2, short2, op>, FMXImpl_AOAOA<double3, int3, ushort3, short3, op>, FMXImpl_AOAOA<double4, int4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<double, int, ushort, int, op>, FMXImpl_AOAOA<double2, int2, ushort2, int2, op>, FMXImpl_AOAOA<double3, int3, ushort3, int3, op>, FMXImpl_AOAOA<double4, int4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<double, int, ushort, float, op>, FMXImpl_AOAOA<double2, int2, ushort2, float2, op>, FMXImpl_AOAOA<double3, int3, ushort3, float3, op>, FMXImpl_AOAOA<double4, int4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<double, int, ushort, double, op>, FMXImpl_AOAOA<double2, int2, ushort2, double2, op>, FMXImpl_AOAOA<double3, int3, ushort3, double3, op>, FMXImpl_AOAOA<double4, int4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, int, short, uchar, op>, FMXImpl_AOAOA<double2, int2, short2, uchar2, op>, FMXImpl_AOAOA<double3, int3, short3, uchar3, op>, FMXImpl_AOAOA<double4, int4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, int, short, schar, op>, FMXImpl_AOAOA<double2, int2, short2, char2, op>, FMXImpl_AOAOA<double3, int3, short3, char3, op>, FMXImpl_AOAOA<double4, int4, short4, char4, op>  },
                    { FMXImpl_AOAOA<double, int, short, ushort, op>, FMXImpl_AOAOA<double2, int2, short2, ushort2, op>, FMXImpl_AOAOA<double3, int3, short3, ushort3, op>, FMXImpl_AOAOA<double4, int4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, int, short, short, op>, FMXImpl_AOAOA<double2, int2, short2, short2, op>, FMXImpl_AOAOA<double3, int3, short3, short3, op>, FMXImpl_AOAOA<double4, int4, short4, short4, op>  },
                    { FMXImpl_AOAOA<double, int, short, int, op>, FMXImpl_AOAOA<double2, int2, short2, int2, op>, FMXImpl_AOAOA<double3, int3, short3, int3, op>, FMXImpl_AOAOA<double4, int4, short4, int4, op>  },
                    { FMXImpl_AOAOA<double, int, short, float, op>, FMXImpl_AOAOA<double2, int2, short2, float2, op>, FMXImpl_AOAOA<double3, int3, short3, float3, op>, FMXImpl_AOAOA<double4, int4, short4, float4, op>  },
                    { FMXImpl_AOAOA<double, int, short, double, op>, FMXImpl_AOAOA<double2, int2, short2, double2, op>, FMXImpl_AOAOA<double3, int3, short3, double3, op>, FMXImpl_AOAOA<double4, int4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, int, int, uchar, op>, FMXImpl_AOAOA<double2, int2, int2, uchar2, op>, FMXImpl_AOAOA<double3, int3, int3, uchar3, op>, FMXImpl_AOAOA<double4, int4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, int, int, schar, op>, FMXImpl_AOAOA<double2, int2, int2, char2, op>, FMXImpl_AOAOA<double3, int3, int3, char3, op>, FMXImpl_AOAOA<double4, int4, int4, char4, op>  },
                    { FMXImpl_AOAOA<double, int, int, ushort, op>, FMXImpl_AOAOA<double2, int2, int2, ushort2, op>, FMXImpl_AOAOA<double3, int3, int3, ushort3, op>, FMXImpl_AOAOA<double4, int4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, int, int, short, op>, FMXImpl_AOAOA<double2, int2, int2, short2, op>, FMXImpl_AOAOA<double3, int3, int3, short3, op>, FMXImpl_AOAOA<double4, int4, int4, short4, op>  },
                    { FMXImpl_AOAOA<double, int, int, int, op>, FMXImpl_AOAOA<double2, int2, int2, int2, op>, FMXImpl_AOAOA<double3, int3, int3, int3, op>, FMXImpl_AOAOA<double4, int4, int4, int4, op>  },
                    { FMXImpl_AOAOA<double, int, int, float, op>, FMXImpl_AOAOA<double2, int2, int2, float2, op>, FMXImpl_AOAOA<double3, int3, int3, float3, op>, FMXImpl_AOAOA<double4, int4, int4, float4, op>  },
                    { FMXImpl_AOAOA<double, int, int, double, op>, FMXImpl_AOAOA<double2, int2, int2, double2, op>, FMXImpl_AOAOA<double3, int3, int3, double3, op>, FMXImpl_AOAOA<double4, int4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, int, float, uchar, op>, FMXImpl_AOAOA<double2, int2, float2, uchar2, op>, FMXImpl_AOAOA<double3, int3, float3, uchar3, op>, FMXImpl_AOAOA<double4, int4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, int, float, schar, op>, FMXImpl_AOAOA<double2, int2, float2, char2, op>, FMXImpl_AOAOA<double3, int3, float3, char3, op>, FMXImpl_AOAOA<double4, int4, float4, char4, op>  },
                    { FMXImpl_AOAOA<double, int, float, ushort, op>, FMXImpl_AOAOA<double2, int2, float2, ushort2, op>, FMXImpl_AOAOA<double3, int3, float3, ushort3, op>, FMXImpl_AOAOA<double4, int4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, int, float, short, op>, FMXImpl_AOAOA<double2, int2, float2, short2, op>, FMXImpl_AOAOA<double3, int3, float3, short3, op>, FMXImpl_AOAOA<double4, int4, float4, short4, op>  },
                    { FMXImpl_AOAOA<double, int, float, int, op>, FMXImpl_AOAOA<double2, int2, float2, int2, op>, FMXImpl_AOAOA<double3, int3, float3, int3, op>, FMXImpl_AOAOA<double4, int4, float4, int4, op>  },
                    { FMXImpl_AOAOA<double, int, float, float, op>, FMXImpl_AOAOA<double2, int2, float2, float2, op>, FMXImpl_AOAOA<double3, int3, float3, float3, op>, FMXImpl_AOAOA<double4, int4, float4, float4, op>  },
                    { FMXImpl_AOAOA<double, int, float, double, op>, FMXImpl_AOAOA<double2, int2, float2, double2, op>, FMXImpl_AOAOA<double3, int3, float3, double3, op>, FMXImpl_AOAOA<double4, int4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, int, double, uchar, op>, FMXImpl_AOAOA<double2, int2, double2, uchar2, op>, FMXImpl_AOAOA<double3, int3, double3, uchar3, op>, FMXImpl_AOAOA<double4, int4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, int, double, schar, op>, FMXImpl_AOAOA<double2, int2, double2, char2, op>, FMXImpl_AOAOA<double3, int3, double3, char3, op>, FMXImpl_AOAOA<double4, int4, double4, char4, op>  },
                    { FMXImpl_AOAOA<double, int, double, ushort, op>, FMXImpl_AOAOA<double2, int2, double2, ushort2, op>, FMXImpl_AOAOA<double3, int3, double3, ushort3, op>, FMXImpl_AOAOA<double4, int4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, int, double, short, op>, FMXImpl_AOAOA<double2, int2, double2, short2, op>, FMXImpl_AOAOA<double3, int3, double3, short3, op>, FMXImpl_AOAOA<double4, int4, double4, short4, op>  },
                    { FMXImpl_AOAOA<double, int, double, int, op>, FMXImpl_AOAOA<double2, int2, double2, int2, op>, FMXImpl_AOAOA<double3, int3, double3, int3, op>, FMXImpl_AOAOA<double4, int4, double4, int4, op>  },
                    { FMXImpl_AOAOA<double, int, double, float, op>, FMXImpl_AOAOA<double2, int2, double2, float2, op>, FMXImpl_AOAOA<double3, int3, double3, float3, op>, FMXImpl_AOAOA<double4, int4, double4, float4, op>  },
                    { FMXImpl_AOAOA<double, int, double, double, op>, FMXImpl_AOAOA<double2, int2, double2, double2, op>, FMXImpl_AOAOA<double3, int3, double3, double3, op>, FMXImpl_AOAOA<double4, int4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<double, float, uchar, uchar, op>, FMXImpl_AOAOA<double2, float2, uchar2, uchar2, op>, FMXImpl_AOAOA<double3, float3, uchar3, uchar3, op>, FMXImpl_AOAOA<double4, float4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, float, uchar, schar, op>, FMXImpl_AOAOA<double2, float2, uchar2, char2, op>, FMXImpl_AOAOA<double3, float3, uchar3, char3, op>, FMXImpl_AOAOA<double4, float4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<double, float, uchar, ushort, op>, FMXImpl_AOAOA<double2, float2, uchar2, ushort2, op>, FMXImpl_AOAOA<double3, float3, uchar3, ushort3, op>, FMXImpl_AOAOA<double4, float4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, float, uchar, short, op>, FMXImpl_AOAOA<double2, float2, uchar2, short2, op>, FMXImpl_AOAOA<double3, float3, uchar3, short3, op>, FMXImpl_AOAOA<double4, float4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<double, float, uchar, int, op>, FMXImpl_AOAOA<double2, float2, uchar2, int2, op>, FMXImpl_AOAOA<double3, float3, uchar3, int3, op>, FMXImpl_AOAOA<double4, float4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<double, float, uchar, float, op>, FMXImpl_AOAOA<double2, float2, uchar2, float2, op>, FMXImpl_AOAOA<double3, float3, uchar3, float3, op>, FMXImpl_AOAOA<double4, float4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<double, float, uchar, double, op>, FMXImpl_AOAOA<double2, float2, uchar2, double2, op>, FMXImpl_AOAOA<double3, float3, uchar3, double3, op>, FMXImpl_AOAOA<double4, float4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, float, schar, uchar, op>, FMXImpl_AOAOA<double2, float2, char2, uchar2, op>, FMXImpl_AOAOA<double3, float3, char3, uchar3, op>, FMXImpl_AOAOA<double4, float4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, float, schar, schar, op>, FMXImpl_AOAOA<double2, float2, char2, char2, op>, FMXImpl_AOAOA<double3, float3, char3, char3, op>, FMXImpl_AOAOA<double4, float4, char4, char4, op>  },
                    { FMXImpl_AOAOA<double, float, schar, ushort, op>, FMXImpl_AOAOA<double2, float2, char2, ushort2, op>, FMXImpl_AOAOA<double3, float3, char3, ushort3, op>, FMXImpl_AOAOA<double4, float4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, float, schar, short, op>, FMXImpl_AOAOA<double2, float2, char2, short2, op>, FMXImpl_AOAOA<double3, float3, char3, short3, op>, FMXImpl_AOAOA<double4, float4, char4, short4, op>  },
                    { FMXImpl_AOAOA<double, float, schar, int, op>, FMXImpl_AOAOA<double2, float2, char2, int2, op>, FMXImpl_AOAOA<double3, float3, char3, int3, op>, FMXImpl_AOAOA<double4, float4, char4, int4, op>  },
                    { FMXImpl_AOAOA<double, float, schar, float, op>, FMXImpl_AOAOA<double2, float2, char2, float2, op>, FMXImpl_AOAOA<double3, float3, char3, float3, op>, FMXImpl_AOAOA<double4, float4, char4, float4, op>  },
                    { FMXImpl_AOAOA<double, float, schar, double, op>, FMXImpl_AOAOA<double2, float2, char2, double2, op>, FMXImpl_AOAOA<double3, float3, char3, double3, op>, FMXImpl_AOAOA<double4, float4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, float, ushort, uchar, op>, FMXImpl_AOAOA<double2, float2, ushort2, uchar2, op>, FMXImpl_AOAOA<double3, float3, ushort3, uchar3, op>, FMXImpl_AOAOA<double4, float4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, float, ushort, schar, op>, FMXImpl_AOAOA<double2, float2, ushort2, char2, op>, FMXImpl_AOAOA<double3, float3, ushort3, char3, op>, FMXImpl_AOAOA<double4, float4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<double, float, ushort, ushort, op>, FMXImpl_AOAOA<double2, float2, ushort2, ushort2, op>, FMXImpl_AOAOA<double3, float3, ushort3, ushort3, op>, FMXImpl_AOAOA<double4, float4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, float, ushort, short, op>, FMXImpl_AOAOA<double2, float2, ushort2, short2, op>, FMXImpl_AOAOA<double3, float3, ushort3, short3, op>, FMXImpl_AOAOA<double4, float4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<double, float, ushort, int, op>, FMXImpl_AOAOA<double2, float2, ushort2, int2, op>, FMXImpl_AOAOA<double3, float3, ushort3, int3, op>, FMXImpl_AOAOA<double4, float4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<double, float, ushort, float, op>, FMXImpl_AOAOA<double2, float2, ushort2, float2, op>, FMXImpl_AOAOA<double3, float3, ushort3, float3, op>, FMXImpl_AOAOA<double4, float4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<double, float, ushort, double, op>, FMXImpl_AOAOA<double2, float2, ushort2, double2, op>, FMXImpl_AOAOA<double3, float3, ushort3, double3, op>, FMXImpl_AOAOA<double4, float4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, float, short, uchar, op>, FMXImpl_AOAOA<double2, float2, short2, uchar2, op>, FMXImpl_AOAOA<double3, float3, short3, uchar3, op>, FMXImpl_AOAOA<double4, float4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, float, short, schar, op>, FMXImpl_AOAOA<double2, float2, short2, char2, op>, FMXImpl_AOAOA<double3, float3, short3, char3, op>, FMXImpl_AOAOA<double4, float4, short4, char4, op>  },
                    { FMXImpl_AOAOA<double, float, short, ushort, op>, FMXImpl_AOAOA<double2, float2, short2, ushort2, op>, FMXImpl_AOAOA<double3, float3, short3, ushort3, op>, FMXImpl_AOAOA<double4, float4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, float, short, short, op>, FMXImpl_AOAOA<double2, float2, short2, short2, op>, FMXImpl_AOAOA<double3, float3, short3, short3, op>, FMXImpl_AOAOA<double4, float4, short4, short4, op>  },
                    { FMXImpl_AOAOA<double, float, short, int, op>, FMXImpl_AOAOA<double2, float2, short2, int2, op>, FMXImpl_AOAOA<double3, float3, short3, int3, op>, FMXImpl_AOAOA<double4, float4, short4, int4, op>  },
                    { FMXImpl_AOAOA<double, float, short, float, op>, FMXImpl_AOAOA<double2, float2, short2, float2, op>, FMXImpl_AOAOA<double3, float3, short3, float3, op>, FMXImpl_AOAOA<double4, float4, short4, float4, op>  },
                    { FMXImpl_AOAOA<double, float, short, double, op>, FMXImpl_AOAOA<double2, float2, short2, double2, op>, FMXImpl_AOAOA<double3, float3, short3, double3, op>, FMXImpl_AOAOA<double4, float4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, float, int, uchar, op>, FMXImpl_AOAOA<double2, float2, int2, uchar2, op>, FMXImpl_AOAOA<double3, float3, int3, uchar3, op>, FMXImpl_AOAOA<double4, float4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, float, int, schar, op>, FMXImpl_AOAOA<double2, float2, int2, char2, op>, FMXImpl_AOAOA<double3, float3, int3, char3, op>, FMXImpl_AOAOA<double4, float4, int4, char4, op>  },
                    { FMXImpl_AOAOA<double, float, int, ushort, op>, FMXImpl_AOAOA<double2, float2, int2, ushort2, op>, FMXImpl_AOAOA<double3, float3, int3, ushort3, op>, FMXImpl_AOAOA<double4, float4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, float, int, short, op>, FMXImpl_AOAOA<double2, float2, int2, short2, op>, FMXImpl_AOAOA<double3, float3, int3, short3, op>, FMXImpl_AOAOA<double4, float4, int4, short4, op>  },
                    { FMXImpl_AOAOA<double, float, int, int, op>, FMXImpl_AOAOA<double2, float2, int2, int2, op>, FMXImpl_AOAOA<double3, float3, int3, int3, op>, FMXImpl_AOAOA<double4, float4, int4, int4, op>  },
                    { FMXImpl_AOAOA<double, float, int, float, op>, FMXImpl_AOAOA<double2, float2, int2, float2, op>, FMXImpl_AOAOA<double3, float3, int3, float3, op>, FMXImpl_AOAOA<double4, float4, int4, float4, op>  },
                    { FMXImpl_AOAOA<double, float, int, double, op>, FMXImpl_AOAOA<double2, float2, int2, double2, op>, FMXImpl_AOAOA<double3, float3, int3, double3, op>, FMXImpl_AOAOA<double4, float4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, float, float, uchar, op>, FMXImpl_AOAOA<double2, float2, float2, uchar2, op>, FMXImpl_AOAOA<double3, float3, float3, uchar3, op>, FMXImpl_AOAOA<double4, float4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, float, float, schar, op>, FMXImpl_AOAOA<double2, float2, float2, char2, op>, FMXImpl_AOAOA<double3, float3, float3, char3, op>, FMXImpl_AOAOA<double4, float4, float4, char4, op>  },
                    { FMXImpl_AOAOA<double, float, float, ushort, op>, FMXImpl_AOAOA<double2, float2, float2, ushort2, op>, FMXImpl_AOAOA<double3, float3, float3, ushort3, op>, FMXImpl_AOAOA<double4, float4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, float, float, short, op>, FMXImpl_AOAOA<double2, float2, float2, short2, op>, FMXImpl_AOAOA<double3, float3, float3, short3, op>, FMXImpl_AOAOA<double4, float4, float4, short4, op>  },
                    { FMXImpl_AOAOA<double, float, float, int, op>, FMXImpl_AOAOA<double2, float2, float2, int2, op>, FMXImpl_AOAOA<double3, float3, float3, int3, op>, FMXImpl_AOAOA<double4, float4, float4, int4, op>  },
                    { FMXImpl_AOAOA<double, float, float, float, op>, FMXImpl_AOAOA<double2, float2, float2, float2, op>, FMXImpl_AOAOA<double3, float3, float3, float3, op>, FMXImpl_AOAOA<double4, float4, float4, float4, op>  },
                    { FMXImpl_AOAOA<double, float, float, double, op>, FMXImpl_AOAOA<double2, float2, float2, double2, op>, FMXImpl_AOAOA<double3, float3, float3, double3, op>, FMXImpl_AOAOA<double4, float4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, float, double, uchar, op>, FMXImpl_AOAOA<double2, float2, double2, uchar2, op>, FMXImpl_AOAOA<double3, float3, double3, uchar3, op>, FMXImpl_AOAOA<double4, float4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, float, double, schar, op>, FMXImpl_AOAOA<double2, float2, double2, char2, op>, FMXImpl_AOAOA<double3, float3, double3, char3, op>, FMXImpl_AOAOA<double4, float4, double4, char4, op>  },
                    { FMXImpl_AOAOA<double, float, double, ushort, op>, FMXImpl_AOAOA<double2, float2, double2, ushort2, op>, FMXImpl_AOAOA<double3, float3, double3, ushort3, op>, FMXImpl_AOAOA<double4, float4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, float, double, short, op>, FMXImpl_AOAOA<double2, float2, double2, short2, op>, FMXImpl_AOAOA<double3, float3, double3, short3, op>, FMXImpl_AOAOA<double4, float4, double4, short4, op>  },
                    { FMXImpl_AOAOA<double, float, double, int, op>, FMXImpl_AOAOA<double2, float2, double2, int2, op>, FMXImpl_AOAOA<double3, float3, double3, int3, op>, FMXImpl_AOAOA<double4, float4, double4, int4, op>  },
                    { FMXImpl_AOAOA<double, float, double, float, op>, FMXImpl_AOAOA<double2, float2, double2, float2, op>, FMXImpl_AOAOA<double3, float3, double3, float3, op>, FMXImpl_AOAOA<double4, float4, double4, float4, op>  },
                    { FMXImpl_AOAOA<double, float, double, double, op>, FMXImpl_AOAOA<double2, float2, double2, double2, op>, FMXImpl_AOAOA<double3, float3, double3, double3, op>, FMXImpl_AOAOA<double4, float4, double4, double4, op>  },
                },
            },
            {
                {
                    { FMXImpl_AOAOA<double, double, uchar, uchar, op>, FMXImpl_AOAOA<double2, double2, uchar2, uchar2, op>, FMXImpl_AOAOA<double3, double3, uchar3, uchar3, op>, FMXImpl_AOAOA<double4, double4, uchar4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, double, uchar, schar, op>, FMXImpl_AOAOA<double2, double2, uchar2, char2, op>, FMXImpl_AOAOA<double3, double3, uchar3, char3, op>, FMXImpl_AOAOA<double4, double4, uchar4, char4, op>  },
                    { FMXImpl_AOAOA<double, double, uchar, ushort, op>, FMXImpl_AOAOA<double2, double2, uchar2, ushort2, op>, FMXImpl_AOAOA<double3, double3, uchar3, ushort3, op>, FMXImpl_AOAOA<double4, double4, uchar4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, double, uchar, short, op>, FMXImpl_AOAOA<double2, double2, uchar2, short2, op>, FMXImpl_AOAOA<double3, double3, uchar3, short3, op>, FMXImpl_AOAOA<double4, double4, uchar4, short4, op>  },
                    { FMXImpl_AOAOA<double, double, uchar, int, op>, FMXImpl_AOAOA<double2, double2, uchar2, int2, op>, FMXImpl_AOAOA<double3, double3, uchar3, int3, op>, FMXImpl_AOAOA<double4, double4, uchar4, int4, op>  },
                    { FMXImpl_AOAOA<double, double, uchar, float, op>, FMXImpl_AOAOA<double2, double2, uchar2, float2, op>, FMXImpl_AOAOA<double3, double3, uchar3, float3, op>, FMXImpl_AOAOA<double4, double4, uchar4, float4, op>  },
                    { FMXImpl_AOAOA<double, double, uchar, double, op>, FMXImpl_AOAOA<double2, double2, uchar2, double2, op>, FMXImpl_AOAOA<double3, double3, uchar3, double3, op>, FMXImpl_AOAOA<double4, double4, uchar4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, double, schar, uchar, op>, FMXImpl_AOAOA<double2, double2, char2, uchar2, op>, FMXImpl_AOAOA<double3, double3, char3, uchar3, op>, FMXImpl_AOAOA<double4, double4, char4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, double, schar, schar, op>, FMXImpl_AOAOA<double2, double2, char2, char2, op>, FMXImpl_AOAOA<double3, double3, char3, char3, op>, FMXImpl_AOAOA<double4, double4, char4, char4, op>  },
                    { FMXImpl_AOAOA<double, double, schar, ushort, op>, FMXImpl_AOAOA<double2, double2, char2, ushort2, op>, FMXImpl_AOAOA<double3, double3, char3, ushort3, op>, FMXImpl_AOAOA<double4, double4, char4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, double, schar, short, op>, FMXImpl_AOAOA<double2, double2, char2, short2, op>, FMXImpl_AOAOA<double3, double3, char3, short3, op>, FMXImpl_AOAOA<double4, double4, char4, short4, op>  },
                    { FMXImpl_AOAOA<double, double, schar, int, op>, FMXImpl_AOAOA<double2, double2, char2, int2, op>, FMXImpl_AOAOA<double3, double3, char3, int3, op>, FMXImpl_AOAOA<double4, double4, char4, int4, op>  },
                    { FMXImpl_AOAOA<double, double, schar, float, op>, FMXImpl_AOAOA<double2, double2, char2, float2, op>, FMXImpl_AOAOA<double3, double3, char3, float3, op>, FMXImpl_AOAOA<double4, double4, char4, float4, op>  },
                    { FMXImpl_AOAOA<double, double, schar, double, op>, FMXImpl_AOAOA<double2, double2, char2, double2, op>, FMXImpl_AOAOA<double3, double3, char3, double3, op>, FMXImpl_AOAOA<double4, double4, char4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, double, ushort, uchar, op>, FMXImpl_AOAOA<double2, double2, ushort2, uchar2, op>, FMXImpl_AOAOA<double3, double3, ushort3, uchar3, op>, FMXImpl_AOAOA<double4, double4, ushort4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, double, ushort, schar, op>, FMXImpl_AOAOA<double2, double2, ushort2, char2, op>, FMXImpl_AOAOA<double3, double3, ushort3, char3, op>, FMXImpl_AOAOA<double4, double4, ushort4, char4, op>  },
                    { FMXImpl_AOAOA<double, double, ushort, ushort, op>, FMXImpl_AOAOA<double2, double2, ushort2, ushort2, op>, FMXImpl_AOAOA<double3, double3, ushort3, ushort3, op>, FMXImpl_AOAOA<double4, double4, ushort4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, double, ushort, short, op>, FMXImpl_AOAOA<double2, double2, ushort2, short2, op>, FMXImpl_AOAOA<double3, double3, ushort3, short3, op>, FMXImpl_AOAOA<double4, double4, ushort4, short4, op>  },
                    { FMXImpl_AOAOA<double, double, ushort, int, op>, FMXImpl_AOAOA<double2, double2, ushort2, int2, op>, FMXImpl_AOAOA<double3, double3, ushort3, int3, op>, FMXImpl_AOAOA<double4, double4, ushort4, int4, op>  },
                    { FMXImpl_AOAOA<double, double, ushort, float, op>, FMXImpl_AOAOA<double2, double2, ushort2, float2, op>, FMXImpl_AOAOA<double3, double3, ushort3, float3, op>, FMXImpl_AOAOA<double4, double4, ushort4, float4, op>  },
                    { FMXImpl_AOAOA<double, double, ushort, double, op>, FMXImpl_AOAOA<double2, double2, ushort2, double2, op>, FMXImpl_AOAOA<double3, double3, ushort3, double3, op>, FMXImpl_AOAOA<double4, double4, ushort4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, double, short, uchar, op>, FMXImpl_AOAOA<double2, double2, short2, uchar2, op>, FMXImpl_AOAOA<double3, double3, short3, uchar3, op>, FMXImpl_AOAOA<double4, double4, short4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, double, short, schar, op>, FMXImpl_AOAOA<double2, double2, short2, char2, op>, FMXImpl_AOAOA<double3, double3, short3, char3, op>, FMXImpl_AOAOA<double4, double4, short4, char4, op>  },
                    { FMXImpl_AOAOA<double, double, short, ushort, op>, FMXImpl_AOAOA<double2, double2, short2, ushort2, op>, FMXImpl_AOAOA<double3, double3, short3, ushort3, op>, FMXImpl_AOAOA<double4, double4, short4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, double, short, short, op>, FMXImpl_AOAOA<double2, double2, short2, short2, op>, FMXImpl_AOAOA<double3, double3, short3, short3, op>, FMXImpl_AOAOA<double4, double4, short4, short4, op>  },
                    { FMXImpl_AOAOA<double, double, short, int, op>, FMXImpl_AOAOA<double2, double2, short2, int2, op>, FMXImpl_AOAOA<double3, double3, short3, int3, op>, FMXImpl_AOAOA<double4, double4, short4, int4, op>  },
                    { FMXImpl_AOAOA<double, double, short, float, op>, FMXImpl_AOAOA<double2, double2, short2, float2, op>, FMXImpl_AOAOA<double3, double3, short3, float3, op>, FMXImpl_AOAOA<double4, double4, short4, float4, op>  },
                    { FMXImpl_AOAOA<double, double, short, double, op>, FMXImpl_AOAOA<double2, double2, short2, double2, op>, FMXImpl_AOAOA<double3, double3, short3, double3, op>, FMXImpl_AOAOA<double4, double4, short4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, double, int, uchar, op>, FMXImpl_AOAOA<double2, double2, int2, uchar2, op>, FMXImpl_AOAOA<double3, double3, int3, uchar3, op>, FMXImpl_AOAOA<double4, double4, int4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, double, int, schar, op>, FMXImpl_AOAOA<double2, double2, int2, char2, op>, FMXImpl_AOAOA<double3, double3, int3, char3, op>, FMXImpl_AOAOA<double4, double4, int4, char4, op>  },
                    { FMXImpl_AOAOA<double, double, int, ushort, op>, FMXImpl_AOAOA<double2, double2, int2, ushort2, op>, FMXImpl_AOAOA<double3, double3, int3, ushort3, op>, FMXImpl_AOAOA<double4, double4, int4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, double, int, short, op>, FMXImpl_AOAOA<double2, double2, int2, short2, op>, FMXImpl_AOAOA<double3, double3, int3, short3, op>, FMXImpl_AOAOA<double4, double4, int4, short4, op>  },
                    { FMXImpl_AOAOA<double, double, int, int, op>, FMXImpl_AOAOA<double2, double2, int2, int2, op>, FMXImpl_AOAOA<double3, double3, int3, int3, op>, FMXImpl_AOAOA<double4, double4, int4, int4, op>  },
                    { FMXImpl_AOAOA<double, double, int, float, op>, FMXImpl_AOAOA<double2, double2, int2, float2, op>, FMXImpl_AOAOA<double3, double3, int3, float3, op>, FMXImpl_AOAOA<double4, double4, int4, float4, op>  },
                    { FMXImpl_AOAOA<double, double, int, double, op>, FMXImpl_AOAOA<double2, double2, int2, double2, op>, FMXImpl_AOAOA<double3, double3, int3, double3, op>, FMXImpl_AOAOA<double4, double4, int4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, double, float, uchar, op>, FMXImpl_AOAOA<double2, double2, float2, uchar2, op>, FMXImpl_AOAOA<double3, double3, float3, uchar3, op>, FMXImpl_AOAOA<double4, double4, float4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, double, float, schar, op>, FMXImpl_AOAOA<double2, double2, float2, char2, op>, FMXImpl_AOAOA<double3, double3, float3, char3, op>, FMXImpl_AOAOA<double4, double4, float4, char4, op>  },
                    { FMXImpl_AOAOA<double, double, float, ushort, op>, FMXImpl_AOAOA<double2, double2, float2, ushort2, op>, FMXImpl_AOAOA<double3, double3, float3, ushort3, op>, FMXImpl_AOAOA<double4, double4, float4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, double, float, short, op>, FMXImpl_AOAOA<double2, double2, float2, short2, op>, FMXImpl_AOAOA<double3, double3, float3, short3, op>, FMXImpl_AOAOA<double4, double4, float4, short4, op>  },
                    { FMXImpl_AOAOA<double, double, float, int, op>, FMXImpl_AOAOA<double2, double2, float2, int2, op>, FMXImpl_AOAOA<double3, double3, float3, int3, op>, FMXImpl_AOAOA<double4, double4, float4, int4, op>  },
                    { FMXImpl_AOAOA<double, double, float, float, op>, FMXImpl_AOAOA<double2, double2, float2, float2, op>, FMXImpl_AOAOA<double3, double3, float3, float3, op>, FMXImpl_AOAOA<double4, double4, float4, float4, op>  },
                    { FMXImpl_AOAOA<double, double, float, double, op>, FMXImpl_AOAOA<double2, double2, float2, double2, op>, FMXImpl_AOAOA<double3, double3, float3, double3, op>, FMXImpl_AOAOA<double4, double4, float4, double4, op>  },
                },
                {
                    { FMXImpl_AOAOA<double, double, double, uchar, op>, FMXImpl_AOAOA<double2, double2, double2, uchar2, op>, FMXImpl_AOAOA<double3, double3, double3, uchar3, op>, FMXImpl_AOAOA<double4, double4, double4, uchar4, op>  },
                    { FMXImpl_AOAOA<double, double, double, schar, op>, FMXImpl_AOAOA<double2, double2, double2, char2, op>, FMXImpl_AOAOA<double3, double3, double3, char3, op>, FMXImpl_AOAOA<double4, double4, double4, char4, op>  },
                    { FMXImpl_AOAOA<double, double, double, ushort, op>, FMXImpl_AOAOA<double2, double2, double2, ushort2, op>, FMXImpl_AOAOA<double3, double3, double3, ushort3, op>, FMXImpl_AOAOA<double4, double4, double4, ushort4, op>  },
                    { FMXImpl_AOAOA<double, double, double, short, op>, FMXImpl_AOAOA<double2, double2, double2, short2, op>, FMXImpl_AOAOA<double3, double3, double3, short3, op>, FMXImpl_AOAOA<double4, double4, double4, short4, op>  },
                    { FMXImpl_AOAOA<double, double, double, int, op>, FMXImpl_AOAOA<double2, double2, double2, int2, op>, FMXImpl_AOAOA<double3, double3, double3, int3, op>, FMXImpl_AOAOA<double4, double4, double4, int4, op>  },
                    { FMXImpl_AOAOA<double, double, double, float, op>, FMXImpl_AOAOA<double2, double2, double2, float2, op>, FMXImpl_AOAOA<double3, double3, double3, float3, op>, FMXImpl_AOAOA<double4, double4, double4, float4, op>  },
                    { FMXImpl_AOAOA<double, double, double, double, op>, FMXImpl_AOAOA<double2, double2, double2, double2, op>, FMXImpl_AOAOA<double3, double3, double3, double3, op>, FMXImpl_AOAOA<double4, double4, double4, double4, op>  },
                },
            },
        }
    };

    const int stype = CV_MAKETYPE(std::min(std::min(_src1.depth(), _src2.depth()), _src3.depth()), _src1.channels());
    const int sdepth = CV_MAT_DEPTH(stype);
    const int cn = CV_MAT_CN(stype);
    const int wdepth = std::max(sdepth, CV_32F);
    const int wtype = CV_MAKETYPE(wdepth, cn);

    bool reconstruct = cn <= 4;

    GpuMat src1(_src1), src2(_src2), src3(_src3);

    if(!reconstruct)
        _dst.create(_src1.size(), dtype == -1 ? wtype : CV_MAKETYPE(CV_MAT_DEPTH(dtype), cn ) );
    else
    {
        _dst.create(_src1.rows, _src1.cols * cn, dtype == -1 ? wdepth : CV_MAT_DEPTH(dtype) );

        src1 = src1.reshape(1);
        src2 = src2.reshape(1);
        src3 = src3.reshape(1);
    }

    function_type fun = functions[src1.depth()][src2.depth()][src3.depth()][_dst.depth()][_dst.channels()-1];

    fun(src1, src2, src3, _dst, _mask, _stream);

    if(reconstruct)
        _dst = _dst.reshape(cn);
}

template<int op>
void fmxImpl(const GpuMat& _src1, const Scalar& _src2, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{

}

template<int op>
void fmxImpl(const Scalar& _src1, const GpuMat& _src2, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{

}

template<int op>
void fmxImpl(const Scalar& _src1, const Scalar& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{

}

template<int op>
void fmxImpl(const Scalar& _src1, const GpuMat& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{

}

template<int op>
void fmxImpl(const GpuMat& _src1, const Scalar& _src2, const GpuMat& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{

}

template<int op>
void fmxImpl(const GpuMat& _src1, const GpuMat& _src2, const Scalar& _src3, GpuMat& _dst, const GpuMat& _mask, int dtype, Stream& _stream)
{

}


} // device


} // cuda

} // cv

#endif // if 0
