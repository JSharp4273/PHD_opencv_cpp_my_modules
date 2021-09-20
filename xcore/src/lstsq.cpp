#include "opencv2/xcore/linalg.hpp"
#include "opencv2/xcore.hpp"
#include "opencv2/xcore/template/arguments_io.hpp"
#include "opencv2/xcore/template/intrin.hpp"

#ifndef CV_DISABLE_OPTIMIZATION
#include <atomic>
#endif

namespace cv
{

namespace linalg
{

namespace
{

template<class T>
int apply_cutoff(T& s, T& s1, const double& cutoff, const int& wdepth)
{
    T mask;
    compare(s, cutoff, mask, CMP_GT);
//    divide(mask, 255., mask, 1., wdepth);
    bitwise_and(mask, 1., mask);
    multiply(s1, mask, s1, 1., wdepth);
    transpose(s1, s1);

    return countNonZero(mask);
}


#ifndef CV_DISABLE_OPTIMIZATION

template<class T>
class ParallelApplyCutoff CV_FINAL : public virtual ParallelLoopBody
{
public:

    typedef T value_type;
    typedef T& reference;
    typedef T* pointer;
    typedef const T* const_pointer;

    inline ParallelApplyCutoff(const Mat_<T>& _s, const Mat_<T>& _s1, const T& _cutoff, Mat_<T>& _dst, int& _cnt):
        s(_s),
        s1(_s1),
        cutoff(_cutoff),
        dst(_dst),
        cnt(_cnt),
        a_cnt(0),
        vec_cols(_s.cols - (_s.cols%vector_type::nlanes))
    {}

    virtual ~ParallelApplyCutoff() { this->cnt = this->a_cnt;}

    virtual void operator()(const Range& range) const
    {
        int local_cnt(0);
        vector_type v_cnt = vx_setzeros<value_type>();
        vector_type v_ones = vx_setall<value_type>(static_cast<T>(1.));
        vector_type v_cutoff = vx_setall<value_type>(this->cutoff);

        for(int r=range.start; r<range.end; r++)
        {
            const_pointer it_s = this->s[r];
            const_pointer it_s1 = this->s1[r];
            pointer it_dst = this->dst[r];

            int c=0;

            for(;c<this->vec_cols; c+=inc, it_s+=inc, it_s1+=inc, it_dst+=inc)
            {
                vector_type v_s = vx_load(it_s);
                vector_type v_s1 = vx_load(it_s1);

                vector_type v_mask = (v_s > v_cutoff) ;

                vx_store(it_dst, v_mask & v_s1);

                v_cnt += v_mask & v_ones;
            }

            vx_cleanup();

            for(;c<this->s.cols; c++, it_s++, it_s1++, it_dst++)
            {
                if(*it_s > this->cutoff)
                {
                    local_cnt++;
                }
                else
                {
                    *it_dst = *it_s1;
                }
            }
        }

        local_cnt += v_reduce_sum(v_cnt);

        this->a_cnt += local_cnt;
    }

private:

    typedef typename Type2Vec_Traits<T>::vec_type vector_type;

    static const int inc = vector_type::nlanes;

    const Mat_<T>& s;
    const Mat_<T>& s1;
    const T cutoff;
    Mat_<T>& dst;
    int& cnt;
    mutable std::atomic_int a_cnt;
    const int vec_cols;

};

template<class T>
int worker_apply_cutoff(const Mat& _s, Mat& _s1, const double& cutoff)
{
    Mat_<T> s(_s), s1(_s1);
    int rank(0);
#ifdef HAVE_TBB
    highPrioriyParallelFor(Range(0,s.rows), ParallelApplyCutoff<T>(s, s1, static_cast<T>(cutoff), s1, rank));
#else
    ParallelApplyCutoff<T>(s, s1, static_cast<T>(cutoff), s1, rank)(Range(0,s.rows));
#endif
    transpose(s1, _s1);

    return rank;
}

template<>
int apply_cutoff<Mat>(Mat& s, Mat& s1, const double& cutoff, const int& wdepth)
{

    typedef int(*function_type)(const Mat&, Mat&, const double&);

    static const function_type funcs[2] = {worker_apply_cutoff<float>, worker_apply_cutoff<double>};

    function_type fun(nullptr);

    fun = funcs[wdepth - CV_32F];

    CV_Assert(fun);

    return fun(s,s1,cutoff);

}
#endif

// This code was inspired by the Cupy implementation of the lstsq function.
template <class T>
int lstsq_(InputArray& _X, InputArray& _y, OutputArray& _x, OutputArray& _residues, OutputArray& _s)
{

// Input management

    T X(getInput<T>(_X)), y(getInput<T>(_y));

    const int m(X.rows), n(X.cols);

    const int wdepth = std::max(X.depth(), CV_32F);

    const double rcond = wdepth == CV_32F ? static_cast<double>(std::numeric_limits<float>::epsilon()) : std::numeric_limits<double>::epsilon();

// Compute the SVD.

    T s, u, vh;

    SVDecomp(X, s, u, vh);

    T  s1;

    divide(1., s, s1, 1., wdepth);

//  number of singular values and matrix rank

    double mx(0.);

    minMaxLoc(s, nullptr, &mx);

#if 0
    T mask;

    compare(s, rcond * mx, mask, CMP_GT);
//    divide(mask, 255., mask, 1., wdepth);
    bitwise_and(mask, 1., mask);
    multiply(s1, mask, s1, 1., wdepth);
    transpose(s1, s1);

    const int rank = countNonZero(mask);
#else
    const int rank = apply_cutoff(s, s1, rcond * mx, wdepth);
#endif
//  Solve the least-squares solution

    T z, x;

    gemm(y, u, 1., noArray(), 0., z, GEMM_1_T);
    multiply(s1, z, z, 1., wdepth);
    gemm(vh, z, 1., noArray(), 0., x, GEMM_1_T | GEMM_2_T);

//  Output management

    if(_residues.needed())
    {
        if(m<n || rank!=n)
            _residues.clear();
        else
        {
            T e;

            gemm(X, x, -1., y,1., e);
            transpose(e,e);
            multiply(e, e, e, 1.,wdepth);
            reduce(e, e, 1, REDUCE_SUM, wdepth);

            _residues.assign(e);
        }
    }

    if(_x.needed())
        _x.assign(x);

    if(_s.needed())
        _s.assign(s);

    return rank;
}

} // anonymous

int lstsq(InputArray _X, InputArray _y, OutputArray _x, OutputArray _residues, OutputArray _s)
{
    CV_Assert(isMatrix(_X) && (_X.channels() == 1) && isMatrix(_y) && (_y.channels() == 1) );

    return (_x.needed() && _x.isUMat()) || (_residues.needed() && _residues.isUMat()) || (_s.isUMat() && _s.needed()) ? lstsq_<UMat>(_X, _y, _x, _residues, _s) : lstsq_<Mat>(_X, _y, _x, _residues, _s);
}

} // linalg

} // cv
