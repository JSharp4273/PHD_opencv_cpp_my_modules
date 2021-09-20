#include "opencv2/cudaxcore/linalg.hpp"
#include "opencv2/xcore/template/arguments_io.hpp"

#include "opencv2/cudaarithm.hpp"


namespace cv
{

namespace cuda
{



int lstsq(InputArray _X, InputArray _y, OutputArray _x, OutputArray _residues, OutputArray _s, Stream& stream)
{
    CV_Assert(_X.isGpuMat() && (_X.channels() == 1) && _y.isGpuMat() && (_y.channels() == 1) );

    // Input management

        GpuMat X(_X.getGpuMat()), y(_y.getGpuMat());

        const int m(X.rows), n(X.cols);

        const int wdepth = std::max(X.depth(), CV_32F);

        const double rcond = wdepth == CV_32F ? static_cast<double>(std::numeric_limits<float>::epsilon()) : std::numeric_limits<double>::epsilon();

    // Compute the SVD.

        GpuMat s, u, vh;

        SVDecomp(X, s, u, vh, stream);

        GpuMat  s1;

        divide(1., s, s1, 1., wdepth, stream);

    //  number of singular values and matrix rank

        double mx(0.);

        minMax(s, nullptr, &mx);

        GpuMat mask;

        compare(s, rcond * mx, mask, CMP_GT, stream);
        bitwise_and(mask, 1., mask);

        mask.convertTo(mask, CV_32F, stream);
        multiply(s1, mask, s1, 1., wdepth, stream);// why there is "dtype" argument with an assertion that state src1.type() == src2.type() ?

        transpose(s1, s1, stream);

        const int rank = countNonZero(mask);

    //  Solve the least-squares solution

        GpuMat z, x;

        gemm(y, u, 1., noArray(), 0., z, GEMM_1_T, stream);
        multiply(s1, z, z, 1., wdepth, stream);
        gemm(vh, z, 1., noArray(), 0., x, GEMM_1_T | GEMM_2_T, stream);

    //  Output management

        if(_residues.needed())
        {
            if(m<n || rank!=n)
                _residues.clear();
            else
            {
                GpuMat e;

                gemm(X, x, -1., y,1., e, 0, stream);
                transpose(e,e, stream);
                multiply(e, e, e, 1., wdepth, stream);
                reduce(e, e, 1, REDUCE_SUM, wdepth, stream);

                e.copyTo(_residues, stream);
            }
        }

        if(_x.needed())
            x.copyTo(_x, stream);

        if(_s.needed())
            s.copyTo(_s, stream);

        return rank;
}


} // cuda

} // cv
