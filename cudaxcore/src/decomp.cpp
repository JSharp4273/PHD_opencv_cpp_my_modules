#include <arrayfire.h>

#include "opencv2/cudaxcore/linalg.hpp"
#include "opencv2/cudaxcore/utils.hpp"
#include "opencv2/cudaxcore/arrayfire.hpp"

#include "precomp.hpp"

namespace cv
{

namespace cuda
{

void SVDecomp(InputArray _X, OutputArray _u, OutputArray _s, OutputArray _vt, Stream& stream)
{
    CV_Assert(_X.isGpuMat() && _X.channels() == 1 && _X.depth() >= CV_32F);

    GpuMat X = _X.getGpuMat();    

    af::array aX;
    af::array aU, aS, aVt;

    // management of the multi-device case.
    af::setDevice(cuda::getDevice());

    arrayfire::GpuMat2Array(X, aX, stream);

    af::svd(aS, aU, aVt, aX);

    if(_s.needed())
    {
        CV_Assert(_s.isGpuMat());

        if(aS.dims(0) == X.rows && aS.dims(1) == X.cols)
            arrayfire::Array2GpuMat(aS, _s.getGpuMatRef(), stream);
        else
        {
            GpuMat tmp;

            arrayfire::Array2GpuMat(aS, tmp, stream);

            tmp(Rect(0,0,X.cols, X.rows)).copyTo(_s);
        }
    }

    if(_u.needed())
    {
        CV_Assert(_u.isGpuMat());

        arrayfire::Array2GpuMat(aU, _u.getGpuMatRef(), stream);
    }

    if(_vt.needed())
    {
        CV_Assert(_vt.isGpuMat());

        arrayfire::Array2GpuMat(aVt, _vt.getGpuMatRef(), stream);
    }
}

void QR(InputArray _X, OutputArray _Q, OutputArray _R, OutputArray _tau, Stream& stream)
{
    CV_Assert(_X.isGpuMat() && _X.channels() == 1 && _X.depth() >= CV_32F);

    af::array aX;
    af::array aQ;
    af::array aR;
    af::array atau;

    // management of the multi-device case.
    af::setDevice(cuda::getDevice());

    arrayfire::GpuMat2Array(_X.getGpuMat(), aX, stream);

    af::qr(aQ, aR, atau, aX);

    if(_Q.needed())
    {
        CV_Assert(_Q.isGpuMat());

        arrayfire::Array2GpuMat(aQ, _Q.getGpuMatRef(), stream);
    }

    if(_R.needed())
    {
        CV_Assert(_R.isGpuMat());

        arrayfire::Array2GpuMat(aR, _R.getGpuMatRef(), stream);
    }

    if(_tau.needed())
    {
        CV_Assert(_tau.isGpuMat());

        arrayfire::Array2GpuMat(atau, _tau.getGpuMatRef(), stream);
    }
}

void LU(InputArray _X, OutputArray _L, OutputArray _U, OutputArray _pivot, Stream& stream)
{
    CV_Assert(_X.isGpuMat() && _X.channels() == 1 && _X.depth() >= CV_32F);

    af::array aX;
    af::array aL;
    af::array aU;
    af::array aP;

    // management of the multi-device case.
    af::setDevice(cuda::getDevice());

    arrayfire::GpuMat2Array(_X.getGpuMat(), aX, stream);

    af::lu(aL, aU, aP, aX);

    if(_L.needed())
    {
        CV_Assert(_L.isGpuMat());

        arrayfire::Array2GpuMat(aL, _L.getGpuMatRef(), stream);
    }

    if(_U.needed())
    {
        CV_Assert(_U.isGpuMat());

        arrayfire::Array2GpuMat(aU, _U.getGpuMatRef(), stream);
    }

    if(_pivot.needed())
    {
        CV_Assert(_pivot.isGpuMat());

        arrayfire::Array2GpuMat(aP, _pivot.getGpuMatRef(), stream);
    }


}

} // cuda

} // cv
