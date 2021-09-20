#include "opencv2/cudaml.hpp"

namespace cv
{

namespace cuda
{

bool StatModelAsync::empty() const{ return !this->isTrained(); }

bool StatModelAsync::train( InputArray , int , InputArray , Stream& )
{
    CV_Error(Error::StsNotImplemented,"This functionality has been implemented yet.");
}

} // cuda

} // cv
