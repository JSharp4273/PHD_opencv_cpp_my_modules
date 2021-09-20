#include "opencv2/xcore.hpp"
#include <memory>

using namespace std;

namespace cv
{

Vec2d getMinMax(InputArray _src)
{
    Vec2d ret(0., 0.);

    minMaxLoc(_src, addressof(ret(0)), addressof(ret(1)));

    return ret;
}


} // cv
