#include "opencv2/xcore.hpp"

namespace cv
{

void parallel_for_(const Range& range, std::function<void(const int&)> functor, double nstripes)
{
    parallel_for_(range, ParallelLoopBodyLambdaWrapperInt(functor), nstripes);
}

void parallel_for_(const int& start, const int& end, const ParallelLoopBody& fun)
{
    parallel_for_(Range(start, end), fun);
}

void parallel_for_(const int& start, const int& end, const std::function<void(const Range&)>& fun)
{
    parallel_for_(Range(start, end), fun);
}

void parallel_for_(const int& start, const int& end, const std::function<void(const int&)>& fun)
{
    parallel_for_(Range(start, end), fun);
}





////////////////////////////////////////
/// More parallel loop options       ///
////////////////////////////////////////


#ifdef HAVE_TBB
void highPrioriyParallelFor(const Range& range, const ParallelLoopBody& fun)
{
    tbb::task_group_context tcg;

    tcg.set_priority(tbb::priority_high);

    tbb::parallel_for(tbb::blocked_range<int>(range.start, range.end),[&fun](tbb::blocked_range<int>& _range)->void{ return fun(Range(_range.begin(), _range.end())); });
}
#endif

void highPrioriyParallelFor(const Range& range, const std::function<void(const Range&)>& fun)
{
#ifdef HAVE_TBB
    highPrioriyParallelFor(range, ParallelLoopBodyLambdaWrapper(fun) );
#else
    parallel_for_(range, ParallelLoopBodyLambdaWrapper(fun) );
#endif
}

void highPrioriyParallelFor(const Range& range, const std::function<void(const int&)>& fun)
{
#ifdef HAVE_TBB
    highPrioriyParallelFor(range, ParallelLoopBodyLambdaWrapperInt(fun));
#else
    parallel_for_(range, ParallelLoopBodyLambdaWrapperInt(fun) );
#endif
}

#ifdef HAVE_TBB
tbb::blocked_range<int> convertRange(const Range& range){ return tbb::blocked_range<int>(range.start, range.end); }
#endif

} // cv
