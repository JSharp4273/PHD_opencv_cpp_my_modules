#include "opencv2/xquality/nonfree.hpp"
#include "precomp.hpp"


namespace cv
{

using namespace quality;

namespace xquality
{

namespace
{

class ParallelMatDec : public ParallelLoopBody
{
public:

    inline ParallelMatDec(const Mat1f& _ref, const Mat1f& _target, const Size& _blockSize, Mat1f& _features):
        ref(_ref),
        target(_target),
        blockSize(_blockSize),
        features(_features)
    {}

    virtual ~ParallelMatDec() CV_OVERRIDE = default;

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        SVD svd;

        Mat s_ref, s_tgt, tmp;

        for(int r=range.start, rr = range.start * this->blockSize.height; r<range.end; r++, rr+=this->blockSize.height)
            for(int c=0, cc=0; c<this->features.cols; c++, cc+=this->blockSize.width)
            {
                Rect roi(cc, rr, this->blockSize.width, this->blockSize.height);

                Mat sub_ref = this->ref(roi);
                Mat sub_tgt = this->target(roi);

                svd.compute(sub_ref, s_ref, noArray(), noArray(), SVD::NO_UV);
                svd.compute(sub_tgt, s_tgt, noArray(), noArray(), SVD::NO_UV);

                subtract(s_ref, s_tgt, tmp);
                multiply(tmp, tmp, tmp);

                this->features(r,c) = std::sqrt(sum(tmp)(0));
            }
    }

private:

    const Mat1f& ref;
    const Mat1f& target;
    const Size& blockSize;

    Mat1f& features;

    mutable Mutex mtx;
};

class ParallelMedian : public ParallelLoopBody
{
public:

    inline ParallelMedian(const Mat1f& _features, Mat1f& _mpr):
        features(_features),
        medianPerRow(_mpr)
    {}

    virtual ~ParallelMedian() CV_OVERRIDE = default;

    virtual void operator()(const Range& range)const CV_OVERRIDE
    {
        Mat1f row(1, this->features.cols, 0.f);

        for(int r=range.start; r<range.end; r++)
        {
            std::memcpy(row.data, this->features.ptr(r), row.step);

            std::sort(row.begin(), row.end());

            int argmed = cvFloor(static_cast<float>(row.cols) / 2.f);

            this->medianPerRow(r) = row(argmed);
        }
    }

private:

    const Mat1f& features;
    Mat1f& medianPerRow;
};

double median(const Mat1f& features)
{
    Mat1f tmp(features.rows, 1, 0.f);
#ifndef HAVE_HIGH_PRIORITY_PARFOR
    parallel_for_(Range(0, features.rows), ParallelMedian(features, tmp));
#else
    highPrioriyParallelFor(Range(0, features.rows), ParallelMedian(features, tmp));
#endif
    std::sort(tmp.begin(), tmp.end());

    int argmed = cvFloor(static_cast<float>(tmp.cols) / 2.f);

    return static_cast<double>(tmp(argmed));
}

double blockSVD(const Mat& _ref, const Mat& _target, const Size& blockSize, Mat1f& quality, Mat1f& features)
{
    Mat1f ref = _ref;
    Mat1f target = _target;

    Size gridSize(divUp(_ref.cols, blockSize.width), divUp(_ref.rows, blockSize.height));
    Size newSize(gridSize.width * blockSize.width, gridSize.height * blockSize.height);

    Mat1f tmp = Mat1f::zeros(newSize);

    ref.copyTo(tmp(Rect(Point(), _ref.size() ) ) );
    ref = tmp.clone();

    tmp = 0;

    target.copyTo(tmp(Rect(Point(), _target.size() ) ) );
    target = tmp.clone();

    tmp.release();

    features = Mat1f::zeros(gridSize);
#ifndef HAVE_HIGH_PRIORITY_PARFOR
    parallel_for_(Range(0, gridSize.height), ParallelMatDec(ref, target, blockSize, features) );
#else
    highPrioriyParallelFor(Range(0, gridSize.height), ParallelMatDec(ref, target, blockSize, features) );
#endif
//    (ParallelMatDec(ref, target, blockSize, features))(Range(0, gridSize.height));

    Mat tmp2;
    normalize(features, tmp2, 0, 255, NORM_MINMAX, CV_8U);
    quality = tmp2;
    features = tmp2.clone();
    tmp = quality.clone();

    resize(quality, quality, _ref.size());


    double medval = median(features);

    tmp = abs(tmp - medval);
    tmp /= static_cast<double>( (static_cast<float>(ref.rows)/static_cast<float>(blockSize.height)) * (static_cast<float>(ref.cols)/static_cast<float>(blockSize.width)) );

    return sum(tmp)(0);
}

} // anonymous

class QualityBlockSVDImpl_ CV_FINAL: public QualityBlockSVD
{

public:

    typedef QualityBase::_mat_type _mat_type;

    /**
    @brief Constructor
    @param ref reference image, converted to internal type
    */
    inline QualityBlockSVDImpl_(_mat_type ref, const Size& blockSize)
        : _ref(std::move(ref)),
          _features(),
          _blockSize(blockSize)
    {}

    virtual cv::Scalar compute( InputArrayOfArrays cmpImgs ) CV_OVERRIDE;

    virtual ~QualityBlockSVDImpl_() CV_OVERRIDE = default;

    virtual void clear() CV_OVERRIDE
    {
        QualityBase::clear();

        this->_ref.release();
        this->_features.release();
        this->_qualityMap.release();
        this->_blockSize = Size(8,8);
    }

    virtual bool empty() const CV_OVERRIDE { return _ref.empty() && QualityBase::empty(); }


    virtual Size getBlockSize() const CV_OVERRIDE { return this->_blockSize; }
    virtual void setBlockSize(const Size& size) CV_OVERRIDE { this->_blockSize = size; }

    virtual void getFeaturesMatrix(OutputArray _featuresMatrix) const { _featuresMatrix.assign(this->_features); }

    virtual Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap) CV_OVERRIDE;

private:

    /** @brief Reference image, converted to internal mat type */
    _mat_type _ref;

    /** @brief Features matrix, as an internal mat type */
    _mat_type _features;

    Size _blockSize;

};



Ptr<QualityBlockSVD> QualityBlockSVD::create(InputArray ref, const Size& blockSize)
{
    return makePtr<QualityBlockSVDImpl_>( ref.kind() != _InputArray::NONE ? quality_utils::expand_mat<QualityBlockSVD::_mat_type>(ref) : QualityBlockSVD::_mat_type(), blockSize);
}


Scalar QualityBlockSVDImpl_::compute(InputArray _ref, InputArray _target, OutputArray qualityMap)
{
    CV_Assert( ( (_ref.type() == CV_8UC1) || (_ref.type() == CV_8UC3) || (_ref.type() == CV_32FC1) || (_ref.type() == CV_32FC3) ) &&
               ( (_target.type() == CV_8UC1) || (_target.type() == CV_8UC3) || (_target.type() == CV_32FC1) || (_target.type() == CV_32FC3) ) &&
               ( _ref.type() == _target.type() ) &&
               ( _ref.isMat() || _ref.isUMat() ) &&
               ( _target.isMat() || _target.isUMat() ) &&
               ( qualityMap.isMat() || qualityMap.isUMat() || qualityMap.kind() == _OutputArray::NONE )
               );



    Mat ref = _ref.getMat();
    Mat target = _target.getMat();
    Mat1f quality = Mat1f::zeros(ref.size());


    Scalar ret;

    if(ref.channels() > 0)
    {
        std::vector<Mat> cns_ref, cns_target, tmp_features;

        split(ref, cns_ref);
        split(target, cns_target);

        tmp_features.resize(cns_ref.size());

        for(size_t i=0; i<cns_ref.size(); i++)
        {
            Mat1f tmp = Mat1f::zeros(ref.size());
            Mat1f f;

            ret(static_cast<int>(i)) = blockSVD(cns_ref.at(i), cns_target.at(i), this->_blockSize, tmp, f);

            tmp_features.at(i) = f;

            quality+=tmp;
        }

        quality/=static_cast<double>(cns_ref.size());

        cns_ref.clear();
        cns_target.clear();

        merge(tmp_features, this->_features);
    }
    else
    {
        Mat1f tmp;

        ret(0) = blockSVD(ref, target, this->_blockSize, quality, tmp);

        tmp.copyTo(this->_features);

    }

    if(qualityMap.needed())
        qualityMap.assign(quality);

    return ret;
}

Scalar QualityBlockSVDImpl_::compute( InputArrayOfArrays cmpImgs )
{
    CV_Assert(this->_ref.empty() || cmpImgs.isMatVector() || cmpImgs.isUMatVector());

    return this->compute(this->_ref, cmpImgs, noArray());
}


String QualityBlockSVD::getDefaultName() const{ return "QualityBlockSVD"; }


} // quality

} // cv
