#pragma once

#ifndef XQUALITY_HPP
#define XQUALITY_HPP

#include "opencv2/core.hpp"
#include "opencv2/quality.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"

namespace cv
{

using quality::QualityBase;

namespace xquality
{

/**
@brief Full reference mean absolute error algorithm  https://en.wikipedia.org/wiki/Mean_absolute_error
*/
class CV_EXPORTS_W QualityMAE : public QualityBase {
public:

    /** @brief Computes MAE for reference images supplied in class constructor and provided comparison images
    @param cmpImgs Comparison image(s)
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to potentially max float (worst)
    */
    CV_WRAP Scalar compute( InputArrayOfArrays cmpImgs ) CV_OVERRIDE;

    /** @brief Implements Algorithm::empty()  */
    CV_WRAP bool empty() const CV_OVERRIDE { return _ref.empty() && QualityBase::empty(); }

    /** @brief Implements Algorithm::clear()  */
    CV_WRAP void clear() CV_OVERRIDE { _ref = _mat_type(); QualityBase::clear(); }

    /**
    @brief Create an object which calculates quality
    @param ref input image to use as the reference for comparison
    */
    CV_WRAP static Ptr<QualityMAE> create(InputArray ref);

    /**
    @brief static method for computing quality
    @param ref reference image
    @param cmp comparison image=
    @param qualityMap output quality map, or cv::noArray()
    @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to max float (worst)
    */
    CV_WRAP static Scalar compute( InputArray ref, InputArray cmp, OutputArray qualityMap );

protected:

    /** @brief Reference image, converted to internal mat type */
    QualityBase::_mat_type _ref;

    /**
    @brief Constructor
    @param ref reference image, converted to internal type
    */
    QualityMAE(QualityBase::_mat_type ref)
        : _ref(std::move(ref))
    {}

};  // QualityMAE


} // quality

} // cv

#endif
