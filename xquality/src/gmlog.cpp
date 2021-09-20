#include "opencv2/xquality/nonfree.hpp"
#include "precomp.hpp"


namespace cv
{

namespace xquality
{

namespace
{

class QualityGMLOGImpl_ CV_FINAL : public QualityGMLOG
{
public:

    inline QualityGMLOGImpl_(const String& model_file_path, const String& range_file_path ):
        QualityGMLOGImpl_(ml::SVM::load(model_file_path)
                          , FileStorage(range_file_path, FileStorage::READ)["range"].mat())
    {}

    inline QualityGMLOGImpl_(const Ptr<ml::SVM>& model, const Mat& range):
        _model(model),
        _range(range)
    {}

    virtual ~QualityGMLOGImpl_() CV_OVERRIDE = default;

    Scalar compute(InputArray _src) CV_OVERRIDE
    {
        return this->compute(_src, String(), String());
    }

    Scalar compute(InputArray _src, const String& model_file_path, const String& range_file_path ) CV_OVERRIDE
    {
        CV_Assert(_src.channels()<=4);

        if(!model_file_path.empty())
            this->_model = ml::SVM::load(model_file_path);
        if(!range_file_path.empty())
            this->_range = FileStorage(range_file_path, FileStorage::READ)["range"].mat();

        Mat src = _src.getMat();

        Scalar ret;


        if(src.channels()>1)
        {
            std::vector<Mat> cns;

            split(src, cns);

            Mat Ig, tmp, features;

            if(src.channels()==3)
                cvtColor(src, Ig, COLOR_BGR2GRAY);
            else
                cvtColor(src, Ig, COLOR_BGRA2GRAY);

            cns.push_back(Ig);

            for(int i=0;i<4;i++)
            {
                QualityGMLOG::computeFeatures(cns.at(i), features);

                this->_model->predict(features, tmp);

                ret(i) = saturate_cast<double>(tmp.at<float>(0));
            }
        }
        else
        {
            Mat features, tmp;

            QualityGMLOG::computeFeatures(src, features);

            this->_model->predict(features, tmp);

            ret(0) = saturate_cast<double>(tmp.at<float>(0));
        }

        return ret;
    }

private:

    Ptr<ml::SVM> _model;
    Mat _range;

};

template<class T>
void abs_(const T&, T&);

template<>
void abs_<Mat>(const Mat& a, Mat& b)
{
    b = abs(a);
}

template<>
void abs_<UMat>(const UMat& a, UMat& b)
{
    absdiff(a,0,b);
}

template<class T>
struct Compute_GMLOG_Features_t
{


    inline void compute_gradient_magnitude(const T& _src, T& _dst) const
    {
        static const Mat1f winx = (Mat1f(5,5)<<-0.000017813857,  -0.000114096552,   0.000114096552,   0.000017813857,   0.000000044264,
                                               -0.007186623036,  -0.046029834294,   0.046029834294,   0.007186623036,   0.000017857313,
                                               -0.053102360779,  -0.340117027823,   0.340117027823,   0.053102360779,   0.000131948685,
                                               -0.007186623036,  -0.046029834294,   0.046029834294,   0.007186623036,   0.000017857313,
                                               -0.000017813857,  -0.000114096552,   0.000114096552,   0.000017813857,   0.000000044264);

        static const Mat1f winy = (Mat1f(5,5)<< -0.000017813857,  -0.007186623036,  -0.053102360779,  -0.007186623036,  -0.000017813857,
                                                -0.000114096552,  -0.046029834294,  -0.340117027823,  -0.046029834294,  -0.000114096552,
                                                 0.000114096552,   0.046029834294,   0.340117027823,   0.046029834294,   0.000114096552,
                                                 0.000017813857,   0.007186623036,   0.053102360779,   0.007186623036,   0.000017813857,
                                                 0.000000044264,   0.000017857313,   0.000131948685,   0.000017857313,   0.000000044264);

        const int wdepth = std::max(_src.depth(), CV_32F);

        T dx, dy;

        filter2D(_src, dx, wdepth, winx);
        filter2D(_src, dy, wdepth, winy);

        magnitude(dx, dy, _dst);
    }

    inline void compute_laplacian_of_gaussian(const T& _src, T& _dst) const
    {
        static const Mat1f k = (Mat1f(5,5)<<0.00000095165,   0.00023035291,   0.00132384822,   0.00023035291,   0.00000095165,
                                            0.00023035291,   0.03097699870,   0.07629692706,   0.03097699870,   0.00023035291,
                                            0.00132384822,   0.07629692706,  -0.56376227422,   0.07629692706,   0.00132384822,
                                            0.00023035291,   0.03097699870,   0.07629692706,   0.03097699870,   0.00023035291,
                                            0.00000095165,   0.00023035291,   0.00132384822,   0.00023035291,   0.00000095165);

        const int wdepth = std::max(_src.depth(), CV_32F);

        T tmp;

        filter2D(_src, tmp, wdepth, k);

        abs_(tmp, _dst);
    }

    inline void getNMap(const T& _gm, const T& _log, T& _dst) const
    {
        static const Mat1f k = (Mat1f(7,7)<<0.000019652,   0.000239409,   0.001072958,   0.001769009,   0.001072958,   0.000239409,   0.000019652,
                                0.000239409,   0.002916603,   0.013071308,   0.021550943,   0.013071308,   0.002916603,   0.000239409,
                                0.001072958,   0.013071308,   0.058581536,   0.096584625,   0.058581536,   0.013071308,   0.001072958,
                                0.001769009,   0.021550943,   0.096584625,   0.159241126,   0.096584625,   0.021550943,   0.001769009,
                                0.001072958,   0.013071308,   0.058581536,   0.096584625,   0.058581536,   0.013071308,   0.001072958,
                                0.000239409,   0.002916603,   0.013071308,   0.021550943,   0.013071308,   0.002916603,   0.000239409,
                                0.000019652,   0.000239409,   0.001072958,   0.001769009,   0.001072958,   0.000239409,   0.000019652);

        const int wdepth = std::max(std::max(_gm.depth(), _log.depth()), CV_32F);

        T tmp;

        addWeighted(_gm, 0.5, _log, 0.5, 0., tmp, wdepth);

        filter2D(tmp, tmp, wdepth, k);
        add(tmp, 0.2, tmp);
        sqrt(tmp, _dst);
    }

    template<class D>
    void compute_features(const T& src, D& _features) const
    {
        T gm, log, NMap;

        compute_gradient_magnitude(src, gm);
        compute_laplacian_of_gaussian(src, log);

        // Compute the Joint Adaptative Normalisation.
        getNMap(gm, log, NMap);

        divide(gm, NMap, gm);
        abs_(gm, gm);

        divide(log, NMap, log);
        abs_(log, log);

        // enhance the contrast.
        T gm_qun, log_qun;

        divide(gm, 0.2, gm_qun);
        divide(log, 0.2, log_qun);

        ceil(gm_qun, gm_qun);
        ceil(log_qun, log_qun);

        // normalise each quantified type fo data in 10 bins
        normalize(gm_qun, gm_qun, 1., 10., NORM_MINMAX, CV_8U);
        normalize(log_qun, log_qun, 1., 10., NORM_MINMAX, CV_8U);

        Mat m_gm_qun(toMat(gm_qun)), m_lg_qun(toMat(log_qun));

        Mat N1;

        std::vector<Mat> xy = {m_gm_qun, m_lg_qun};
        // compute the joint adaptative histogram
        calcHist(xy, {0, 0}, noArray(), N1 , {10, 10}, {1.f, 10.f, 1.f, 10.f});

        // compute the features from the marginal probabilities.
        N1/=(sum(N1)(0));

        Mat1f NG(N1.rows,1), NL(1,N1.cols);

        for(int r=0; r<N1.rows; r++)
        {
            const float* it_N1 = N1.ptr<float>(r);

            for(int c=0; c<N1.cols; c++, it_N1++)
            {
                float v = *it_N1;

                NG(r) += v;
                NL(c) += v;
            }
        }
        // compute the features from the conditional probabilities.
        Mat1f cp_GL_H(N1.rows,1), cp_LG_H(1,N1.cols);

        double sm_GL(0.), sm_LG(0.);

        for(int r=0; r<N1.rows; r++)
        {
            const float* it_N1 = N1.ptr<float>(r);

            for(int c=0; c<N1.cols; c++, it_N1++)
            {
                float v = *it_N1;

                float v_GL = v/(NL(c)+1e-4f);
                float v_LG = v/(NG(r)+1e-4f);

                cp_GL_H(r) += v_GL;
                cp_LG_H(c) += v_LG;

                sm_GL += saturate_cast<double>(v_GL);
                sm_LG += saturate_cast<double>(v_LG);
            }
        }

        float sm_GL_f = saturate_cast<float>(sm_GL);
        float sm_LG_f = saturate_cast<float>(sm_LG);

        for(int i=0; i<N1.rows; i++)
        {
            cp_GL_H(i) /= sm_GL_f;
            cp_LG_H(i) /= sm_LG_f;
        }
        // construct the feature vector.
        Mat1f features(1,40);

        std::copy(NG.begin(), NG.end(), features.begin());
        std::copy(NL.begin(), NL.end(), features.begin() + 10);
        std::copy(cp_GL_H.begin(), cp_GL_H.end(), features.begin() + 20);
        std::copy(cp_LG_H.begin(), cp_LG_H.end(), features.begin() + 30);

        features.copyTo(_features);
    }

    inline void operator()(InputArray& _img, OutputArray& _features) const
    {
        T src = getInput<T>(_img);

        this->compute_features(src, _features);
    }

    inline T operator()(const T& _src) const
    {
        T features;

        this->compute_features(_src, features);

        return features;
    }
};

} // anonymous

void QualityGMLOG::computeFeatures(InputArray img, OutputArray features)
{
    CV_Assert(img.isMat() || img.isMatVector() || img.isUMat() || img.isUMatVector());
    if(img.isMat() || img.isUMat())
    {
        func<Compute_GMLOG_Features_t>(img, features);
    }
    else
    {
        features.create(img.size(), img.type());

        if(img.isMatVector())
        {
            Compute_GMLOG_Features_t<Mat> op;

            std::vector<Mat> src;
            std::vector<Mat> dst;

            img.getMatVector(src);
            features.getMatVector(dst);

            std::transform(src.begin(), src.end(), dst.begin(), op);
        }
        else
        {
            Compute_GMLOG_Features_t<UMat> op;

            std::vector<UMat> src;
            std::vector<UMat> dst;

            img.getUMatVector(src);
            features.getUMatVector(dst);

            std::transform(src.begin(), src.end(), dst.begin(), op);
        }
    }

}

Ptr<QualityGMLOG> QualityGMLOG::create( const String& model_file_path, const String& range_file_path )
{
    return makePtr<QualityGMLOGImpl_>(model_file_path, range_file_path);
}

Ptr<QualityGMLOG> QualityGMLOG::create( const Ptr<cv::ml::SVM>& model, const Mat& range )
{
    return makePtr<QualityGMLOGImpl_>(model, range);
}

String QualityGMLOG::getDefaultName() const{ return "QualityGMLOG";}

} // quality

} // cv
