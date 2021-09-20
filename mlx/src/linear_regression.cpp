#include "opencv2/mlx.hpp"
#include "opencv2/xcore/linalg.hpp"
#include "precomp.hpp"


using namespace std;

namespace cv
{

using namespace ml;

namespace mlx
{

namespace
{


class LinearRegressionImpl CV_FINAL : public LinearRegression
{

public:

    inline LinearRegressionImpl():flag(0), N(0), intercept(0){}
    inline LinearRegressionImpl(const String& filename) : LinearRegressionImpl(){ this->read(filename);}

    virtual ~LinearRegressionImpl() = default;

    virtual void setCoef(InputArray _src) CV_OVERRIDE{ this->weights = getInput<Mat>(_src); this->N=-1;}
    virtual void getCoef(OutputArray _dst) const CV_OVERRIDE{ _dst.assign(this->weights); }
    virtual float getIntercept() const CV_OVERRIDE { return this->intercept; }
    virtual void setIntercept(const float& _src) CV_OVERRIDE{ this->intercept = _src; }

    virtual int getVarCount() const CV_OVERRIDE { return this->N; }
    virtual bool isTrained() const CV_OVERRIDE { return !this->weights.empty(); }
    virtual bool isClassifier() const CV_OVERRIDE { return false; }
    virtual String getDefaultName() const CV_OVERRIDE { return "extra_ml_linear_regression"; }

    virtual bool train( const Ptr<TrainData>& trainData, int=0 ) CV_OVERRIDE;
    virtual bool train( InputArray samples, int layout, InputArray responses ) CV_OVERRIDE;
    virtual float predict(InputArray samples, OutputArray results, int flags=0) const CV_OVERRIDE;

    virtual void clear() CV_OVERRIDE;

    virtual void read( const FileNode& fn ) CV_OVERRIDE;
    virtual void write( FileStorage& fs ) const  CV_OVERRIDE;

    virtual void read( const String& filename );
    virtual void write( const String& filename) const;

private:
    int flag;
    int N;
    float intercept;
    Mat weights;



    template<class T>
    bool train_( InputArray& samples, int layout, InputArray& responses );

    template<class T>
    float predict_(InputArray& samples, OutputArray& results, int flags) const;

    template<class T>
    float get_intercept_(const T& mu_X, const Scalar& mu_y, const T& coeffs, const int& wdepth);
};

bool LinearRegressionImpl::train(const Ptr<TrainData> &trainData, int flag)
{
    Mat X = trainData->getTrainSamples();
    Mat y = trainData->getTrainResponses();
    if(trainData->getLayout() == COL_SAMPLE)
        X = X.t();

    this->flag = flag;

    return this->train(X, ROW_SAMPLE, y);
}

template<class T>
bool LinearRegressionImpl::train_( InputArray& samples, int layout, InputArray& responses )
{
    T tmp;

    T X = getInput<T>(samples);
    T y = getInput<T>(responses);

    T mu_X;
    Scalar mu_y;

    const int wdepth = std::max(X.depth(), CV_32F);

    int N(0);

    if(layout == COL_SAMPLE)
        transpose(X, X);

    if(y.rows == 1)
        transpose(y, y);

//    this->intercept = lstsq(X, y, tmp);

    // First centre the data.

    reduce(X, mu_X, 0, REDUCE_AVG, wdepth);
    centreToTheMeanAxis0(X, mu_X, X);

    mu_y = mean(y);
    subtract(y, mu_y, y, noArray(), wdepth);


    // Compute the weights.

    linalg::lstsq(X, y, tmp, noArray(), noArray());


    N = X.rows;

    // Update the weights if needed.

    if(this->flag == UPDATE_MODEL)
    {
        double alpha(1. / static_cast<double>(this->N)), beta(1. / static_cast<double>(N) );

        T tmp_w;

        this->weights.copyTo(tmp_w);

        addWeighted(tmp_w, alpha, tmp, beta, 0., tmp);

        N+=this->N;
    }

    transpose(tmp, tmp);

    tmp.copyTo(this->weights);
    this->N = N;

    this->flag = 0;

    // Compute the interception coefficient.

    this->intercept = this->get_intercept_<T>(mu_X, mu_y, tmp, wdepth);


    return true;
}

template<class T>
float LinearRegressionImpl::get_intercept_(const T& mu_X, const Scalar& mu_y, const T& coefs, const int& wdepth)
{
    T tmp;
    Mat tmp2;

    if(mu_X.rows > 1)
        gemm(mu_X.row(0), coefs, -1., T(1,1, wdepth, mu_y(0)), 1., tmp, GEMM_2_T);
    else
        gemm(mu_X, coefs, -1., T(1,1, wdepth, mu_y(0)), 1., tmp, GEMM_2_T);

    tmp.copyTo(tmp2);

    return tmp2.at<float>(0);
}

bool LinearRegressionImpl::train( InputArray samples, int layout, InputArray responses )
{
    CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1) && ( (responses.rows() == 1) || (responses.cols() == 1) ) );

    return samples.isUMat() || responses.isUMat() ? this->train_<UMat>(samples, layout, responses) : this->train_<Mat>(samples, layout, responses);
}

// support function.
template<class T>
float get_y(const T&);

template<>
float get_y<Mat>(const Mat& tmp)
{
    return Mat1f(tmp)(0);
}

template<>
float get_y<UMat>(const UMat& tmp)
{
    Mat tmp2;

    tmp.copyTo(tmp2);

    return Mat1f(tmp2)(0);
}

template<class T>
float LinearRegressionImpl::predict_(InputArray& samples, OutputArray& results, int ) const
{

    T X = getInput<T>(samples);

    T mu_X, y;

    const int wdepth = X.depth();

    // Centre the samples.

    reduce(X, mu_X, 0, REDUCE_AVG);

    // Compute the prediction.

    gemm(X, this->weights, 1., T(X.rows, this->weights.rows, wdepth, this->intercept), 1., y, GEMM_2_T);

    if(results.needed())
        results.assign(y);

    return X.rows == 1 ? abs(get_y(y)) : 0.f;
}

float LinearRegressionImpl::predict(InputArray samples, OutputArray results, int flags) const
{
    CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1));

    return samples.isUMat() || (results.needed() && results.isUMat()) ? this->predict_<UMat>(samples, results, flags) : this->predict_<Mat>(samples, results, flags);
}

void LinearRegressionImpl::clear()
{
    this->weights.release();
    this->N = 0;
    this->intercept = 0.f;
    this->flag = 0;
}

void LinearRegressionImpl::read(const FileNode &fn)
{
    if(fn.name() != this->getDefaultName() && fn.name() != "extra_cuda_ml_linear_regression")
        CV_Error(cv::Error::StsError, "The filename provided, was not generated for a linear regression algorithm.");

    this->clear();

    this->N = (int)fn["N"];
    this->intercept = (float)fn["intercept"];
    fn["weights"] >> this->weights;
}

void LinearRegressionImpl::write(FileStorage &fs) const
{
    fs << "N" << this->N;
    fs << "intercept" << this->intercept;
    fs << "weights" << this->weights;
}

void LinearRegressionImpl::read(const String & _filename)
{
    if(!utils::fs::exists(_filename))
        CV_Error(cv::Error::StsError, "The filename provided, does not exists.");

    String fn = _filename;

    // First step, ensure make every letter lower, to ease the checks.
    locale loc;
    for_each(fn.begin(), fn.end(), bind(tolower<typename String::value_type>,placeholders::_1,loc));
    bool file_type_found(false);

    if(fn.find(".yml") != String::npos || fn.find(".xml") != String::npos || fn.find(".json") != String::npos)
    {

        file_type_found = true;

        FileStorage fs(_filename, FileStorage::READ);

        CV_Assert(fs.isOpened());

        this->read(fs.getFirstTopLevelNode());
    }

    if(fn.find(".h5") != String::npos)
    {
        file_type_found = true;

        auto fid = hdf::open(fn);

        CV_Assert(fid->hlexists("weights") && fid->hlexists("N") && fid->hlexists("intercept") );

        fid->dsread(this->weights, "weights");

        Mat tmp;

        fid->dsread(tmp, "intercept");
        this->intercept = tmp.at<float>(0);

        fid->dsread(tmp, "N");
        this->N = tmp.at<int>(0);
    }

    if(!file_type_found)
        CV_Error(cv::Error::StsError, "The file format provided is not supported yet! Only XML, YAML, JSON, and HDF5 file format are supported.");
}

void LinearRegressionImpl::write(const String& _filename) const
{
    String fn = _filename;

    locale loc;
    for_each(fn.begin(), fn.end(), bind(tolower<typename String::value_type>,placeholders::_1,loc));
    bool file_type_found(false);

    if(fn.find(".yml") != String::npos || fn.find(".xml") != String::npos || fn.find(".json") != String::npos)
    {
        file_type_found = true;

        this->save(_filename);
    }

    if(fn.find(".h5") != String::npos)
    {

        file_type_found = true;

        auto fid = hdf::open(fn);

        fid->dswrite(this->weights, "weights");

        Mat tmp(1,1, CV_32F);
        tmp.at<float>(0) = this->intercept;
        fid->dswrite(tmp.clone(), "intercept");

        tmp.create(1,1,CV_32S);
        tmp.at<int>(0) = this->N;
        fid->dswrite(tmp.clone(), "N");
    }

    if(!file_type_found)
        CV_Error(cv::Error::StsError, "The file format provided is not supported yet! Only XML, YAML, JSON, and HDF5 file format are supported.");
}

} // anonymous

Ptr<LinearRegression> LinearRegression::create(){ return makePtr<LinearRegressionImpl>(); }

Ptr<LinearRegression> LinearRegression::load(const String& filepath){ return makePtr<LinearRegressionImpl>(filepath); }


} // mlx

} // cv
