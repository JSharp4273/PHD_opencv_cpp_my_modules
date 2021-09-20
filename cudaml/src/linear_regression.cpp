#include "opencv2/cudaml.hpp"
#include "precomp.hpp"

using namespace std;

namespace cv
{

namespace cuda
{

namespace
{

class LinearRegressionImpl CV_FINAL : public LinearRegression
{

public:

    inline LinearRegressionImpl():flag(0), N(0), intercept(0){}
    inline LinearRegressionImpl(const String& filename) : LinearRegressionImpl(){ this->read(filename);}

    virtual ~LinearRegressionImpl() = default;

    virtual void setCoef(InputArray _src, Stream& _stream) CV_OVERRIDE;
    virtual void getCoef(OutputArray _dst, Stream& _stream) const CV_OVERRIDE;
    virtual float getIntercept() const CV_OVERRIDE { return this->intercept; }
    virtual void setIntercept(const float& _src) CV_OVERRIDE{ this->intercept = _src; }

    virtual int getVarCount() const CV_OVERRIDE { return this->N; }
    virtual bool isTrained() const CV_OVERRIDE { return !this->weights.empty(); }
    virtual bool isClassifier() const CV_OVERRIDE { return false; }
    virtual String getDefaultName() const CV_OVERRIDE { return "extra_cuda_ml_linear_regression"; }

    virtual bool train( InputArray samples, int layout, InputArray responses, Stream& stream ) CV_OVERRIDE;
    virtual float predict(InputArray samples, OutputArray results, int flags, Stream& stream) const CV_OVERRIDE;

    virtual void clear() CV_OVERRIDE;

    virtual void read( const FileNode& fn ) CV_OVERRIDE;
    virtual void write( FileStorage& fs ) const  CV_OVERRIDE;

    virtual void read( const String& filename );
    virtual void write( const String& filename) const;

private:
    int flag;
    int N;
    float intercept;
    GpuMat weights;

    float get_intercept_(const GpuMat& mu_X, const Scalar& mu_y, const GpuMat& coeffs, const int& wdepth, Stream& stream);
};

void LinearRegressionImpl::setCoef(InputArray _src, Stream &_stream)
{
    this->N = 1;

    if(!_src.isGpuMat())
        this->weights.upload(_src, _stream);
    else
        this->weights = _src.getGpuMat();
}

void LinearRegressionImpl::getCoef(OutputArray _dst, Stream &_stream) const
{
    if(_dst.isGpuMat())
        this->weights.copyTo(_dst, _stream);
    else
        this->weights.download(_dst, _stream);
}

float LinearRegressionImpl::get_intercept_(const GpuMat& mu_X, const Scalar& mu_y, const GpuMat& coefs, const int& wdepth, Stream& stream)
{
    GpuMat tmp;
    Mat tmp2;

    if(mu_X.rows > 1)
        gemm(mu_X.row(0), coefs, -1., GpuMat(1,1, wdepth, mu_y(0)), 1., tmp, GEMM_2_T, stream);
    else
        gemm(mu_X, coefs, -1., GpuMat(1,1, wdepth, mu_y(0)), 1., tmp, GEMM_2_T, stream);

    tmp.download(tmp2, stream);

    if(tmp2.depth() != CV_32F)
        tmp2.convertTo(tmp2, CV_32F);

    return tmp2.at<float>(0);
}


bool LinearRegressionImpl::train( InputArray samples, int layout, InputArray responses, Stream& stream )
{
    CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1) && ( (responses.rows() == 1) || (responses.cols() == 1) ) );

    GpuMat tmp;

    GpuMat X;
    GpuMat y;

    if(!samples.isGpuMat())
        X.upload(samples, stream);
    else
        X = getInput<GpuMat>(samples);

    if(!responses.isGpuMat())
        y.upload(responses, stream);
    else
        y = getInput<GpuMat>(responses);

    GpuMat mu_X;
    Scalar mu_y;

    const int wdepth = std::max(X.depth(), CV_32F);

    int N(0);

    if(layout == ml::COL_SAMPLE)
        transpose(X, X);

    if(y.rows == 1)
        transpose(y, y);


    // First centre the data.

    reduce(X, mu_X, 0, REDUCE_AVG, wdepth, stream);
#if 0
    gemm(GpuMat(X.rows, 1, X.depth, Scalar::all(1.)), mu_X, -1., X, 1., X, 0, stream);
#else
    centreToTheMeanAxis0(X, mu_X, X, wdepth, stream);
#endif

    mu_y = mean(y);
    subtract(y, mu_y, y, noArray(), wdepth, stream);


    // Compute the weights.

    lstsq(X, y, tmp, noArray(), noArray(), stream);

    N = X.rows;

    // Update the weights if needed.

    if(this->flag == UPDATE_MODEL)
    {
        double alpha(1. / static_cast<double>(this->N)), beta(1. / static_cast<double>(N) );

        GpuMat tmp_w;

        this->weights.copyTo(tmp_w, stream);

        addWeighted(tmp_w, alpha, tmp, beta, 0., tmp, -1, stream);

        N+=this->N;
    }

    transpose(tmp, tmp, stream);

    tmp.copyTo(this->weights);
    this->N = N;

    this->flag = 0;

    // Compute the interception coefficient.

    this->intercept = this->get_intercept_(mu_X, mu_y, tmp, wdepth, stream);

    return true;
}

float LinearRegressionImpl::predict(InputArray samples, OutputArray results, int , Stream &stream) const
{
    CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1) && samples.isGpuMat() && results.isGpuMat());

    GpuMat X = samples.getGpuMat();

    GpuMat mu_X, y;

    const int wdepth = X.depth();

    // Centre the samples.

    reduce(X, mu_X, 0, REDUCE_AVG, wdepth, stream);

    // Compute the prediction.

    gemm(X, this->weights, 1., GpuMat(X.rows, this->weights.rows, wdepth, Scalar::all(this->intercept)), 1., y, GEMM_2_T, stream);

    if(results.needed())
        y.copyTo(results, stream);

    if(X.rows > 1)
        return 0.f;

    Mat tmp;

    y.download(tmp, stream);

    tmp.convertTo(tmp, CV_32F);

    return tmp.at<float>(0);
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
    if(fn.name() != this->getDefaultName() && fn.name() != "extra_ml_linear_regression")
        CV_Error(cv::Error::StsError, "The filename provided, was not generated for a linear regression algorithm.");

    this->clear();

    this->N = (int)fn["N"];
    this->intercept = (float)fn["intercept"];

    Mat tmp;

    fn["weights"] >> tmp;

    this->weights.upload(tmp);
}

void LinearRegressionImpl::write(FileStorage &fs) const
{
    fs << "N" << this->N;
    fs << "intercept" << this->intercept;
    Mat tmp;
    this->weights.download(tmp);
    fs << "weights" << tmp;
}

void LinearRegressionImpl::read(const String & _filename)
{
    if(!cv::utils::fs::exists(_filename))
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


} // cuda

} // cv
