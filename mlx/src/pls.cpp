#include "opencv2/mlx.hpp"
#include "precomp.hpp"


using namespace cv::ml;
using namespace std;


namespace cv
{

namespace mlx
{

namespace
{

#define IMPL_VOID_ACCESSOR(name, variable) \
    virtual void get##name(OutputArray _dst) const CV_OVERRIDE { _dst.assign(variable);} \
    virtual void set##name(InputArray _dst) CV_OVERRIDE { variable = _dst.getMat();}

class PLSRImpl CV_FINAL : public PLSR
{
public:

    inline PLSRImpl(const int& _nb_components):
        nb_components(_nb_components),
        N(0)
    {}

    inline PLSRImpl(const String &filename){ this->read(filename);}

    virtual ~PLSRImpl() = default;

    IMPL_VOID_ACCESSOR(XScores, xscores_)
    IMPL_VOID_ACCESSOR(YScores, yscores_)

    IMPL_VOID_ACCESSOR(XLoadings, xloadings_)
    IMPL_VOID_ACCESSOR(YLoadings, yloadings_)

    IMPL_VOID_ACCESSOR(Projections, projections_)

    IMPL_VOID_ACCESSOR(Coefficients, coefficients_)

    IMPL_VOID_ACCESSOR(Fitted, fitted_)

    virtual int getLatenetSpaceDimension() const CV_OVERRIDE { return this->nb_components; }
    virtual void setLatenetSpaceDimension(const int& n_dimensions) CV_OVERRIDE { this->nb_components = n_dimensions;}

    virtual int getVarCount() const CV_OVERRIDE { return this->N; }
    virtual bool isTrained() const CV_OVERRIDE { return !this->fitted_.empty(); }
    virtual bool isClassifier() const CV_OVERRIDE { return false; }
    virtual String getDefaultName() const CV_OVERRIDE { return "extra_ml_patiral_least_square_regression"; }

    virtual bool train( const Ptr<TrainData>& trainData, int ) CV_OVERRIDE;
    virtual bool train( InputArray samples, int layout, InputArray responses ) CV_OVERRIDE;
    virtual float predict(InputArray samples, OutputArray results, int flags) const CV_OVERRIDE;

    virtual void clear() CV_OVERRIDE;

    virtual void read( const FileNode& fn ) CV_OVERRIDE;
    virtual void write( FileStorage& fs ) const  CV_OVERRIDE;

    virtual void read( const String& filename ) CV_OVERRIDE;
    virtual void write( const String& filename) const  CV_OVERRIDE;

private:

    int nb_components;
    int N;

    Mat xloadings_;
    Mat xscores_;

    Mat yloadings_;
    Mat yscores_;

    Mat projections_;
    Mat fitted_;

    Mat coefficients_;

    template<class T>
    bool train_(InputArray& _X, int layout, InputArray& _y);

    template<class T>
    float predict_(InputArray& _X, OutputArray& _results, int) const;
};

template<class L, class R>
struct copy_or_assign_t
{
    // Note that, in the case where L is a UMat and R is a Mat making a copy provide a more stable result.
    static void op(const L& left, R& right){ left.copyTo(right);}
};

template<class T>
struct copy_or_assign_t<T,T>
{
    static void op(const T& left, T& right){ right = left;}
};

template<class T>
inline double get_scalar(const T& s){Mat tmp; s.convertTo(tmp, CV_64F); return tmp.at<double>(0);}



template<class I, class O>
void plsfit(const I& X, const I& y, const int& ncomp, O& coefficients, O& XSCORES, O& XLOADINGS, O& YSCORES, O& YLOADINGS, O& projection, O& fitted)
{
//    const int ncomp = 3;

    const int wdepth = std::max(X.depth(), CV_32F);

    const int nobs = X.rows;
    const int npred = X.cols;
    const int nresp = y.cols;

    I R = I::zeros (npred, ncomp, wdepth);
    I P = I::zeros (npred, ncomp, wdepth);
    I V = I::zeros (npred, ncomp, wdepth);;
    I T = I::zeros (nobs, ncomp, wdepth);
    I U = I::zeros (nobs, ncomp, wdepth);
    I Q = I::zeros (nresp, ncomp, wdepth);

    I B, Qt, mu_X;

    // Mean centering Data matrix
    reduce(X, mu_X, 0, REDUCE_AVG, wdepth);
    gemm(Mat::ones(X.rows, 1, wdepth), mu_X, -1., X, 1., X);

    if(nresp>1)
    {
        // Mean centering responses
        I mu_y;
        reduce(y, mu_y, 0, REDUCE_AVG, wdepth);
        gemm(Mat::ones(y.rows, 1, wdepth), mu_y, -1., y, 1., y);
        // y -= mu_y

        I S;
        gemm(X, y, 1., noArray(), 0., S, GEMM_1_T);
        // S = X.t() * y

        for(int a=0; a<ncomp; a++)
        {

            I SS;
            gemm(S, S, 1., noArray(), 0., SS, GEMM_1_T);
            // SS = S.t() * S

            // Y factor weights
            I eigval, eigvec;

            eigen(SS, eigval, eigvec);

            // get dominant eigenvector
            Point max_eigval_;

            minMaxLoc(eigval, nullptr, nullptr, nullptr, &max_eigval_);

            int max_eigval = max_eigval_.x;

            I q = max_eigval == 0 ? eigvec.col(max_eigval) : eigvec.colRange(0, max_eigval);

            I r, t, mu_t;

            // X block factor weights
            gemm(S, q, 1., noArray(), 0., r);
            // r = S * q

            // X block factor scores
            gemm(X, r, 1., noArray(), 0., t);
            // t = X * r

            reduce(t, mu_t, 0, REDUCE_AVG, wdepth);
            gemm(I::ones(t.rows, 1, wdepth), mu_t, -1., t, 1., t);
            // t-=mu_t

            // compute norm
            Mat nt;
            gemm(t, t, 1., noArray(), 0., nt, GEMM_1_T);
            // ny = t.t() * t

            double _nt = std::sqrt(get_scalar(nt));

            divide(t, _nt, t, 1., wdepth);
            divide(r, _nt, r, 1., wdepth);
            // t/= sqrt(_nt)
            // r/= sqrt(_nt)

            Mat p, u, v;

            // X block factor loadings
            gemm(X, t, 1., noArray(), 0., p, GEMM_1_T);
//            p = X.t() * t;

            // Y block factor loadings
            gemm(y, t, 1., noArray(), 0., q, GEMM_1_T);
//            q = y.t() * t;

            // Y block factor scores
            gemm(y, q, 1., noArray(), 0., u);
//            u = y * q;

            v = p.clone();

            // Ensure orthogonality
            if(a)
            {
                Mat tmp;

                gemm(V, p, 1., noArray(), 0., tmp, GEMM_1_T);
                gemm(V, tmp, -1., v, 1., v);

//                v -= V * (V.t() * p);

                gemm(T, u, 1., noArray(), 0., tmp, GEMM_1_T);
                gemm(T, tmp, -1., u, 1., u);

//                u -= T * (T.t() * u);


            }

            // Normalize the orthogonal loadings.
            Mat vv;
            gemm(v, v, 1., noArray(), 0., vv, GEMM_1_T);
//            vv = v.t() * v;

            double _vv = std::sqrt(get_scalar(vv));

            divide(v, _vv, v, 1., wdepth);
//            v /= _vv;

            // deflate S wrt loadings.
            Mat tmp;
            gemm(v, S, 1., noArray(), 0., tmp, GEMM_1_T);
            gemm(v, tmp, -1., S, 1., S);

//            S -= v * (v.t() * S);


            r.copyTo(R.col(a));
            t.copyTo(T.col(a));
            p.copyTo(P.col(a));
            q.copyTo(Q.col(a));
            u.copyTo(U.col(a));
            v.copyTo(V.col(a));
        }


        transpose(Q, Qt);

        gemm(Mat::ones(T.rows, 1, wdepth), mu_y, 1., noArray(), 0., fitted);
        gemm(T, Qt, 1., fitted, 1., fitted);
//        fitted += T * Qt;
    }
    else
    {
        Mat _Q = Mat::zeros (nresp, ncomp, wdepth);

        // Mean centering responses
        Scalar mu_y = mean(y);
        subtract(y, mu_y, y, noArray(), wdepth);

        I S;
        gemm(X, y, 1., noArray(), 0., S, GEMM_1_T);

        I tmp;

        for(int a=0; a<ncomp; a++)
        {

            I SS;
            gemm(S, S, 1., noArray(), 0., SS, GEMM_1_T);
            // SS = S.t() * S

            // Y factor weights

            double _q = get_scalar(SS);

            I r, t, mu_t;

            // X block factor weights
            multiply(S, _q, r, 1., wdepth);


            // X block factor scores
            gemm(X, r, 1., noArray(), 0., t);
            // t = X * r

            subtract(t, mean(t), t, noArray(), wdepth);
            // t-=mu_t

            // compute norm
            I nt;
            gemm(t, t, 1., noArray(), 0., nt, GEMM_1_T);
            // ny = t.t() * t

            double _nt = std::sqrt(get_scalar(nt));

            divide(t, _nt, t, 1., wdepth);
            divide(r, _nt, r, 1., wdepth);
            // t/= _nt
            // r/= _nt

            I p, u, v;

            // X block factor loadings
            gemm(X, t, 1., noArray(), 0., p, GEMM_1_T);
//            p = X.t() * t;

            // Y block factor loadings
            gemm(y, t, 1., noArray(), 0., tmp, GEMM_1_T);
            _q = get_scalar(tmp);
//            q = y.t() * t;

            // Y block factor scores
            multiply(y, _q, u, 1., wdepth);
//            u = y * q;

            v = p.clone();

            // Ensure orthogonality
            if(a)
            {
                gemm(V, p, 1., noArray(), 0., tmp, GEMM_1_T);
                gemm(V, tmp, -1., v, 1., v);

//                v -= V * (V.t() * p);

                gemm(T, u, 1., noArray(), 0., tmp, GEMM_1_T);
                gemm(T, tmp, -1., u, 1., u);

//                u -= T * (T.t() * u);
            }

            // Normalize the orthogonal loadings.
            I vv;
            gemm(v, v, 1., noArray(), 0., vv, GEMM_1_T);
//            vv = v.t() * v;

            double _vv = std::sqrt(get_scalar(vv));

            divide(v, _vv, v, 1., wdepth);
//            v /= _vv;

            // deflate S wrt loadings.

            gemm(v, S, 1., noArray(), 0., tmp, GEMM_1_T);
            gemm(v, tmp, -1., S, 1., S);

//            S -= v * (v.t() * S);


            r.copyTo(R.col(a));
            t.copyTo(T.col(a));
            p.copyTo(P.col(a));
            wdepth == CV_32F ? _Q.at<float>(a) = _q : _Q.at<double>(a) = _q;
            u.copyTo(U.col(a));
            v.copyTo(V.col(a));
        }

        _Q.copyTo(Q);

        transpose(Q, Qt);

        gemm(T, Qt, 1., noArray(), 0., fitted);
        add(fitted, mu_y, fitted, noArray(), wdepth);
//        fitted += T * Qt + mu_y;
    }


    // Regression Coefficients
    gemm(R, Qt, 1., noArray(), 0., B);
//    B = R * Qt;


    copy_or_assign_t<I,O>::op(B, coefficients);
    copy_or_assign_t<I,O>::op(T, XSCORES);
    copy_or_assign_t<I,O>::op(P, XLOADINGS);
    copy_or_assign_t<I,O>::op(U, YSCORES);
    copy_or_assign_t<I,O>::op(Q, YLOADINGS);
    copy_or_assign_t<I,O>::op(R, projection);
}







template<class Tp>
bool PLSRImpl::train_(InputArray& _X, int layout, InputArray& _y)
{
    typedef  Tp MatType;

    MatType X(getInput<MatType>(_X)), y(getInput<MatType>(_y));

    if(layout == COL_SAMPLE)
    {
        transpose(X, X);
        if(y.rows == 1)
            transpose(y, y);
    }



    if(y.rows == X.cols)
        transpose(y,y);

    this->N = y.rows;

    plsfit<Tp, Mat>(X, y,
                    this->nb_components, this->coefficients_,
                    this->xscores_, this->xloadings_,
                    this->yscores_, this->yloadings_,
                    this->projections_, this->fitted_);


    return true;
}


bool PLSRImpl::train( const Ptr<TrainData>& trainData, int )
{
    CV_Assert(trainData);

    return this->train(trainData->getTrainSamples(), trainData->getLayout(), trainData->getTrainResponses());
}

bool PLSRImpl::train( InputArray samples, int layout, InputArray responses )
{
    CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1) && ( ( (layout == ml::ROW_SAMPLE) && (responses.rows() == samples.rows() ) ) || ( (layout == ml::COL_SAMPLE) && (responses.cols() == samples.cols()) ) ) );

    return samples.isUMat() || responses.isUMat() ? this->train_<UMat>(samples, layout, responses) : this->train_<Mat>(samples, layout, responses);
}


template<class T>
float PLSRImpl::predict_(InputArray& _samples, OutputArray& _results, int) const
{
    T X(getInput<T>(_samples)), mu_X;
    T y;

    const int wdepth = this->coefficients_.depth();

    reduce(X, mu_X, 0, REDUCE_AVG, wdepth);
    centreToTheMeanAxis0(X, mu_X, X);

    gemm(X, this->coefficients_, 1., noArray(), 0., y);

    _results.assign(y);

    return static_cast<float>(get_scalar(y));
}


float PLSRImpl::predict(InputArray samples, OutputArray results, int flags) const
{
    CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1));

    return samples.isUMat() || (results.needed() && results.isUMat()) ? this->predict_<UMat>(samples, results, flags) : this->predict_<Mat>(samples, results, flags);
}


void PLSRImpl::clear()
{
    this->xscores_.release();
    this->yscores_.release();
    this->xloadings_.release();
    this->yloadings_.release();
    this->fitted_.release();
    this->coefficients_.release();
    this->projections_.release();
    this->nb_components = 0;
    this->N = 0;
}

void PLSRImpl::read(const FileNode &fn)
{
    // Compatibility between CUDA and non-CUDA versions.
    if(fn.name() != this->getDefaultName() && fn.name() != "extra_cuda_ml_patiral_least_square_regression")
        CV_Error(cv::Error::StsError, "The filename provided, was not generated for a linear regression algorithm.");

    this->clear();

    this->N = (int)fn["N"];
    this->nb_components = (int)fn["dim_latent"];



    fn["xscores"] >> this->xscores_;
    fn["yscores"] >> this->yscores_;
    fn["xloadings"] >> this->xloadings_;
    fn["yloadings"] >> this->yloadings_;
    fn["fitted"] >> this->fitted_;
    fn["coefficients"] >> this->coefficients_;
    fn["projections"] >> this->projections_;
}

void PLSRImpl::write(FileStorage &fs) const
{
    fs << "N" << this->N;
    fs << "dim_latent" << this->nb_components;
    fs << "xscores" << this->xscores_;
    fs << "yscores" << this->yscores_;
    fs << "xloadings" << this->xloadings_;
    fs << "yloadings" << this->yloadings_;
    fs << "fitted" << this->fitted_;
    fs << "coefficients" << this->coefficients_;
    fs << "projections" << this->projections_;
}




void PLSRImpl::read(const String & _filename)
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





        std::vector<read_h5_ds_t> to_load = {{&this->xscores_, "xscores"},
                                        {&this->yscores_, "yscores"},
                                        {&this->xloadings_, "xloadings"},
                                        {&this->yloadings_, "yloadings"},
                                        {&this->fitted_, "fitted"},
                                        {&this->coefficients_, "coefficients"},
                                        {&this->projections_, "projections"}
                                       };

        CV_Assert(fid->hlexists("N") && fid->hlexists("dim_latent") );


        Mat tmp;

        fid->dsread(tmp, "N");
        this->N = tmp.at<float>(0);

        fid->dsread(tmp, "dim_latent");
        this->nb_components = tmp.at<float>(0);

        for(read_h5_ds_t& obj : to_load)
        {
            CV_Assert(fid->hlexists(obj.ds_name));
            fid->dsread(*obj.ds, obj.ds_name);
        }
    }

    if(!file_type_found)
        CV_Error(cv::Error::StsError, "The file format provided is not supported yet! Only XML, YAML, JSON, and HDF5 file format are supported.");
}

void PLSRImpl::write(const String& _filename) const
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

        Mat1i tmp(1,1);

        tmp(0) = this->N;
        fid->dswrite(tmp.clone(), "N");

        tmp(0) = this->nb_components;
        fid->dswrite(tmp.clone(), "dim_latent");

        std::vector<write_h5_ds_t> to_save = { {&this->xscores_, "xscores"},
                                         {&this->yscores_, "yscores"},
                                         {&this->xloadings_, "xloadings"},
                                         {&this->yloadings_, "yloadings"},
                                         {&this->fitted_, "fitted"},
                                         {&this->coefficients_, "coefficients"},
                                         {&this->projections_, "projections"} };

        for(write_h5_ds_t& obj : to_save)
            fid->dswrite(*obj.ds, obj.ds_name);
    }

    if(!file_type_found)
        CV_Error(cv::Error::StsError, "The file format provided is not supported yet! Only XML, YAML, JSON, and HDF5 file format are supported.");
}

} // annymous

Ptr<PLSR> PLSR::create(const int &n_dimensions){ return makePtr<PLSRImpl>(n_dimensions);}

Ptr<PLSR> PLSR::load(const String &filename) { return makePtr<PLSRImpl>(filename);}

} // mlx

} // cv
