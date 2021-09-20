#include "opencv2/cudaml.hpp"
#include "precomp.hpp"

using namespace std;

namespace cv
{

namespace cuda
{

namespace
{

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
    virtual bool isTrained() const CV_OVERRIDE { return !this->coefficients_.empty(); }
    virtual bool isClassifier() const CV_OVERRIDE { return false; }
    virtual String getDefaultName() const CV_OVERRIDE { return "extra_cuda_ml_patiral_least_square_regression"; }

    virtual bool train( InputArray samples, int layout, InputArray responses, Stream& stream ) CV_OVERRIDE;
    virtual float predict(InputArray samples, OutputArray results, int flags, Stream& stream) const CV_OVERRIDE;

    virtual void clear() CV_OVERRIDE;

    virtual void read( const FileNode& fn ) CV_OVERRIDE;
    virtual void write( FileStorage& fs ) const  CV_OVERRIDE;

    virtual void read( const String& filename ) CV_OVERRIDE;
    virtual void write( const String& filename) const  CV_OVERRIDE;

private:

    int nb_components;
    int N;

    GpuMat xloadings_;
    GpuMat xscores_;

    GpuMat yloadings_;
    GpuMat yscores_;

    GpuMat projections_;
    GpuMat fitted_;

    GpuMat coefficients_;

};

bool PLSRImpl::train( InputArray samples, int layout, InputArray responses, Stream& stream )
{
    CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1) && ( (responses.rows() == 1) || (responses.cols() == 1) ) );

    GpuMat X(samples.getGpuMat());
    GpuMat y(responses.getGpuMat());

    if(layout == ml::COL_SAMPLE)
    {
        GpuMat tmp;
        transpose(X, tmp, stream);
        X = tmp;

        tmp.release();

        transpose(y, tmp);
        y = tmp;
    }

    const int wdepth = std::max(X.depth(), CV_32F);

    const int ncomp = this->nb_components;
    const int nobs = X.rows;
    const int npred = X.cols;
    const int nresp = y.cols;

    GpuMat R(npred, ncomp, wdepth, Scalar::all(0.));
    GpuMat P(npred, ncomp, wdepth, Scalar::all(0.));
    GpuMat V(npred, ncomp, wdepth, Scalar::all(0.));
    GpuMat T(nobs, ncomp, wdepth, Scalar::all(0.));
    GpuMat U(nobs, ncomp, wdepth, Scalar::all(0.));
    GpuMat Q(nresp, ncomp, wdepth, Scalar::all(0.));

    GpuMat B, Qt, mu_X;

    GpuMat tmp;

    // Mean centering Data matrix
    reduce(X, mu_X, 0, REDUCE_AVG, wdepth, stream);
    gemm(GpuMat(X.rows, 1, wdepth, Scalar::all(1.)), mu_X, -1., X, 1., X, 0, stream);

    Mat _Q = Mat::zeros (nresp, ncomp, wdepth);

    // Mean centering responses
//    Scalar mu_y = mean(Mat(y)); // in OpenCV's cuda API there is no cv::cuda::mean, and cv::cuda::meanStdDev want CV_8UC1 source. Really ?
    Scalar mu_y;
#if 0
    reduce(y, tmp, 0, REDUCE_AVG, wdepth, stream);
    mu_y(0) = get_scalar(tmp);
#else
   mu_y = cuda::mean(y);
#endif
    subtract(y, mu_y, y, noArray(), wdepth, stream);

    GpuMat S;
    gemm(X, y, 1., noArray(), 0., S, GEMM_1_T, stream);



    for(int a=0; a<ncomp; a++)
    {

        GpuMat SS;
        gemm(S, S, 1., noArray(), 0., SS, GEMM_1_T, stream);
        // SS = S.t() * S

        // Y factor weights

        double _q = get_scalar(SS);

        GpuMat r, t, mu_t;

        // X block factor weights
        multiply(S, _q, r, 1., wdepth, stream);


        // X block factor scores
        gemm(X, r, 1., noArray(), 0., t, 0, stream);
        // t = X * r
#if 0
        reduce(t, tmp, 0, REDUCE_AVG, wdepth, stream);
        subtract(t, get_scalar(tmp), t, noArray(), wdepth, stream);
#else
        subtract(t, cuda::mean(tmp), t, noArray(), wdepth, stream);
#endif

        // t-=mu_t

        // compute norm
        GpuMat nt;
        gemm(t, t, 1., noArray(), 0., nt, GEMM_1_T, stream);
        // ny = t.t() * t

        double _nt = std::sqrt(get_scalar(nt));

        divide(t, _nt, t, 1., wdepth, stream);
        divide(r, _nt, r, 1., wdepth, stream);
        // t/= _nt
        // r/= _nt

        GpuMat p, u, v;

        // X block factor loadings
        gemm(X, t, 1., noArray(), 0., p, GEMM_1_T, stream);
        //            p = X.t() * t;

        // Y block factor loadings
        gemm(y, t, 1., noArray(), 0., tmp, GEMM_1_T, stream);
        _q = get_scalar(tmp);
        //            q = y.t() * t;

        // Y block factor scores
        multiply(y, _q, u, 1., wdepth, stream);
        //            u = y * q;

        v = p.clone();

        // Ensure orthogonality
        if(a)
        {
            gemm(V, p, 1., noArray(), 0., tmp, GEMM_1_T, stream);
            gemm(V, tmp, -1., v, 1., v, 0, stream);

            //                v -= V * (V.t() * p);

            gemm(T, u, 1., noArray(), 0., tmp, GEMM_1_T, stream);
            gemm(T, tmp, -1., u, 1., u, 0, stream);

            //                u -= T * (T.t() * u);
        }

        // Normalize the orthogonal loadings.
        GpuMat vv;
        gemm(v, v, 1., noArray(), 0., vv, GEMM_1_T, stream);
        //            vv = v.t() * v;

        double _vv = std::sqrt(get_scalar(vv));

        divide(v, _vv, v, 1., wdepth, stream);
        //            v /= _vv;

        // deflate S wrt loadings.

        gemm(v, S, 1., noArray(), 0., tmp, GEMM_1_T, stream);
        gemm(v, tmp, -1., S, 1., S, 0, stream);

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

    gemm(T, Qt, 1., noArray(), 0., this->fitted_, 0, stream);
    add(this->fitted_, mu_y, this->fitted_, noArray(), wdepth, stream);
    //      fitted += T * Qt + mu_y;

    // Regression Coefficients
    gemm(R, Qt, 1., noArray(), 0., B, 0, stream);
    //      B = R * Qt;

    this->coefficients_ = B;
    this->xscores_ = T;
    this->xloadings_ = P;
    this->yscores_ = U;
    this->yloadings_ = Q;
    this->projections_ = R;

    return true;
}


float PLSRImpl::predict(InputArray samples, OutputArray results, int , Stream& stream) const
{
        CV_Assert( (samples.depth() >= CV_32F) && (samples.channels() == 1));

        GpuMat X = samples.getGpuMat();
        GpuMat mu_X, y;


        const int wdepth = this->coefficients_.depth();

        reduce(X, mu_X, 0, REDUCE_AVG, wdepth, stream);
        gemm(GpuMat(X.rows, 1, X.depth(), Scalar::all(1.)), mu_X, -1., X, 1., X, 0, stream);

        gemm(X, this->coefficients_, 1., noArray(), 0., y, 0, stream);

        y.copyTo(results, stream);

        return static_cast<float>(get_scalar(y));
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
    if(fn.name() != this->getDefaultName() && fn.name() != "extra_ml_patiral_least_square_regression")
        CV_Error(cv::Error::StsError, "The filename provided, was not generated for a linear regression algorithm.");

    this->clear();

    this->N = (int)fn["N"];
    this->nb_components = (int)fn["dim_latent"];

    tuple<GpuMat*, String> tmp[7] = {make_tuple(&this->xscores_, "xscores"),
                                     make_tuple(&this->yscores_, "yscores"),
                                     make_tuple(&this->xloadings_, "xloadings"),
                                     make_tuple(&this->yloadings_, "yloadings"),
                                     make_tuple(&this->fitted_, "fitted"),
                                     make_tuple(&this->coefficients_, "coefficients"),
                                     make_tuple(&this->projections_, "projections"),
                                    };

    for(auto& obj : tmp)
    {
        String name;
        GpuMat* gtmp;
        Mat tmp;

        std::tie(gtmp, name) = obj;

        fn[name] >> tmp;

        gtmp->upload(tmp);
    }

//    fn["xscores"] >> this->xscores_;
//    fn["yscores"] >> this->yscores_;
//    fn["xloadings"] >> this->xloadings_;
//    fn["yloadings"] >> this->yloadings_;
//    fn["fitted"] >> this->fitted_;
//    fn["coefficients"] >> this->coefficients_;
//    fn["projections"] >> this->projections_;
}

void PLSRImpl::write(FileStorage &fs) const
{
    fs << "N" << this->N;
    fs << "dim_latent" << this->nb_components;
    fs << "xscores" << Mat(this->xscores_);
    fs << "yscores" << Mat(this->yscores_);
    fs << "xloadings" << Mat(this->xloadings_);
    fs << "yloadings" << Mat(this->yloadings_);
    fs << "fitted" << Mat(this->fitted_);
    fs << "coefficients" << Mat(this->coefficients_);
    fs << "projections" << Mat(this->projections_);
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





        std::vector<std::tuple<GpuMat*, String> > to_load = {std::make_tuple(&this->xscores_, "xscores"),
                                        std::make_tuple(&this->yscores_, "yscores"),
                                        std::make_tuple(&this->xloadings_, "xloadings"),
                                        std::make_tuple(&this->yloadings_, "yloadings"),
                                        std::make_tuple(&this->fitted_, "fitted"),
                                        std::make_tuple(&this->coefficients_, "coefficients"),
                                        std::make_tuple(&this->projections_, "projections")
                                       };

        CV_Assert(fid->hlexists("N") && fid->hlexists("dim_latent") );


        Mat tmp;

        fid->dsread(tmp, "N");
        this->N = tmp.at<float>(0);

        fid->dsread(tmp, "dim_latent");
        this->nb_components = tmp.at<float>(0);

        for(auto& obj : to_load)
        {
            String name;
            GpuMat* gtmp;

            std::tie(gtmp, name) = obj;

            CV_Assert(fid->hlexists(name));
            fid->dsread(tmp, name);

            gtmp->upload(tmp);
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

        std::vector<std::tuple<Mat, String> > to_save = { std::make_tuple(Mat(this->xscores_), "xscores"),
                                         std::make_tuple(Mat(this->yscores_), "yscores"),
                                         std::make_tuple(Mat(this->xloadings_), "xloadings"),
                                         std::make_tuple(Mat(this->yloadings_), "yloadings"),
                                         std::make_tuple(Mat(this->fitted_), "fitted"),
                                         std::make_tuple(Mat(this->coefficients_), "coefficients"),
                                         std::make_tuple(Mat(this->projections_), "projections") };

        for(auto& obj : to_save)
        {
            Mat ds;
            String name;

            std::tie(ds, name) = obj;

            fid->dswrite(ds, name);
        }
    }

    if(!file_type_found)
        CV_Error(cv::Error::StsError, "The file format provided is not supported yet! Only XML, YAML, JSON, and HDF5 file format are supported.");
}

} // anonymous


Ptr<PLSR> PLSR::create(const int &n_dimensions){ return makePtr<PLSRImpl>(n_dimensions);}

Ptr<PLSR> PLSR::load(const String &filename) { return makePtr<PLSRImpl>(filename);}

} // cuda

} // cv
