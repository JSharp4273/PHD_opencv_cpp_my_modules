#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"


namespace cv
{

namespace cuda
{
/** @brief Base class for statistical models in OpenCV ML.
 */
class CV_EXPORTS_W StatModelAsync : public Algorithm
{
public:

    virtual ~StatModelAsync() = default;

    /** Predict options */
    enum Flags {
        UPDATE_MODEL = 1,
        RAW_OUTPUT=1, //!< makes the method return the raw results (the sum), not the class label
        COMPRESSED_INPUT=2,
        PREPROCESSED_INPUT=4
    };

    /** @brief Returns the number of variables in training samples */
    CV_WRAP virtual int getVarCount() const = 0;

//    CV_WRAP virtual bool empty() const CV_OVERRIDE { return !this->isTrained(); }
    CV_WRAP virtual bool empty() const CV_OVERRIDE;

    /** @brief Returns true if the model is trained */
    CV_WRAP virtual bool isTrained() const = 0;
    /** @brief Returns true if the model is classifier */
    CV_WRAP virtual bool isClassifier() const = 0;


    /** @brief Trains the statistical model

    @param samples training samples
    @param layout See ml::SampleTypes.
    @param responses vector of responses associated with the training samples.
    */
    CV_WRAP virtual bool train( InputArray samples, int layout, InputArray responses, Stream& stream = Stream::Null() );


    /** @brief Predicts response(s) for the provided sample(s)

    @param samples The input samples, floating-point matrix
    @param results The optional output matrix of results.
    @param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
     */
    CV_WRAP virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0, Stream& stream = Stream::Null() ) const = 0;

};

///
/// \brief The LinearRegression class : implementation of the a multi-dimensional linear regression.
///
/// This code is inspired by the work of [MC89]
///
/// @note [MC89] Generalized Linear Models, Second Edition. Boca Raton: Chapman and Hall/CRC. ISBN 0-412-31760-5.
///
///
class LinearRegression : public StatModelAsync
{
public:

    /** @brief Creates the empty model

    The static method creates empty decision tree with the specified parameters. It should be then
    trained using train method (see StatModel::train). Alternatively, you can load the model from
    file using Algorithm::load\<LinearRegression\>(filename).
     */
    CV_WRAP static Ptr<LinearRegression> create();

    /** @brief Loads and creates a serialized svm from a file
     *
     * Use LinearRegression::save to serialize and store an SVM to disk.
     * Load the LinearRegression from this file again, by calling this function with the path to the file.
     *
     * @param filepath path to serialized svm
     */
    CV_WRAP static Ptr<LinearRegression> load(const String& filepath);

    virtual ~LinearRegression() = default;

    ///
    /// \brief setCoef : set the weights of the model.
    /// \param _src : input, replace the weight of the curent model, by those prodided.
    /// \param stream : Stream of the asynchronous version.
    ///
    virtual void setCoef(InputArray _src, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getCoef : return the weiths of the model.
    /// \param _dst : variable to set with the weights of the current model.
    /// \param stream : Stream of the asynchronous version.
    ///
    virtual void getCoef(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief getIntercept
    /// \return return an offset parameter.
    ///
    virtual float getIntercept() const = 0;

    ///
    /// \brief setIntercept
    /// \param set the offset parameter.
    ///
    virtual void setIntercept(const float& _src) = 0;

    ///
    /// \brief read : set the internal settings from a file. The file can be a .h5, .xml, .yml or .json.
    /// \param filename : name of the file to read.
    ///
    CV_WRAP virtual void read(const String& filename) = 0;

    ///
    /// \brief write : write the internal settings to a file. The file can be a .h5, .xml, .yml or .json
    /// \param filename : name of the file to create.
    ///
    CV_WRAP virtual void write(const String& filename) const = 0;
};

///
/// \brief The PLSR class : implementation of the SIMPLS algorithm.
///
/// For details, see : SIMPLS: an alternative approach to partial least squares regression, by De Jong and Sijmen, 1993.
///
/// @note [DJ93] SIMPLS: an alternative approach to partial least squares regression
/// Sijmen, De Jong. In Chemometrics and intelligent laboratory systems, Elsevier, Amsterdam, NL, March 1993.
///
class PLSR : public StatModelAsync
{

public:

    /** @brief Creates the empty model

    The static method creates empty decision tree with the specified parameters. It should be then
    trained using train method (see StatModel::train). Alternatively, you can load the model from
    file using Algorithm::load\<LinearRegression\>(filename).
     */
    CV_WRAP static Ptr<PLSR> create(const int& n_dimensions);


    /** @brief Loads and creates a serialized svm from a file
     *
     * Use LinearRegression::save to serialize and store an SVM to disk.
     * Load the LinearRegression from this file again, by calling this function with the path to the file.
     *
     * @param filepath path to serialized svm
     */
    CV_WRAP static Ptr<PLSR> load(const String& filepath);

    virtual ~PLSR() = default;

    ///
    /// \brief getXScores : return the XScores parameter
    /// \param _dst : copy of the internal XScore parameter.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void getXScores(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief setXScores : set the internal XScores parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal XScores parameter with.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void setXScores(InputArray _src, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getYScores : return the YScores parameter
    /// \param _dst : copy of the internal YScore parameter.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void getYScores(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief setYScores : set the internal YScores parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal YScores parameter with.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void setYScores(InputArray _src, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getXLoadings : return the XLoadings parameter
    /// \param _dst : copy of the internal XLoadings parameter.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void getXLoadings(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief setXLoadings : set the internal XLoadings parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal XLoadings parameter with.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void setXLoadings(InputArray _src, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getYLoadings : return the YLoadings parameter
    /// \param _dst : copy of the internal YLoadings parameter.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void getYLoadings(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief setYLoadings : set the internal YLoadings parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal YLoadings parameter with.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void setYLoadings(InputArray _src, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getProjections : return the Projections parameter
    /// \param _dst : copy of the internal Projections parameter.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void getProjections(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief setProjections : set the internal Projections parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal Projections parameter with.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void setProjections(InputArray _dst, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getCoefficients : return the Coefficients parameter
    /// \param _dst : copy of the internal Coefficients parameter.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void getCoefficients(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief setCoefficients : set the internal Coefficients parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal Coefficients parameter with.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void setCoefficients(InputArray _dst, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getFitted : return the Fitted parameter
    /// \param _dst : copy of the internal Fitted parameter.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void getFitted(OutputArray _dst, Stream& stream = Stream::Null()) const = 0;

    ///
    /// \brief setFitted : set the internal Fitted parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal Fitted parameter with.
    /// \param stream : Stream of the asynchronous version.
    ///
    CV_WRAP virtual void setFitted(InputArray _dst, Stream& stream = Stream::Null()) = 0;

    ///
    /// \brief getLatenetSpaceDimension : return the number of latent dimensions.
    /// \return the number of latent dimensions.
    ///
    CV_WRAP virtual int getLatenetSpaceDimension() const = 0;

    ///
    /// \brief setLatenetSpaceDimension : set the internal number of latent dimensions.
    /// \param the number of latent dimensions.
    ///
    CV_WRAP virtual void setLatenetSpaceDimension(const int& n_dimensions) = 0;

    ///
    /// \brief read : set the internal settings from a file. The file can be a .h5, .xml, .yml or .json.
    /// \param filename : name of the file to read.
    ///
    CV_WRAP virtual void read(const String& filename) = 0;

    ///
    /// \brief write : write the internal settings to a file. The file can be a .h5, .xml, .yml or .json
    /// \param filename : name of the file to create.
    ///
    CV_WRAP virtual void write(const String& filename) const = 0;

};

} // cuda

} // cv
