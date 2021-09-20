#ifndef MLX_HPP
#define MLX_HPP

#include "opencv2/core.hpp"
#include "opencv2/ml.hpp"

namespace cv
{

namespace mlx
{

/****************************************************************************************\
*                                   Linear Regression                                    *
\****************************************************************************************/

///
/// \brief The LinearRegression class
///
class LinearRegression : public ml::StatModel
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
    /// \param _src : input, cv::Mat or cv::UMat, replace the weight of the curent model, by those prodided.
    ///
    virtual void setCoef(InputArray _src) = 0;

    ///
    /// \brief getCoef : return the weiths of the model.
    /// \param _dst :
    ///
    virtual void getCoef(OutputArray _dst) const = 0;

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
    /// \brief read : load the data from a file, either xml, yaml, json or h5
    /// \param filename : name of the file to read
    ///
    virtual void read(const String& filename) = 0;

    ///
    /// \brief write : write the main parameters of the model to the specified filename.
    /// T              The file extension can be xml, yaml, json or h5.
    /// \param filename : name of the file to write.
    ///
    virtual void write(const String& filename) const = 0;
};

///
/// \brief The PLSR class : implementation of the SIMPLS algorithm.
///
/// For details, see : SIMPLS: an alternative approach to partial least squares regression, by De Jong and Sijmen, 1993.
///
/// @note [DJ93] SIMPLS: an alternative approach to partial least squares regression
/// Sijmen, De Jong. In Chemometrics and intelligent laboratory systems, Elsevier, Amsterdam, NL, March 1993.
///
class PLSR : public ml::StatModel
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
    ///
    CV_WRAP virtual void getXScores(OutputArray _dst) const = 0;

    ///
    /// \brief setXScores : set the internal XScores parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal XScores parameter with.
    ///
    CV_WRAP virtual void setXScores(InputArray _src) = 0;

    ///
    /// \brief getYScores : return the YScores parameter
    /// \param _dst : copy of the internal YScore parameter.
    ///
    CV_WRAP virtual void getYScores(OutputArray _dst) const = 0;

    ///
    /// \brief setYScores : set the internal YScores parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal YScores parameter with.
    ///
    CV_WRAP virtual void setYScores(InputArray _src) = 0;

    ///
    /// \brief getXLoadings : return the XLoadings parameter
    /// \param _dst : copy of the internal XLoadings parameter.
    ///
    CV_WRAP virtual void getXLoadings(OutputArray _dst) const = 0;

    ///
    /// \brief setXLoadings : set the internal XLoadings parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal XLoadings parameter with.
    ///
    CV_WRAP virtual void setXLoadings(InputArray _src) = 0;

    ///
    /// \brief getYLoadings : return the YLoadings parameter
    /// \param _dst : copy of the internal YLoadings parameter.
    ///
    CV_WRAP virtual void getYLoadings(OutputArray _dst) const = 0;

    ///
    /// \brief setYLoadings : set the internal YLoadings parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal YLoadings parameter with.
    ///
    CV_WRAP virtual void setYLoadings(InputArray _src) = 0;

    ///
    /// \brief getProjections : return the Projections parameter
    /// \param _dst : copy of the internal Projections parameter.
    ///
    CV_WRAP virtual void getProjections(OutputArray _dst) const = 0;

    ///
    /// \brief setProjections : set the internal Projections parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal Projections parameter with.
    ///
    CV_WRAP virtual void setProjections(InputArray _src) = 0;

    ///
    /// \brief getCoefficients : return the Coefficients parameter
    /// \param _dst : copy of the internal Coefficients parameter.
    ///
    CV_WRAP virtual void getCoefficients(OutputArray _dst) const = 0;

    ///
    /// \brief setCoefficients : set the internal Coefficients parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal Coefficients parameter with.
    ///
    CV_WRAP virtual void setCoefficients(InputArray _src) = 0;

    ///
    /// \brief getFitted : return the Fitted parameter
    /// \param _dst : copy of the internal Fitted parameter.
    ///
    CV_WRAP virtual void getFitted(OutputArray _dst) const = 0;

    ///
    /// \brief setFitted : set the internal Fitted parameter to the argument _src.
    /// \param _src : Matrix to set or replace, the internal Fitted parameter with.
    ///
    CV_WRAP virtual void setFitted(InputArray _src) = 0;

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
    /// \brief read : load the data from a file, either xml, yaml, json or h5
    /// \param filename : name of the file to read
    ///
    CV_WRAP virtual void read(const String& filename) = 0;

    ///
    /// \brief write : write the main parameters of the model to the specified filename.
    /// T              The file extension can be xml, yaml, json or h5.
    /// \param filename : name of the file to write.
    ///
    CV_WRAP virtual void write(const String& filename) const = 0;

};

} //mlx

} // cv

#endif // MLX_HPP
