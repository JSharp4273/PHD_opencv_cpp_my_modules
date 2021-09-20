#include "opencv2/xcore/template/arguments_io.hpp"

#include "opencv2/cvconfig.h"
#ifdef HAVE_CUDA
#include "opencv2/core/cuda.hpp"
#endif
namespace cv
{

////////////////////////////////////////
/// Input Output function            ///
////////////////////////////////////////


namespace
{

template<class T>
bool isMatrixVector_(T& _src)
{
    return _src.isMatVector() || _src.isUMatVector()
#ifdef HAVE_CUDA
            || _src.isGpuMatVector()
#endif
            ;
}

template <class T>
bool isScalar_(T& _src)
{
//    return !isMatrixVector_(_src) && (_src.total() * _src.channels() == 4);
    return _src.isMatx() && (_src.total() <= 4);
}

template<class T>
bool isMatrix_(T& _src)
{
    return !isMatrixVector_(_src) && !isScalar_(_src);
}

} // anonymous


bool isMatrix(InputArray& _src){ return isMatrix_<InputArray>(_src);}
bool isMatrix(InputOutputArray& _src){ return isMatrix_<InputOutputArray>(_src);}
bool isMatrix(OutputArray& _src){ return isMatrix_<OutputArray>(_src);}

bool isVectorOfMatrix(InputArray& _src){ return isMatrixVector_<InputArray>(_src); }
bool isVectorOfMatrix(InputOutputArray& _src){ return isMatrixVector_<InputOutputArray>(_src); }
bool isVectorOfMatrix(OutputArray& _src){ return isMatrixVector_<OutputArray>(_src); }

bool isScalar(InputArray& _src){ return isScalar_<InputArray>(_src); }
bool isScalar(InputOutputArray& _src){ return isScalar_<InputOutputArray>(_src); }
bool isScalar(OutputArray& _src){ return isScalar_<OutputArray>(_src); }


namespace
{

template<class A, class B, bool cpy>
struct get_input_helper
{
    typedef A output_type;
    typedef B input_type;

    static output_type get_arg(input_type){return output_type();}
    static void get_arg2(input_type, output_type& ){}
};

template<class B>
struct get_input_helper<Mat, B, false>
{
    typedef Mat output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret = _src.getMat();
#ifdef HAVE_CUDA
        if(_src.isGpuMat())
            ret = Mat(_src.getGpuMat());
#endif
        if(_src.isMatVector() || _src.isUMatVector())
            ret = _src.getMat(0);

#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;
            _src.getGpuMatVector(tmp);

            ret = Mat(tmp.front());
        }
#endif

        _dst = ret;
    }
};

template<class B>
struct get_input_helper<Mat, B, true>
{
    typedef Mat output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;
        bool copy_needed(false);

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
        {
            ret = _src.getMat();
            copy_needed = _src.isMat();
        }
#ifdef HAVE_CUDA
        if(_src.isGpuMat())
            ret = Mat(_src.getGpuMat());
#endif
        if(_src.isMatVector() || _src.isUMatVector())
        {
            ret = _src.getMat(0);
            copy_needed = _src.isMatVector();
        }

#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;
            _src.getGpuMatVector(tmp);

            ret = Mat(tmp.front());
        }
#endif
        if(copy_needed)
            ret.copyTo(_dst);
        else
            _dst = ret;
    }
};

template<class B>
struct get_input_helper<UMat, B, false>
{
    typedef UMat output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret = _src.getUMat();

#ifdef HAVE_CUDA
        if(_src.isGpuMat())
            Mat(_src.getGpuMat()).copyTo(ret);
#endif
        if(_src.isMatVector() || _src.isUMatVector())
            ret = _src.getUMat(0);

#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;

            _src.getGpuMatVector(tmp);

            Mat(tmp.front()).copyTo(ret);
        }
#endif

        _dst = ret;
    }
};

template<class B>
struct get_input_helper<UMat, B, true>
{
    typedef UMat output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;
        bool copy_needed(false);

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
        {
            ret = _src.getUMat();
            copy_needed = _src.isUMat();
        }

#ifdef HAVE_CUDA
        if(_src.isGpuMat())
            Mat(_src.getGpuMat()).copyTo(ret);
#endif
        if(_src.isMatVector() || _src.isUMatVector())
        {
            ret = _src.getUMat(0);
            copy_needed = _src.isUMatVector();
        }

#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;

            _src.getGpuMatVector(tmp);

            Mat(tmp.front()).copyTo(ret);
        }
#endif
        if(copy_needed)
            ret.copyTo(_dst);
        else
            _dst = ret;
    }
};

#ifdef HAVE_CUDA

template<class B>
struct get_input_helper<cuda::GpuMat, B, false>
{
    typedef cuda::GpuMat output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret = cuda::GpuMat(_src.getMat());

        if(_src.isGpuMat())
            ret = _src.getGpuMat();


        if(_src.isMatVector() || _src.isUMatVector())
            ret = cuda::GpuMat(_src.getMat(0));

        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;

            ret = tmp.front();
        }

        _dst = ret;
    }
};


template<class B>
struct get_input_helper<cuda::GpuMat, B, true>
{
    typedef cuda::GpuMat output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;

        bool copy_needed(false);

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret = cuda::GpuMat(_src.getMat());

        if(_src.isGpuMat())
        {
            ret = _src.getGpuMat();
            copy_needed = true;
        }


        if(_src.isMatVector() || _src.isUMatVector())
            ret = cuda::GpuMat(_src.getMat(0));

        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;

            ret = tmp.front();

            copy_needed = true;
        }

        if(copy_needed)
            ret.copyTo(_dst);
        else
            _dst = ret;
    }
};



#endif

template<class B>
struct get_input_helper<std::vector<Mat>, B, false>
{
    typedef std::vector<Mat> output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& ret)
    {
        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret.push_back(_src.getMat());
#ifdef HAVE_CUDA
        if(_src.isGpuMat())
            ret.push_back(Mat(_src.getGpuMat()));
#endif
        if(_src.isMatVector() || _src.isUMatVector())
            _src.getMatVector(ret);

#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;
            _src.getGpuMatVector(tmp);

            ret.resize(tmp.size());
            ret.shrink_to_fit();

            std::transform(tmp.begin(), tmp.end(), ret.begin(), [](const cuda::GpuMat& g)->Mat{return Mat(g);});
        }
#endif
        ret.shrink_to_fit();
    }
};

template<class B>
struct get_input_helper<std::vector<Mat>, B, true>
{
    typedef std::vector<Mat> output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;

        bool copy_needed(false);

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
        {
            ret.push_back(_src.getMat());
            copy_needed = _src.isMat();
        }
#ifdef HAVE_CUDA
        if(_src.isGpuMat())
            ret.push_back(Mat(_src.getGpuMat()));
#endif
        if(_src.isMatVector() || _src.isUMatVector())
        {
            _src.getMatVector(ret);
            copy_needed = _src.isMatVector();
        }

#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;
            _src.getGpuMatVector(tmp);

            ret.resize(tmp.size());
            ret.shrink_to_fit();

            std::transform(tmp.begin(), tmp.end(), ret.begin(), [](const cuda::GpuMat& g)->Mat{return Mat(g);});
        }
#endif
        ret.shrink_to_fit();


        if(copy_needed)
        {
            _dst.resize(ret.size());
            _dst.shrink_to_fit();

            transform(ret.begin(), ret.end(), _dst.begin(),[](const Mat& m)->Mat{return m.clone();});
        }
        else
        {
            _dst = std::move(ret);
        }
    }
};

template<class B>
struct get_input_helper<std::vector<UMat>, B, false>
{
    typedef std::vector<UMat> output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& ret)
    {
        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret.push_back(_src.getUMat());
#ifdef HAVE_CUDA
        if(_src.isGpuMat())
        {
            UMat tmp;
            Mat(_src.getGpuMat()).copyTo(tmp);

            ret.push_back(tmp);
        }
#endif
        if(_src.isMatVector() || _src.isUMatVector())
            _src.getUMatVector(ret);


#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;
            _src.getGpuMatVector(tmp);

            ret.resize(tmp.size());

            std::transform(tmp.begin(), tmp.end(), ret.begin(), [](const cuda::GpuMat& g)->UMat{UMat tmp; Mat(g).copyTo(tmp); return tmp;});
        }
#endif

        ret.shrink_to_fit();
    }
};

template<class B>
struct get_input_helper<std::vector<UMat>, B, true>
{
    typedef std::vector<UMat> output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;

        bool copy_needed(false);

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
        {
            ret.push_back(_src.getUMat());

            copy_needed = _src.isUMat();
        }
#ifdef HAVE_CUDA
        if(_src.isGpuMat())
        {
            UMat tmp;
            Mat(_src.getGpuMat()).copyTo(tmp);

            ret.push_back(tmp);
        }
#endif
        if(_src.isMatVector() || _src.isUMatVector())
        {
            _src.getUMatVector(ret);

            copy_needed = _src.isUMatVector();
        }


#ifdef HAVE_CUDA
        if(_src.isGpuMatVector())
        {
            std::vector<cuda::GpuMat> tmp;
            _src.getGpuMatVector(tmp);

            ret.resize(tmp.size());

            std::transform(tmp.begin(), tmp.end(), ret.begin(), [](const cuda::GpuMat& g)->UMat{UMat tmp; Mat(g).copyTo(tmp); return tmp;});
        }
#endif

        ret.shrink_to_fit();

        if(copy_needed)
        {
            _dst.resize(ret.size());
            _dst.shrink_to_fit();

            transform(ret.begin(), ret.end(), _dst.begin(), [](const UMat& m)->UMat{ return m.clone();});
        }
        else
        {
            _dst = std::move(ret);
        }
    }
};

#ifdef HAVE_CUDA

template<class B>
struct get_input_helper<std::vector<cuda::GpuMat>, B, false>
{
    typedef std::vector<cuda::GpuMat> output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& ret)
    {
        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret.push_back(cuda::GpuMat(_src.getMat()));

        if(_src.isGpuMat())
            ret.push_back(_src.getGpuMat());


        if(_src.isMatVector() || _src.isUMatVector())
        {
            std::vector<Mat> tmp_src;

            _src.getMatVector(tmp_src);

            ret.resize(tmp_src.size());

            std::transform(tmp_src.begin(), tmp_src.end(), ret.begin(), [](const Mat& m)->cuda::GpuMat{return cuda::GpuMat(m);});
        }

        if(_src.isGpuMatVector())
            _src.getGpuMatVector(ret);

        ret.shrink_to_fit();
    }
};

template<class B>
struct get_input_helper<std::vector<cuda::GpuMat>, B, true>
{
    typedef std::vector<cuda::GpuMat> output_type;
    typedef B input_type;

    static output_type get_arg(input_type _src)
    {
        output_type ret;

        get_arg2(_src, ret);

        return ret;
    }

    static void get_arg2(input_type _src, output_type& _dst)
    {
        output_type ret;

        bool copy_needed(false);

        if(_src.isMat() ||_src.isUMat() || _src.kind() == _InputArray::EXPR || isScalar(_src) )
            ret.push_back(cuda::GpuMat(_src.getMat()));

        if(_src.isGpuMat())
        {
            ret.push_back(_src.getGpuMat());
            copy_needed = true;
        }


        if(_src.isMatVector() || _src.isUMatVector())
        {
            std::vector<Mat> tmp_src;

            _src.getMatVector(tmp_src);

            ret.resize(tmp_src.size());

            std::transform(tmp_src.begin(), tmp_src.end(), ret.begin(), [](const Mat& m)->cuda::GpuMat{return cuda::GpuMat(m);});
        }

        if(_src.isGpuMatVector())
        {
            _src.getGpuMatVector(ret);
            copy_needed = true;
        }

        ret.shrink_to_fit();

        if(copy_needed)
        {
            _dst.resize(ret.size());
            _dst.shrink_to_fit();

            transform(ret.begin(), ret.end(), _dst.begin(),[](const cuda::GpuMat& m)->cuda::GpuMat{ cuda::GpuMat tmp; m.copyTo(tmp); return tmp;});
        }
        else
        {
            _dst = std::move(ret);
        }
    }
};

#endif

template<class A, class B>
struct set_output_helper
{
    typedef A argument_type;
    typedef B output_type;

    static void set_arg(output_type& , const argument_type& ){}
};

template<class B>
struct set_output_helper<Mat, B>
{
    typedef Mat argument_type;
    typedef B output_type;

    static void set_arg(output_type& _dst, const argument_type& _arg)
    {
        if(_dst.isMat()  || _dst.isUMat())
            _dst.assign(_arg);
#ifdef HAVE_CUDA
        if(_dst.isGpuMat())
            cuda::GpuMat(_arg).copyTo(_dst);
#endif
        if(_dst.isMatVector())
        {
            _dst.create(1, 1, CV_8U);

            _dst.getMatRef(0) = _arg;
        }

        if(_dst.isUMatVector())
        {
            _dst.create(1, 1, CV_8U);

            _arg.copyTo(_dst.getUMatRef(0));
        }

#ifdef HAVE_CUDA
        if(_dst.isGpuMatVector())
        {
            std::vector<cuda::GpuMat>& tmp = _dst.getGpuMatVecRef();

            tmp.clear();

            tmp.resize(1, cuda::GpuMat(_arg));

            tmp.shrink_to_fit();
        }
#endif
    }
};

template<class B>
struct set_output_helper<UMat, B>
{
    typedef UMat argument_type;
    typedef B output_type;

    static void set_arg(output_type& _dst, const argument_type& _arg)
    {
        if(_dst.isMat()  || _dst.isUMat())
            _dst.assign(_arg);
#ifdef HAVE_CUDA
        if(_dst.isGpuMat())
            cuda::GpuMat(_arg).copyTo(_dst);
#endif
        if(_dst.isMatVector())
        {
            _dst.create(1, 1, CV_8U);

            _arg.copyTo(_dst.getMatRef(0));
        }

        if(_dst.isUMatVector())
        {
            _dst.create(1, 1, CV_8U);

            _dst.getUMatRef(0) = _arg;
        }

#ifdef HAVE_CUDA
        if(_dst.isGpuMatVector())
        {
            std::vector<cuda::GpuMat>& tmp = _dst.getGpuMatVecRef();
            Mat tmp2;

            _arg.copyTo(tmp2);

            tmp.clear();

            tmp.resize(1, cuda::GpuMat(tmp2));

            tmp.shrink_to_fit();
        }
#endif

    }
};

#ifdef HAVE_CUDA

template<class B>
struct set_output_helper<cuda::GpuMat, B>
{
    typedef cuda::GpuMat argument_type;
    typedef B output_type;

    static void set_arg(output_type& _dst, const argument_type& _arg)
    {
        if(_dst.isMat()  || _dst.isUMat())
            _dst.assign(Mat(_arg));

        if(_dst.isGpuMat())
            _arg.copyTo(_dst);

        if(_dst.isMatVector())
        {
            _dst.create(1, 1, CV_8U);

            _dst.getMatRef(0) = Mat(_arg);
        }

        if(_dst.isUMatVector())
        {
            _dst.create(1, 1, CV_8U);

            Mat(_arg).copyTo(_dst.getUMatRef(0) );
        }

        if(_dst.isGpuMatVector())
        {
            std::vector<cuda::GpuMat>& tmp = _dst.getGpuMatVecRef();

            tmp.clear();

            tmp.resize(1, _arg);

            tmp.shrink_to_fit();
        }
    }
};

#endif


template<class B>
struct set_output_helper<std::vector<Mat>, B>
{
    typedef std::vector<Mat> argument_type;
    typedef B output_type;

    static void set_arg(output_type& _dst, const argument_type& _arg)
    {
        CV_Assert(isVectorOfMatrix(_dst));


        if(_dst.isMatVector() || _dst.isUMatVector())
        {
            _dst.create(_arg.size(), 1, CV_8U);

            _dst.assign(_arg);
        }

#ifdef HAVE_CUDA
        if(_dst.isGpuMatVector())
        {
            std::vector<cuda::GpuMat>& tmp = _dst.getGpuMatVecRef();

            tmp.clear();

            tmp.resize(_arg.size());

            tmp.shrink_to_fit();

            std::transform(_arg.begin(), _arg.end(), tmp.begin(), [](const Mat& m)->cuda::GpuMat{ return cuda::GpuMat(m);});
        }
#endif
    }
};

template<class B>
struct set_output_helper<std::vector<UMat>, B>
{
    typedef std::vector<UMat> argument_type;
    typedef B output_type;

    static void set_arg(output_type& _dst, const argument_type& _arg)
    {
        CV_Assert(isVectorOfMatrix(_dst));


        if(_dst.isMatVector() || _dst.isUMatVector())
        {
            _dst.create(_arg.size(), 1, CV_8U);

            _dst.assign(_arg);
        }

#ifdef HAVE_CUDA
        if(_dst.isGpuMatVector())
        {
            std::vector<cuda::GpuMat>& tmp = _dst.getGpuMatVecRef();

            tmp.clear();

            tmp.resize(_arg.size());

            tmp.shrink_to_fit();

            std::transform(_arg.begin(), _arg.end(), tmp.begin(), [](const UMat& m)->cuda::GpuMat{Mat tmp; m.copyTo(tmp); return cuda::GpuMat(tmp);});
        }
#endif
    }
};

#ifdef HAVE_CUDA
template<class B>
struct set_output_helper<std::vector<cuda::GpuMat>, B>
{
    typedef std::vector<cuda::GpuMat> argument_type;
    typedef B output_type;

    static void set_arg(output_type& _dst, const argument_type& _arg)
    {
        CV_Assert(isVectorOfMatrix(_dst));

        if(_dst.isMatVector() || _dst.isUMatVector())
        {
            _dst.create(_arg.size(), 1, CV_8U);

            size_t i=0;

            for(const cuda::GpuMat& s : _arg)
                _dst.getMatRef(i++) = Mat(s);
        }

#ifdef HAVE_CUDA
        if(_dst.isGpuMatVector())
        {
            std::vector<cuda::GpuMat>& tmp = _dst.getGpuMatVecRef();

            tmp = std::move(_arg);
        }
#endif
    }
};
#endif

} // anonymous

#define IMPL_GET_INPUT(type)\
    template<>\
    type getInput<type>(InputArray& _src, const bool& copy)\
    {\
        return copy ? get_input_helper<type, InputArray, true>::get_arg(_src) : get_input_helper<type, InputArray, false>::get_arg(_src);\
    }\
    \
    template<>\
    type getInput<type>(InputOutputArray& _src, const bool& copy)\
    {\
        return copy ? get_input_helper<type, InputOutputArray, true>::get_arg(_src) : get_input_helper<type, InputOutputArray, false>::get_arg(_src);\
    }\
    \
    template<>\
    void getInput<type>(InputArray& _src, type& _dst, const bool& copy)\
    {\
        copy ? get_input_helper<type, InputArray, true>::get_arg2(_src, _dst) : get_input_helper<type, InputArray, false>::get_arg2(_src, _dst);\
    }\
    \
    template<>\
    void getInput<type>(InputOutputArray& _src, type& _dst, const bool& copy)\
    {\
        copy ? get_input_helper<type, InputOutputArray, true>::get_arg2(_src, _dst) : get_input_helper<type, InputOutputArray, false>::get_arg2(_src, _dst);\
    }

IMPL_GET_INPUT(Mat)
IMPL_GET_INPUT(UMat)
#ifdef HAVE_CUDA
IMPL_GET_INPUT(cuda::GpuMat)
#endif
IMPL_GET_INPUT(std::vector<Mat>)
IMPL_GET_INPUT(std::vector<UMat>)
#ifdef HAVE_CUDA
IMPL_GET_INPUT(std::vector<cuda::GpuMat>)
#endif
#undef IMPL_GET_INPUT

#define IMPL_SET_OUTPUT(type)\
    template<>\
    void setOutput<type>(OutputArray& _dst, const type& _arg)\
    {\
        set_output_helper<type, OutputArray>::set_arg(_dst, _arg);\
    }\
    \
    template<>\
    void setOutput<type>(InputOutputArray& _dst, const type& _arg)\
    {\
        set_output_helper<type, InputOutputArray>::set_arg(_dst, _arg);\
    }


IMPL_SET_OUTPUT(Mat)
IMPL_SET_OUTPUT(UMat)
#ifdef HAVE_CUDA
IMPL_SET_OUTPUT(cuda::GpuMat)
#endif
IMPL_SET_OUTPUT(std::vector<Mat>)
IMPL_SET_OUTPUT(std::vector<UMat>)
#ifdef HAVE_CUDA
IMPL_SET_OUTPUT(std::vector<cuda::GpuMat>)
#endif
#undef IMPL_SET_OUTPUT


namespace
{

template<class T>
Scalar getScalar_(T& _src)
{
    CV_Assert( isScalar_<T>(_src) );

    Scalar ret;

    Mat tmp = getInput<Mat>(_src);

    tmp.convertTo(Mat(tmp.size(), CV_64FC(tmp.channels()), ret.val), CV_64F);

    return ret;
}

} // anonymous

Scalar getScalar(InputArray& _src)
{
    return getScalar_<InputArray>(_src);
}

Scalar getScalar(InputOutputArray& _src)
{
    return getScalar_<InputOutputArray>(_src);
}

Scalar fromAnythingToScalar(InputArray _src)
{
    return getScalar_(_src);
}


Mat toMat(InputArray _src)
{
    return getInput<Mat>(_src);
}

UMat toUMat(InputArray _src)
{
    return getInput<UMat>(_src);
}
#ifdef HAVE_CUDA
cuda::GpuMat toGpuMat(InputArray _src)
{
    return getInput<cuda::GpuMat>(_src);
}
#endif
std::vector<Mat> toMatVector(InputArray _src)
{
    return getInput<std::vector<Mat> >(_src);
}
std::vector<UMat> toUMatVector(InputArray _src)
{
    return getInput<std::vector<UMat> >(_src);
}
#ifdef HAVE_CUDA
std::vector<cuda::GpuMat> toGpuMatVector(InputArray _src)
{
    return getInput<std::vector<cuda::GpuMat> >(_src);
}
#endif




void toMat(InputArray _src, Mat& ret)
{
    getInput<Mat>(_src, ret);
}
void toUMat(InputArray _src, UMat& ret)
{
    getInput<UMat>(_src, ret);
}
#ifdef HAVE_CUDA
void toGpuMat(InputArray _src, cuda::GpuMat& ret)
{
    getInput<cuda::GpuMat>(_src, ret);
}
#endif
void toMatVector(InputArray _src, std::vector<Mat>& ret)
{
    getInput<std::vector<Mat> >(_src, ret);
}
void toUMatVector(InputArray _src, std::vector<UMat>& ret)
{
    getInput<std::vector<UMat> >(_src, ret);
}
#ifdef HAVE_CUDA
void toGpuMatVector(InputArray _src, std::vector<cuda::GpuMat>& ret)
{
    getInput<std::vector<cuda::GpuMat> >(_src, ret);
}
#endif


}// cv
