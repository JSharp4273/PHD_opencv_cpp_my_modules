#include "opencv2/xcore.hpp"
#include "opencv2/xcore/template/intrin.hpp"
#include "opencv2/core/ocl.hpp"

#include "opencv2/core/ocl_genbase.hpp"

//#include <fstream>

namespace cv
{


namespace ocl
{

namespace xcore
{


extern struct cv::ocl::internal::ProgramEntry xcore_oclkron;

struct cv::ocl::internal::ProgramEntry kron =
{
    "xcore",
    "kron",
    "#define noconvert\n"
    "\n"
    "__kernel void kkron(__global const uchar* src1_ptr, const int src1_step, const int src1_offset, const int src1_rows, const int src1_cols,\n"
    "                    __global const uchar* src2_ptr, const int src2_step, const int src2_offset, const int src2_rows, const int src2_cols,\n"
    "                    __global uchar* dst_ptr, const int dst_step, const int dst_offset, const int dst_rows, const int dst_cols)\n"
    "                    {\n"
    "\n"
    "    int x = get_global_id(0);\n"
    "    int y0 = get_global_id(1) * rowsPerWI;\n"
    "\n"
    "    if(y0>=src1_rows || x>=src1_cols)\n"
    "        return;\n"
    "\n"
    "\n"
    "    int src1_index = mad24(y0, src1_step, mad24(x, (int)sizeof(T), src1_offset));\n"
    "    int src2_index = src2_offset;\n"
    "    int dst_index = mad24(mul24(y0, src2_rows), dst_step, mad24(mul24(x, src2_cols), (int)sizeof(T), dst_offset));\n"
    "\n"
    "    T v = *((__global const T*)(src1_ptr+src1_index));\n"
    "\n"
    "    __global const T* current_src2_ptr = (__global const T*)(src2_ptr + src2_index);\n"
    "    __global T* current_dst_ptr = (__global T*)(dst_ptr + dst_index);\n"
    "\n"
    "    int src2_step1 = src2_step / (int)sizeof(T);\n"
    "    int dst_step1 = dst_step / (int)sizeof(T);\n"
    "\n"
    "    for(int r=0; r<src2_rows; r++, current_src2_ptr+=src2_step1, current_dst_ptr+=dst_step1)\n"
    "    {\n"
    "        __global const T* it_src = current_src2_ptr;\n"
    "        __global T* it_dst = current_dst_ptr;\n"
    "\n"
    "        for(int c=0; c<src2_cols; c++, it_src++, it_dst++)\n"
    "            *it_dst = v * *it_src;\n"
    "    }\n"
    "}\n",
    "61e807be22793264eddbe34d6e9e28bd",
    nullptr
};

} // xcore

} // ocl


namespace
{


template<class T>
class ParallelKron : public ParallelLoopBody
{
public:

    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;

    ParallelKron(const Mat& _A, const Mat& _B, Mat& _C);

    virtual ~ParallelKron() = default;

    virtual void operator()(const Range& range) const;


private:

    typedef typename Type2Vec_Traits<T>::vec_type vec_type;

    const Mat& A;
    const Mat& B;
    Mat& C;
#if CV_SIMD
   const int vec_width;
#endif

};

template<class T>
ParallelKron<T>::ParallelKron(const Mat& _A, const Mat& _B, Mat& _C):
    A(_A),
    B(_B),
    C(_C)
#if CV_SIMD
    , vec_width(_B.cols - (_B.cols%vec_type::nlanes))
#endif
{}

Mutex mtx;

template<class T>
void ParallelKron<T>::operator()(const Range& range)const
{
#if CV_SIMD
    static const int inc = static_cast<int>(vec_type::nlanes);
#endif

    for(int r=range.start, rr=range.start * B.rows; r<range.end; r++, rr+=B.rows)
    {
        for(int c=0, cc=0; c<A.cols; c++, cc+=B.cols)
        {
            value_type v = A.template at<value_type>(r,c);
#if CV_SIMD
            vec_type vv = vx_setall(v);
#endif
            const_pointer ptr_B = this->B.template ptr<value_type>();

            pointer ptr_dst = this->C.template ptr<value_type>(rr, cc);

            cv::AutoLock lck(mtx);

            for(int rk=0, ck=0; rk<this->B.rows; rk++, ck=0, ptr_B+=this->B.step1(), ptr_dst+=this->C.step1())
            {
                const_pointer it_B = ptr_B;
                pointer it_dst = ptr_dst;

#if CV_SIMD
                for(; ck<this->vec_width;ck+=inc, it_B+=inc, it_dst+=inc)
                    vx_store(it_dst, vx_load(it_B) * vv);
#endif

#if CV_ENABLE_UNROLLED
                    for(;ck<=this->B.cols-4;ck+=4, it_B+=4, it_dst+=4)
                    {
                        value_type b0 = it_B[0];
                        value_type b1 = it_B[1];

                        b0*=v;
                        b1*=v;

                        it_dst[0] = b0;
                        it_dst[1] = b1;


                        b0 = it_B[2];
                        b1 = it_B[3];

                        b0*=v;
                        b1*=v;

                        it_dst[2] = b0;
                        it_dst[3] = b1;
                    }
#endif
                    for(;ck<this->B.cols;ck++, it_B++, it_dst++)
                        *it_dst = *it_B * v;

            }
        }
    }
}

template<class T>
void kron_(const Mat& _A, const Mat& _B, Mat& _C)
{
    Mat_<T> A(_A), B(_B), C(_C);


#ifdef HAVE_HIGH_PRIORITY_PARFOR
highPrioriyParallelFor(
#else
parallel_for_(
#endif
                Range(0, A.rows), ParallelKron<T>(A,B,C) );
}

/////////////////////////////////////////////////



/////////////////////////////////////////////////



bool ocl_kron(InputArray& _src1, InputArray& _src2, OutputArray& _dst)
{

    UMat usrc1 = _src1.getUMat();
    UMat usrc2 = _src2.getUMat();


    const int sdepth = usrc1.depth();
    const int wdepth = std::max(sdepth, CV_32F);


    UMat udst(usrc1.rows * usrc2.rows, usrc1.cols * usrc2.cols, sdepth, 0.);


    const ocl::Device& dev = ocl::Device::getDefault();

    if(!dev.doubleFPConfig() && (wdepth == CV_64F))
        return false;



    std::size_t wgs = dev.maxWorkGroupSize();

    std::size_t lsz = wgs;

    static ocl::ProgramSource ps;
    static bool is_kernel_loaded = false;

//    if(!is_kernel_loaded)
//    {
//        std::fstream stream("../ml/xcore/opencl/kron.cl", std::ios::in);


//        if(stream)
//        {
//            std::stringstream stream_buf;

//            stream_buf<<stream.rdbuf();

//            ps = ocl::ProgramSource(stream_buf.str());

//            is_kernel_loaded = true;
//        }
//        else
//            CV_Error(Error::StsError,"Kron: OpenCL kernel not found");
//    }


    const int rows_steps = static_cast<size_t>(usrc1.rows) > wgs ? cvCeil(static_cast<float>(usrc1.rows)/static_cast<float>(wgs)) : 1;
    const int cols_steps = static_cast<size_t>(usrc1.cols) > wgs ? cvCeil(static_cast<float>(usrc1.cols)/static_cast<float>(wgs)) : 1;

    const int cn = dev.type() & ocl::Device::TYPE_GPU ? 4 : dev.type() & ocl::Device::TYPE_CPU ? 8 : 1;

    char CV_DECL_ALIGNED(0x10) buffer[0x40];

    int rowsPerWI = dev.isIntel() ? 4 : 1;

    ocl::Kernel k("kkron", ocl::xcore::kron, format(
                      "-D T=%s "
                      "-D TCN=%s "
                      "-D ROWS_STEPS=%d "
                      "-D COLS_STEPS=%d "
                      "-D cn=%d "
                      "-D LoadV=vload%d "
                      "-D StoreV=vstore%d "
                      "-D convertToW=%s "
                      "-D convertToDst=%s "
                      "-D rowsPerWI=%d ",
                      ocl::typeToStr(sdepth),
                      ocl::typeToStr(CV_MAKETYPE(sdepth, cn) ),
                      rows_steps,
                      cols_steps,
                      cn,
                      cn,
                      cn,
                      ocl::convertTypeStr(sdepth, wdepth, cn, buffer),
                      ocl::convertTypeStr(wdepth, sdepth, cn, buffer+0x20),
                      rowsPerWI
                      )
                  );

    if (k.empty())
        return false;

    k.args(ocl::KernelArg::ReadOnly(usrc1),
           ocl::KernelArg::ReadOnly(usrc2),
           ocl::KernelArg::WriteOnly(udst));

    std::size_t dims[2] = {static_cast<size_t>(usrc1.rows), static_cast<size_t>(usrc1.cols)};


    if(!k.run(2,dims, nullptr, false))
        return false;

    if(!_dst.fixedType() || (_dst.fixedType() && udst.type() == _dst.type()))
        udst.copyTo(_dst);
    else
        udst.convertTo(_dst, _dst.depth());

    return true;
}


}


void kron(InputArray _src1, InputArray _src2, OutputArray _dst)
{
    CV_Assert(_src1.channels() == 1 && _src2.channels() == 1 && _src1.depth() == _src2.depth());

    if(ocl::haveOpenCL() && _dst.isUMat() && ocl_kron(_src1, _src2, _dst))
        return;

    typedef void(*function_type)(const Mat&, const Mat&, Mat&);

    static const function_type funcs[2] = { kron_<float>, kron_<double>};


    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat dst;

    const int sdepth = src1.depth();
    const int wdepth = std::max(sdepth, CV_32F);

    dst = Mat::zeros(src1.rows * src2.rows, src1.cols * src2.cols, wdepth);

    function_type fun = funcs[wdepth-CV_32F];

    fun(src1, src2, dst);

    sdepth == wdepth ? dst.copyTo(_dst) : dst.convertTo(_dst, sdepth);
}

}
