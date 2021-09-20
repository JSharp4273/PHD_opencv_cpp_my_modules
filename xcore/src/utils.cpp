//#include "utils.hpp"

//#include "template/traits.hpp"

//#ifdef _DEBUG
//#include <iostream>
//#endif

//namespace cv
//{

//namespace
//{
//// template vector class. It is similar to STL's vector,
//// with a few important differences:
////   1) it can be created on top of user-allocated data w/o copying it
////   2) vector b = a means copying the header,
////      not the underlying data (use clone() to make a deep copy)
//template <typename _Tp> class CV_EXPORTS Vector
//{
//public:
//    typedef _Tp value_type;
//    typedef _Tp* iterator;
//    typedef const _Tp* const_iterator;
//    typedef _Tp& reference;
//    typedef const _Tp& const_reference;

//    struct CV_EXPORTS Hdr
//    {
//        Hdr() : data(0), datastart(0), refcount(0), size(0), capacity(0) {};
//        _Tp* data;
//        _Tp* datastart;
//        int* refcount;
//        size_t size;
//        size_t capacity;
//    };

//    Vector() = default;
//    inline Vector(size_t _size)  { resize(_size); }
//    inline Vector(size_t _size, const _Tp& val)
//    {
//        resize(_size);
//        for(size_t i = 0; i < _size; i++)
//            hdr.data[i] = val;
//    }
//    inline Vector(_Tp* _data, size_t _size, bool _copyData=false)
//    { set(_data, _size, _copyData); }

//    template<int n> Vector(const Vec<_Tp, n>& vec)
//    { set((_Tp*)&vec.val[0], n, true); }

//    inline Vector(const std::vector<_Tp>& vec, bool _copyData=false)
//    { set((_Tp*)&vec[0], vec.size(), _copyData); }

//    inline Vector(const Vector& d) { *this = d; }

//    Vector(const Vector& d, const Range& r_)
//    {
//        Range r = r_ == Range::all() ? Range(0, d.size()) : r_;

//        if( r.size() > 0 && r.start >= 0 && r.end <= d.size() )
//        {
//            if( d.hdr.refcount )
//                CV_XADD(d.hdr.refcount, 1);
//            hdr.refcount = d.hdr.refcount;
//            hdr.datastart = d.hdr.datastart;
//            hdr.data = d.hdr.data + r.start;
//            hdr.capacity = hdr.size = r.size();
//        }
//    }

//    Vector<_Tp>& operator = (const Vector& d)
//    {
//        if( this != &d )
//        {
//            if( d.hdr.refcount )
//                CV_XADD(d.hdr.refcount, 1);
//            release();
//            hdr = d.hdr;
//        }
//        return *this;
//    }

//    ~Vector()  { release(); }

//    inline Vector<_Tp> clone() const
//    { return hdr.data ? Vector<_Tp>(hdr.data, hdr.size, true) : Vector<_Tp>(); }

//    void copyTo(Vector<_Tp>& vec) const
//    {
//        size_t i, sz = size();
//        vec.resize(sz);
//        const _Tp* src = hdr.data;
//        _Tp* dst = vec.hdr.data;
//        for( i = 0; i < sz; i++ )
//            dst[i] = src[i];
//    }

//    void copyTo(std::vector<_Tp>& vec) const
//    {
//        size_t i, sz = size();
//        vec.resize(sz);
//        const _Tp* src = hdr.data;
//        _Tp* dst = sz ? &vec[0] : 0;
//        for( i = 0; i < sz; i++ )
//            dst[i] = src[i];
//    }

//    inline operator Mat() const
//    { return Mat((int)size(), 1, type(), (void*)hdr.data); }

//    inline _Tp& operator [] (size_t i) { CV_DbgAssert( i < size() ); return hdr.data[i]; }
//    inline const _Tp& operator [] (size_t i) const { CV_DbgAssert( i < size() ); return hdr.data[i]; }
//    inline Vector operator() (const Range& r) const { return Vector(*this, r); }
//    inline _Tp& back() { CV_DbgAssert(!empty()); return hdr.data[hdr.size-1]; }
//    inline const _Tp& back() const { CV_DbgAssert(!empty()); return hdr.data[hdr.size-1]; }
//    inline _Tp& front() { CV_DbgAssert(!empty()); return hdr.data[0]; }
//    inline const _Tp& front() const { CV_DbgAssert(!empty()); return hdr.data[0]; }

//    inline _Tp* begin() { return hdr.data; }
//    inline _Tp* end() { return hdr.data + hdr.size; }
//    inline const _Tp* begin() const { return hdr.data; }
//    inline const _Tp* end() const { return hdr.data + hdr.size; }

//    inline void addref() { if( hdr.refcount ) CV_XADD(hdr.refcount, 1); }
//    void release()
//    {
//        if( hdr.refcount && CV_XADD(hdr.refcount, -1) == 1 )
//        {
//            delete[] hdr.datastart;
//            delete hdr.refcount;
//        }
//        hdr = Hdr();
//    }

//    void set(_Tp* _data, size_t _size, bool _copyData=false)
//    {
//        if( !_copyData )
//        {
//            release();
//            hdr.data = hdr.datastart = _data;
//            hdr.size = hdr.capacity = _size;
//            hdr.refcount = 0;
//        }
//        else
//        {
//            reserve(_size);
//            for( size_t i = 0; i < _size; i++ )
//                hdr.data[i] = _data[i];
//            hdr.size = _size;
//        }
//    }

//    void reserve(size_t newCapacity)
//    {
//        _Tp* newData;
//        int* newRefcount;
//        size_t i, oldSize = hdr.size;
//        if( (!hdr.refcount || *hdr.refcount == 1) && hdr.capacity >= newCapacity )
//            return;
//        newCapacity = std::max(newCapacity, oldSize);
//        newData = new _Tp[newCapacity];
//        newRefcount = new int(1);
//        for( i = 0; i < oldSize; i++ )
//            newData[i] = hdr.data[i];
//        release();
//        hdr.data = hdr.datastart = newData;
//        hdr.capacity = newCapacity;
//        hdr.size = oldSize;
//        hdr.refcount = newRefcount;
//    }

//    void resize(size_t newSize)
//    {
//        size_t i;
//        newSize = std::max(newSize, (size_t)0);
//        if( (!hdr.refcount || *hdr.refcount == 1) && hdr.size == newSize )
//            return;
//        if( newSize > hdr.capacity )
//            reserve(std::max(newSize, std::max((size_t)4, hdr.capacity*2)));
//        for( i = hdr.size; i < newSize; i++ )
//            hdr.data[i] = _Tp();
//        hdr.size = newSize;
//    }

//    Vector<_Tp>& push_back(const _Tp& elem)
//    {
//        if( hdr.size == hdr.capacity )
//            reserve( std::max((size_t)4, hdr.capacity*2) );
//        hdr.data[hdr.size++] = elem;
//        return *this;
//    }

//    Vector<_Tp>& pop_back()
//    {
//        if( hdr.size > 0 )
//            --hdr.size;
//        return *this;
//    }

//    inline size_t size() const { return hdr.size; }
//    inline size_t capacity() const { return hdr.capacity; }
//    inline bool empty() const { return hdr.size == 0; }
//    inline void clear() { resize(0); }
//    inline int type() const { return DataType<_Tp>::type; }

//protected:
//    Hdr hdr;
//};


//template<typename _Tp> inline typename DataType<_Tp>::work_type
//dot(const Vector<_Tp>& v1, const Vector<_Tp>& v2)
//{
//    typedef typename DataType<_Tp>::work_type _Tw;
//    size_t i, n = v1.size();
//    assert(v1.size() == v2.size());

//    _Tw s = 0;
//    const _Tp *ptr1 = &v1[0], *ptr2 = &v2[0];
//    for( i = 0; i <= n - 4; i += 4 )
//        s += (_Tw)ptr1[i]*ptr2[i] + (_Tw)ptr1[i+1]*ptr2[i+1] +
//                (_Tw)ptr1[i+2]*ptr2[i+2] + (_Tw)ptr1[i+3]*ptr2[i+3];
//    for( ; i < n; i++ )
//        s += (_Tw)ptr1[i]*ptr2[i];
//    return s;
//}



//};


//struct MatBuffer_CW::hdr_t
//{
//    int _type;
//    Vector<uchar> data;

//    inline hdr_t():
//        _type(-1)
//    {}

//    inline hdr_t(const int& type, const size_t& size):
//        _type(type),
//        data(size)
//    {}

//    ~hdr_t() = default;

//};

//MatBuffer_CW::MatBuffer_CW():
//    rows(0),
//    cols(0),
//    step(0),
//    data(nullptr),
//    head(new hdr_t)
//{}

//MatBuffer_CW::MatBuffer_CW(const int& _rows, const int& _cols, const int& type):
//    rows(_rows),
//    cols(_cols),
//    step(_rows * CV_ELEM_SIZE(type)),
//    data(nullptr),
//    head(new hdr_t(type, _cols * CV_ELEM_SIZE(type) * _rows) )
//{
//    size_t elemsize = CV_ELEM_SIZE(type);

//    this->data = this->head->data.begin();
//}

//MatBuffer_CW::MatBuffer_CW(const Mat& m):
//    MatBuffer_CW(m.rows, m.cols, m.type())
//{
//    size_t elemsize = CV_ELEM_SIZE(this->head->_type);

//    for(int r=0; r<m.rows; r++)
//        for(int c=0; c<m.cols; c++)
//            memcpy(this->_ptr(r,c), m.ptr(r,c), elemsize);
//}



////! returns element type
//int MatBuffer_CW::type() const
//{
//    return this->head->_type;
//}

////! returns element type
//int MatBuffer_CW::depth() const
//{
//    return CV_MAT_DEPTH(this->head->_type);
//}

////! returns number of channels
//int MatBuffer_CW::channels() const
//{
//    return CV_MAT_CN(this->head->_type);
//}

////! returns element size in bytes
//size_t MatBuffer_CW::elemSize() const
//{
//    return CV_ELEM_SIZE(this->head->_type);
//}

////! returns the size of element channel in bytes
//size_t MatBuffer_CW::elemSize1() const
//{
//    return CV_ELEM_SIZE1(this->head->_type);
//}

//uchar* MatBuffer_CW::ptr(int c)
//{
//    return this->_ptr(c);
//}

//uchar* MatBuffer_CW::ptr(int r, int c)
//{
//    return this->_ptr(r,c);
//}

//const uchar* MatBuffer_CW::ptr(int c) const
//{
//    return this->_ptr(c);
//}

//const uchar* MatBuffer_CW::ptr(int r, int c) const
//{
//    return this->_ptr(r,c);
//}

//MatBuffer_CW::operator Mat() const
//{
//    Mat ret(this->rows * this->cols, 1, this->head->_type);
//    std::memcpy(ret.data, this->data, this->head->data.size());

//    return ret;
//}

//template<class T>
//MatBuffer_CW::operator Mat_<T>() const
//{
//    CV_Assert(type2flag<T>::flag == this->head->_type);

//    Mat ret = (*this);

//    return ret;
//}

//template<class T>
//MatBuffer_CW::operator std::vector<T>() const
//{
//    CV_Assert(type2flag<T>::flag == this->head->_type);

//    Mat ret = (*this);

//    return ret;
//}

//// Private stuff

//uchar* MatBuffer_CW::_ptr(int c)
//{
//    return this->data + c * this->step;
//}

//const uchar* MatBuffer_CW::_ptr(int c) const
//{
//    return this->data + c * this->step;
//}

//uchar* MatBuffer_CW::_ptr(int r, int c)
//{
//    return this->_ptr(c) + r * CV_ELEM_SIZE(this->head->_type);
//}

//const uchar* MatBuffer_CW::_ptr(int r, int c) const
//{
//    return this->_ptr(c) + r * CV_ELEM_SIZE(this->head->_type);
//}


//};
