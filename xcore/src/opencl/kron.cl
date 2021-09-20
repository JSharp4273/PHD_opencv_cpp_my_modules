#define noconvert

__kernel void kkron(__global const uchar* src1_ptr, const int src1_step, const int src1_offset, const int src1_rows, const int src1_cols,
                    __global const uchar* src2_ptr, const int src2_step, const int src2_offset, const int src2_rows, const int src2_cols,
                    __global uchar* dst_ptr, const int dst_step, const int dst_offset, const int dst_rows, const int dst_cols)
                    {

    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if(y0>=src1_rows || x>=src1_cols)
        return;


    int src1_index = mad24(y0, src1_step, mad24(x, (int)sizeof(T), src1_offset));
    int src2_index = src2_offset;
    int dst_index = mad24(mul24(y0, src2_rows), dst_step, mad24(mul24(x, src2_cols), (int)sizeof(T), dst_offset));

    T v = *((__global const T*)(src1_ptr+src1_index));

    __global const T* current_src2_ptr = (__global const T*)(src2_ptr + src2_index);
    __global T* current_dst_ptr = (__global T*)(dst_ptr + dst_index);

    int src2_step1 = src2_step / (int)sizeof(T);
    int dst_step1 = dst_step / (int)sizeof(T);

    for(int r=0; r<src2_rows; r++, current_src2_ptr+=src2_step1, current_dst_ptr+=dst_step1)
    {
        __global const T* it_src = current_src2_ptr;
        __global T* it_dst = current_dst_ptr;

        for(int c=0; c<src2_cols; c++, it_src++, it_dst++)
            *it_dst = v * *it_src;
    }
}

