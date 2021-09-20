#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define noconvert

__kernel void centreToTheMeanAxis0(__global const uchar* src1ptr, const int step_src1, const int offset_src1,
                                   __global const uchar* src2ptr, const int step_src2, const int offset_src2,
                                   __global uchar* dstptr, const int step_dst, const int offset_dst, const int rows, const int cols)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if(x>=cols || y0>=rows)
        return;

    int src1_index = mad24(y0, step_src1, mad24(x, (int)sizeof(srcT1), offset_src1));
    int src2_index = mad24(x, (int)sizeof(srcT2), offset_src2);
    int dst_index = mad24(y0, step_dst, mad24(x, (int)sizeof(dstT), offset_dst));

    for (int y = y0, y1 = min(rows, y0 + rowsPerWI); y < y1; y++, src1_index += step_src1, dst_index += step_dst)
    {
        __global const srcT1 * src1 = (__global const srcT1 *)(src1ptr + src1_index);
        __global const srcT2 * src2 = (__global const srcT2 *)(src2ptr + src2_index);
        __global dstT * dst = (__global dstT *)(dstptr + dst_index);

        *dst = convertToDT(convertToWT1(*src1) - convertToWT2(*src2));
    }
}
