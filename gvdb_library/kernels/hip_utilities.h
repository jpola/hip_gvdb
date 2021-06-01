#ifndef HIP_UTILITIES_H
#define HIP_UTILITIES_H

#include "hip_math.h"

#define EXPAND3(a) a.x, a.y, a.z

namespace {

using HIPBlockDim = __HIP_Coordinates<__HIP_BlockDim>;
using HIPBlockIdx = __HIP_Coordinates<__HIP_BlockIdx>;
using HIPThreadIdx = __HIP_Coordinates<__HIP_ThreadIdx>;

template<class BI, class BD, class TI>
inline __device__ uint3 Get3DPixelJob(const BI& bi, const BD& bd, const TI& ti)
{
    return make_uint3(EXPAND3(bi)) * make_uint3(EXPAND3(bd)) + make_uint3(EXPAND3(ti));
}
}



#endif //HIP_UTILITIES_H