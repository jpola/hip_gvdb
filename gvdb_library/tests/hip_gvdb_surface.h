#ifndef HIP_GVDV_SURFACE_H
#define HIP_GVDV_SURFACE_H

#if defined (__HIP__)
#include <hip/hip_runtime.h>


#define __SURFACE_FUNCTIONS_DECL__ static inline __device__
template <class T>
__SURFACE_FUNCTIONS_DECL__ void surf3Dread(T* data, hipSurfaceObject_t surfObj, int x, int y, int z, int boundaryMode = hipBoundaryModeTrap) {
    hipArray* arrayPtr = (hipArray*)surfObj;
    size_t width = arrayPtr->width;
    size_t height = arrayPtr->height;
    size_t depth = arrayPtr->depth;
    int32_t xOffset = x / sizeof(T);

    T* dataPtr = (T*)arrayPtr->data;
    
    if ((xOffset > width) || (xOffset < 0) || (y > height) || (y < 0) || (z > depth) || (z < 0)) {
        if (boundaryMode == hipBoundaryModeZero) {
            *data = T {0};
        }
    } else {
        *data = *(dataPtr + width * (y + (depth * z)) + xOffset);
    }
}

#define __SURFACE_FUNCTIONS_DECL__ static inline __device__
template <class T>
__SURFACE_FUNCTIONS_DECL__ T surf3Dread(hipSurfaceObject_t surfObj, int x, int y, int z, int boundaryMode = hipBoundaryModeTrap) {
    
    T value = T{0};
    surf3Dread(&value, surfObj, x, y, z, boundaryMode);
    return value;
}


template <class T>
__SURFACE_FUNCTIONS_DECL__ void surf3Dwrite(T data, hipSurfaceObject_t surfObj, int x, int y, int z, int boundaryMode = hipBoundaryModeTrap) {
    hipArray* arrayPtr = (hipArray*)surfObj;
    size_t width = arrayPtr->width;
    size_t height = arrayPtr->height;
    size_t depth = arrayPtr->depth;
    int32_t xOffset = x / sizeof(T);
    T* dataPtr = (T*)arrayPtr->data;
    if (!((xOffset > width) || (xOffset < 0) || (y > height) || (y < 0))) {
        *(dataPtr + width * (y + (depth * z)) + xOffset) = data;
    }
}

#endif 

#endif //HIP_GVDV_SURFACE_H