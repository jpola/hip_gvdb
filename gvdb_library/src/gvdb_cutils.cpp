//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "gvdb_cutils.h"
//#include <rocthrust/rocprim.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>


// Sorts inSource (an array of uint64_t elements of length inCount) in place
// using radix sort.
void gvdbDeviceRadixSort(hipDeviceptr_t inSource, const uint32_t inCount){
	// Interpret inSource as an array, and get iterators to its beginning and end.
	
	thrust::device_ptr<uint64_t> d_begin = thrust::device_pointer_cast((uint64_t*)inSource);
	thrust::device_ptr<uint64_t> d_end = d_begin + inCount;
	thrust::sort(d_begin, d_end, thrust::less<uint64_t>());
}

// Computes the maximum element of inData, and stores it in the first element of inMax.
// Types:
//   inMax: a floating-point array of length at least 1.
//   inData: a floating-point array of length inCount.
void gvdbDeviceMaxElementF(hipDeviceptr_t inMax, hipDeviceptr_t inData, const uint32_t inCount){
	thrust::device_ptr<float> d_data_begin = thrust::device_pointer_cast((float*)inData);
	thrust::device_ptr<float> d_data_end = d_data_begin + inCount;
	thrust::device_ptr<float> d_max = thrust::device_pointer_cast((float*)inMax);
    thrust::device_vector<float>::iterator iter = thrust::max_element(d_data_begin, d_data_end);
    d_max[0] = *iter;
}

// Computes the minimum element of inData, and stores it in the first element of inMin.
// Types:
//   inMin: a floating-point array of length at least 1.
//   inData: a floating-point array of length inCount.
void gvdbDeviceMinElementF(hipDeviceptr_t inMin, hipDeviceptr_t inData, const uint32_t inCount) {
	thrust::device_ptr<float> d_data_begin = thrust::device_pointer_cast((float*)inData);
	thrust::device_ptr<float> d_data_end = d_data_begin + inCount;
	thrust::device_ptr<float> d_min = thrust::device_pointer_cast((float*)inMin);
	thrust::device_vector<float>::iterator iter = thrust::min_element(d_data_begin, d_data_end);
	d_min[0] = *iter;
}