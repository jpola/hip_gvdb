//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#ifndef __CUH_GVDB_CUTILS__
#define __CUH_GVDB_CUTILS__

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

void gvdbDeviceRadixSort(hipDeviceptr_t inSource, const uint32_t inCount);
void gvdbDeviceMaxElementF(hipDeviceptr_t inSource, hipDeviceptr_t inDest, const uint32_t inCount);
void gvdbDeviceMinElementF(hipDeviceptr_t inSource, hipDeviceptr_t inDest, const uint32_t inCount);


#endif