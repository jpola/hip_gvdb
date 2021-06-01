#include <hip/hip_runtime.h>
#include <stdio.h>
#include "hip_gvdb_surface.h"

texture<float, hipTextureType3D, hipReadModeElementType>  volumeTexIn;
surface<void,  3>                                         volumeTexOut;

__global__ void
surf_write(float *data,hipExtent volumeSize)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
    {
        return;
    }
    float output = data[z*(volumeSize.width*volumeSize.height)+y*(volumeSize.width)+x];
    // surface writes need byte offsets for x!
    //surf3Dwrite(output,volumeTexOut,x, y, z);
    surf3Dwrite(output,volumeTexOut,x * sizeof(float), y, z);

}

__global__ void
tex_read(float x, float y, float z){
    printf("x: %f, y: %f, z:%f, val: %f\n", x,y,z,tex3D(volumeTexIn,x,y,z));
}

void runtest(float *data, hipExtent vol, float x, float y, float z)
{
    // create 3D array
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();
    hipArray_t content;
    checkCudaErrors(hipMalloc3DArray(&content, &channelDesc, vol, hipArraySurfaceLoadStore));

    // copy data to device
    float *d_data;
    checkCudaErrors(hipMalloc(&d_data, vol.width*vol.height*vol.depth*sizeof(float)));
    checkCudaErrors(hipMemcpy(d_data, data, vol.width*vol.height*vol.depth*sizeof(float), hipMemcpyHostToDevice));

    dim3 blockSize(8,8,8);
    dim3 gridSize((vol.width+7)/8,(vol.height+7)/8,(vol.depth+7)/8);
    volumeTexIn.filterMode     = hipFilterModeLinear;
    checkCudaErrors(cudaBindSurfaceToArray(volumeTexOut,content));
    hipLaunchKernelGGL(surf_write, dim3(gridSize), dim3(blockSize), 0, 0, d_data, vol);
    // bind array to 3D texture
    checkCudaErrors(hipBindTextureToArray(volumeTexIn, content));
    hipLaunchKernelGGL(tex_read, dim3(1), dim3(1), 0, 0, x, y, z);
    checkCudaErrors(hipDeviceSynchronize());
    hipFreeArray(content);
    hipFree(d_data);
    return;
}

int main(){
   const int dim = 8;
   float *data = (float *)malloc(dim*dim*dim*sizeof(float));
   for (int z = 0; z < dim; z++)
     for (int y = 0; y < dim; y++)
       for (int x = 0; x < dim; x++)
         data[z*dim*dim+y*dim+x] = z*100+y*10+x;
   hipExtent vol = {dim,dim,dim};
   runtest(data, vol, 1.5, 1.5, 1.5);
   runtest(data, vol, 1.6, 1.6, 1.6);
   r