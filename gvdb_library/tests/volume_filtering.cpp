#include <stdio.h>
//#include <helper_cuda.h>

texture<float, cudaTextureType3D, cudaReadModeElementType>  volumeTexIn;
surface<void,  3>                                    volumeTexOut;

__global__ void
surf_write(float *data,cudaExtent volumeSize)
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
    surf3Dwrite(output,volumeTexOut,x * sizeof(float),y,z);

}

__global__ void
tex_read(float x, float y, float z){
    printf("x: %f, y: %f, z:%f, val: %f\n", x,y,z,tex3D(volumeTexIn,x,y,z));
}

void runtest(float *data, cudaExtent vol, float x, float y, float z)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t content;
    checkCudaErrors(cudaMalloc3DArray(&content, &channelDesc, vol, cudaArraySurfaceLoadStore));

    // copy data to device
    float *d_data;
    checkCudaErrors(cudaMalloc(&d_data, vol.width*vol.height*vol.depth*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_data, data, vol.width*vol.height*vol.depth*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(8,8,8);
    dim3 gridSize((vol.width+7)/8,(vol.height+7)/8,(vol.depth+7)/8);
    volumeTexIn.filterMode     = cudaFilterModeLinear;
    checkCudaErrors(cudaBindSurfaceToArray(volumeTexOut,content));
    surf_write<<<gridSize, blockSize>>>(d_data, vol);
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(volumeTexIn, content));
    tex_read<<<1,1>>>(x, y, z);
    checkCudaErrors(cudaDeviceSynchronize());
    cudaFreeArray(content);
    cudaFree(d_data);
    return;
}

int main(){
   const int dim = 8;
   float *data = (float *)malloc(dim*dim*dim*sizeof(float));
   for (int z = 0; z < dim; z++)
     for (int y = 0; y < dim; y++)
       for (int x = 0; x < dim; x++)
         data[z*dim*dim+y*dim+x] = z*100+y*10+x;
   cudaExtent vol = {dim,dim,dim};
   runtest(data, vol, 1.5, 1.5, 1.5);
   runtest(data, vol, 1.6, 1.6, 1.6);
   r