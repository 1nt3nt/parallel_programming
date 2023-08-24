#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "constMem.cuh"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>


/**
 * @brief loop unrolling disable 
 * @param res result variable on GPU
 */
__global__ void sequent_dis_unroll(float* res, float* input2, int width){
    float temp[NUM_ELEMENT];
    __shared__ float Nds[NUM_ELEMENT];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;

    // loading input2 into shared memory
    Nds[ty*width+tx] = input2[row*width+col];
    __syncthreads();  

    // computing
    if(row < width && col < width)
    {
        temp[col] = 0.0;
        #pragma unroll 1
        for(int j = 0; j < width; j++)
        {
            temp[col] += Nds[j]; 
            //printf("temp: %f\n", temp[col]);
            res[row*width+j] = temp[col];
            #pragma unroll 1
            for(int k = 0; k < width; k++)
            {
                //printf("2");
                res[row*width+j] += in1[j] * in1[k];
                //printf("res: %f\n", res[row*width+col]);
            }
        }
    }
    
}

/**
 * @brief loop completely unroll 
 * @param res result variable on GPU
 */
__global__ void sequent_complete_unroll(float* res, float* input2, int width){
    float temp[NUM_ELEMENT];
    __shared__ float Nds[NUM_ELEMENT];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;

    // loading input2 into shared memory
    Nds[ty*width+tx] = input2[row*width+col];
    __syncthreads();  

    // computing
    if(row < width && col < width)
    {
        temp[col] = 0.0;
        #pragma unroll 
        for(int j = 0; j < width; j++)
        {
            temp[col] += Nds[j]; 
            //printf("temp: %f\n", temp[col]);
            res[row*width+j] = temp[col];
            #pragma unroll 
            for(int k = 0; k < width; k++)
            {
                //printf("2");
                res[row*width+j] += in1[j] * in1[k];
                //printf("res: %f\n", res[row*width+col]);
            }
        }
    }
}

// /**
//  * @brief loop unroll 4 times
//  * @param res result variable on GPU
//  */
__global__ void sequent_unroll_sixteen(float* res, float* input2, int width){
    float temp[NUM_ELEMENT];
    __shared__ float Nds[NUM_ELEMENT];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;

    // loading input2 into shared memory
    Nds[ty*width+tx] = input2[row*width+col];
    __syncthreads();  

    // computing
    if(row < width && col < width)
    {
        temp[col] = 0.0;
        #pragma unroll 4
        for(int j = 0; j < width; j++)
        {
            temp[col] += Nds[j]; 
            //printf("temp: %f\n", temp[col]);
            res[row*width+j] = temp[col];
            #pragma unroll 4
            for(int k = 0; k < width; k++)
            {
                //printf("2");
                res[row*width+j] += in1[j] * in1[k];
                //printf("res: %f\n", res[row*width+col]);
            }
        }
    }
}
