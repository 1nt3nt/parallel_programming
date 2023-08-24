/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_


// **===----------------- MP3 - Modify this function ---------------------===**
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(float *g_data, int n)
{
    __shared__ float partialSum[1024];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    partialSum[t] =g_data[t+start];
    partialSum[blockDim.x+t] = g_data[start+blockDim.x+t];

    for(unsigned int stride = blockDim.x/2; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if(t < stride )
        {
            partialSum[t] += partialSum[t+stride];
        }
    }
    __syncthreads();
    *g_data = partialSum[0];
}

//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
//  for vector size > 512
// **===------------------------------------------------------------------===**
__global__ void reductionLarge(float *g_data, int arr_size)
{
    int temp = arr_size / 512;
    int remainder = arr_size % 512;
    int division = remainder == 0 ? temp : temp+1;

    __shared__ float partialSum[1024];
    //__shared__ float* temp;
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    
    for(int i = 0; i < division; ++i)
    {
        if(start + t < 512){
            partialSum[t] += g_data[t+start + i * 512];
        }
        if(start+t+blockDim.x < 512){
            partialSum[t+blockDim.x] += g_data[start+t+blockDim.x + i * 512];
        }
    }

    for(unsigned int stride = blockDim.x/2; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if(t < stride )
        {
            partialSum[t] += partialSum[t+stride];
        }
    }
    __syncthreads();    

    *g_data = partialSum[0];
}














/**
    @param d_data input data
    @param d_ouput output data. result would be the last element from this vec
    @param arr_size input number of elements to scan from input data
*/
__global__ void brent_kung(float* d_data, float* d_ouput, int arr_size)
{
    // Get our global thread ID, initialize variable in share memory
    __shared__ float result[512];
    int t = 2*blockIdx.x*blockDim.x+threadIdx.x;
    //each thread loads one value from the input.
    if(t < arr_size) {
        result[threadIdx.x] = d_data[t];
        printf("1");
    }
    if(t+blockDim.x < arr_size){
        result[threadIdx.x+blockDim.x] = d_data[t+blockDim.x];  
        printf("2");
    } 

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        __syncthreads();
        int index = (threadIdx.x+1) * 2 * stride -1;
        if(index < 512){
            result[index] += result[index - stride];
        }
    }

    for(int stride = 512/4; stride > 0; stride /= 2){
        __syncthreads();
        int index = (threadIdx.x+1) * 2 * stride -1;
        if(index + stride < 512)
            result[index+stride] += result[index];
    }

    __syncthreads();
    if(t < arr_size) d_ouput[t] = result[threadIdx.x];
    if(t+blockDim.x < arr_size) d_ouput[t+blockDim.x] = result[threadIdx.x+blockDim.x];
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
