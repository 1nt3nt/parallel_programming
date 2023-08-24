#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <ctype.h>
#include "constMem.cuh"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

//-------------------- declare function ------------------------
void init_arr(float* arr, int n, int index);
void allocateConstantMem(float* c, float* input, int n);
float* computeOnDevice(float* result, float* input1, float* input2, int N, int kernel);
bool verify_test(const float* c_res, const float* g_res, const int n);
void computeGold(float* result, float* input1, float* input2, float* temp);
__global__ void sequent_dis_unroll(float* res, float* input2, int width);
__global__ void sequent_complete_unroll(float* res, float* input2, int width);
__global__ void sequent_unroll_sixteen(float* res, float* input2, int width);
// ---------------------- End ---------------------------------

void init_arr(float* arr, int n, int index){
    // initialize input randomly
    if(index == 1)
    {
        for (int i = 0; i < n; i++)
        {
            arr[i] = (rand()*3 / (float)RAND_MAX);
        }
    }

    if( index == 2)
    {
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++)
            {
                arr[i*n+k] = (rand()*3 / (float)RAND_MAX);
            }
            
        }
    }
}

void allocateConstantMem(float* c, float* input, int n){
    for (int i = 0; i < n; i++)
    {
        c[i] = input[i];
    }
    //printf("constant memory done \n");
}

// ----------------- execute Device version --------------------------
float* computeOnDevice(float* result, float* input1, float* input2, int N, int kernel){
    // device varibale
    float* d_a; //input 2
    //float* d_b; // temp
    float *d_c; // output

    // Size, in bytes, of each vector 
    size_t bytes2 = N*N*sizeof(float);
    size_t bytes1 = CHUCK_SIZE*sizeof(float);

    // allocate device vec memory
    cudaMalloc(&d_a, bytes2);
    cudaMalloc(&d_c, bytes2);

    // // Copy host vectors to device 
    cudaMemcpy( d_a, input2, bytes2, cudaMemcpyHostToDevice);

    // blocksize, gridsize
    dim3 dimGrid(N/TILE_SIZE,N/1);
    dim3 dimBlock(TILE_SIZE,1);

    // initialize && allocate constant memory
    float* c_in1 = new float[N];
    allocateConstantMem(c_in1, input1, N);
    cudaMemcpyToSymbol(in1, c_in1, bytes1);

    // CUDA timer
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // execute kernel
    // 0: loop unrolling disable 
    // 1: loop completely unroll
    // 2: loop unroll 4 times
    if(kernel == 0) sequent_dis_unroll<<<dimGrid, dimBlock>>>(d_c, d_a, N);
    if(kernel == 1) sequent_complete_unroll<<<dimGrid, dimBlock>>>(d_c, d_a, N);
    if(kernel == 2) sequent_unroll_sixteen<<<dimGrid, dimBlock>>>(d_c, d_a, N);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("run time: %f\n", time);
    // Copy result from device memory to host memory
    cudaMemcpy(result, d_c, bytes2, cudaMemcpyDeviceToHost);

    // de allocate memory on host and device
    cudaFree(d_a);
    cudaFree(d_c);

    return result;
}

// comparing device and host result
bool verify_test(const float* c_res, const float* g_res, const int n){
    bool flag;
    for(int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            //printf("CPU: %f, GPU: %f ", c_res[i*n+j], g_res[i*n+j]);
            flag = c_res[i*n+j] == g_res[i*n+j] ? true:false;
            if(!flag) break;
        }
        //printf("\n");
        if(!flag) break;
    }
    return flag;
}

// to check if get valid arguments
__host__ bool isNumber(char number[]){
    int i = 0;
    if(number[i] == '-')
        return false;
    
    for(; number[i]; ++i){
        if(!isdigit(number[i]))
            return false;
    }
    return true;
}

int main(int argc, char** argv){
    // host input arr
    float* result; //result back to host from device
    float* input1;
    float *input2;
    float *temp;
    float* reference;

    // size in bytes of each vector
    size_t bytes1 = NUM_ELEMENT*sizeof(float);
    size_t bytes2 = NUM_ELEMENT*NUM_ELEMENT*sizeof(float);
    
    // allocate memory for each vector on host
    input1 = (float*)malloc(bytes1);
    input2 = (float*)malloc(bytes2);
    result = (float*)malloc(bytes2);
    reference = (float*)malloc(bytes2);
    temp = (float*)malloc(bytes1);

    // initialize input randomly
    init_arr(input1, NUM_ELEMENT, 1);
    init_arr(input2, NUM_ELEMENT, 2);


    // get #kernel
    // 0: no unrlling, 1: unrolling, 2:
    int kernel = 0;
    char* tmp; // indicate kernel
    if(argc == 2)
        tmp = argv[1];
    else{
        printf("Please indicate which kernel to execute by following number: \n");
        printf("0: loop unrolling disable \n 1: loop completely unroll \n 2: loop unroll 4 times");
        return 1;
    }

    // checking if input is integer
    if(isNumber(tmp))
        kernel = atoi(tmp);
    else
    {
        printf("Please indicate which kernel to execute by following number: \n");
        printf("0: loop unrolling disable \n 1: loop completely unroll \n 2: loop unroll 4 times");
        return 1;        
    }

    // calculate on device 
    computeOnDevice(result, input1, input2, NUM_ELEMENT, kernel);
    fprintf(stderr, "Device done\n");

    // calculate on host
    computeGold(reference, input1, input2, temp);
    fprintf(stderr, "Host done\n");

    // verify result from both host and device
    bool test = verify_test(reference, result, NUM_ELEMENT);
    if(test) printf("Pass\n");
    else printf("Fail\n");

    free(input1);
    free(input2);
    free(result);
    free(reference);
    free(temp);
    return 0;
}