#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <fstream>
#include <time.h>
#define N 128


__constant__ float input1_const[N];

__global__ void loopUnrolling(float* g_input2, float* g_result, float* g_temp, int mode)
{
    __shared__ float N_ds[N];
	

	// Get global thread ID
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int g_idy = blockIdx.y * blockDim.y + threadIdx.y;
	int g_id = g_idy * N + g_idx;


	int tile_idx = threadIdx.x; 
	int tile_idy = threadIdx.y;
    

 

    if (g_idx >= N || g_idy >= N) {
		return;
	}

    N_ds[tile_idy*N + tile_idx] = g_input2[g_id];

    //unrolling
    if(mode == 0) {
      if(g_idx ==  0) {
            for(size_t j = 0; j < N; j++) {
                g_temp[g_idy] += N_ds[j];
                g_result[g_idy*N + j] = g_temp[g_idy];
                for(size_t k = 0; k < N; ++k) {
                    g_result[g_idy*N+j] += input1_const[j] * input1_const[k];
                }
                    
            }  
        }
    } //factor = 2
    else if(mode == 2) {
        if(g_idx ==  0 && (g_idy%mode == 0 || g_idy%mode == 1)) {
            for(size_t j = 0; j < N; j++) {
                g_temp[g_idy] += N_ds[j];
               
                g_result[g_idy*N + j] = g_temp[g_idy];
               
                for(size_t k = 0; k < N; ++k) {
                    g_result[g_idy*N+j] += input1_const[j] * input1_const[k];
                }
            }  
        }     
    }
    //factor = 4
    else if(mode == 4) {
        if(g_idx ==  0 && (g_idy%mode == 0 ||g_idy%mode == 1 || g_idy%mode == 2 || g_idy%mode == 3)) {
            for(size_t j = 0; j < N; j++) {
                g_temp[g_idy] += N_ds[j];
               
                g_result[g_idy*N + j] = g_temp[g_idy];
               
                for(size_t k = 0; k < N; ++k) {
                    g_result[g_idy*N+j] += input1_const[j] * input1_const[k];
                }
            }  
        }
    }
     
}



float* initData(int dimension, int flag) {
    int size = N;
    if (dimension == 2)
        size = N * N;
    float* res = (float*)malloc(size*sizeof(float));
    if (flag == 1) {
        for(unsigned int i = 0; i < size; i++) {
            res[i] = ((float)rand()/(float)RAND_MAX) * 10;
        }
    }
    else {
        for(unsigned int i = 0; i < size; i++) {
            res[i] = 0.0;
        }
    }
    return res;

}

void cpuLoop(float* input1, float* input2, float* result, float* temp) {
    for (unsigned int i = 0; i < N; ++i) {
        temp[i] = 0.0;
        
        for(unsigned int j = 0; j < N; ++j) {
            temp[i] += input2[i*N + j];
            result[i*N + j] = temp[i];
            for (unsigned int k = 0; k < N; ++k) {
                result[i*N + j] += input1[j] * input1[k];
            }
        }
    }
}

void compareCPUAndCUDAResult (float* res_cpu, float* res_cuda) {
    bool res = true;
	for(unsigned int i=0;i<N;i++) {
		for(unsigned int j=0; j<N;j++) {
			if(abs(res_cpu[i*N + j] -res_cuda[i*N+j]) > 0.001) {
				res = false;
                //debug
                printf("(%d,%d), %f/%f", i, j,res_cpu[i*N + j],res_cuda[i*N+j] );
				break;
			}
		}
        if (res == false)
            break;
	}
	
	res ? printf("Test PASSED!\n"): printf("Test FAILED!\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    // M * N on the device
    //CPU results
    float* h_input1 = initData(1, 1);
 
    float* h_input2 = initData(2, 1);

    float* result_cpu= initData(2, 0);

    float* h_temp = initData(1, 0);


    // CPU result
    cpuLoop(h_input1, h_input2, result_cpu, h_temp);
    printf("CPU compuation Done!!\n");

    memset(h_temp,0.0,sizeof(float)*N);
   
    // Start to loop unrolling by CUDA
    int mode = atoi(argv[1]);

    // ++++++++++++++++Data transfer++++++++++++++++++++++//
     // input1 data
    size_t bytes = sizeof(float)*N;
    cudaMemcpyToSymbol(input1_const, h_input1, bytes);
    cudaMemcpy(h_input1, input1_const, bytes, cudaMemcpyDeviceToHost);
     // temp data
    float* d_temp;
    cudaMalloc(&d_temp, bytes);
    cudaMemcpy(d_temp, h_temp, bytes, cudaMemcpyHostToDevice);

    // input2 data
    size_t bytes2 = sizeof(float)*N*N;
    float* d_input2;
    cudaMalloc(&d_input2, bytes2);
    cudaMemcpy(d_input2, h_input2, bytes2, cudaMemcpyHostToDevice);

    // restlt data 
    float* d_result;
    cudaMalloc(&d_result, bytes2);
  
    // ++++++++++++++++Block and grid setting++++++++++++++++++++++//
    dim3 blockSize(N, 1);
    dim3 gridSize((int)ceil((float)N/ blockSize.x), (int)ceil((float)N / blockSize.y));

    // Launch the device computation threads!
    cudaEvent_t start, end;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    loopUnrolling<<<gridSize, blockSize>>>(d_input2, d_result, d_temp, mode);
    cudaDeviceSynchronize();
            
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("Unrolling loop with factor = %d cost %f ms\n", mode, time);
    float* result_cuda = (float*) malloc( bytes2);
    cudaMemcpy(result_cuda, d_result, bytes2, cudaMemcpyDeviceToHost);
    
    compareCPUAndCUDAResult(result_cpu, result_cuda);
    
    cudaFree(d_input2);
    cudaFree(d_result);
    cudaFree(d_temp);


    printf("CUDA compuation Done!!\n");

    /* //debug
    for (int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            printf("CPU: %f, CUDA: %f ", result_cpu[i*N + j], result_cuda[i*N+j]);
        }
        printf("\n");
    }*/
    free(h_input1);
    free(h_input2);
    free(result_cpu);
    free(h_temp);
    free(result_cuda);
    return 0;
}
#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
