#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

// each thread producing one column of the output matrix
__global__ void rowAdd(float *Md, float *Nd, float *Pd, int width)
{
    // Get our global thread ID
    int id_x = blockIdx.x*blockDim.x+threadIdx.x; // column
    int id_y = blockIdx.y*blockDim.y+threadIdx.y; // row
    
    float Pvalue = 0;
    // Make sure we do not go out of bounds
    if(id_y < width && id_x < width){
        for(int i = 0; i < width; i++){
            Pvalue += Md[id_y+i] + Nd[i*width+id_x];
        }
    }
       
    Pd[id_y*width + id_x] = Pvalue;

}

// each thread producing one row of the output matrix
__global__ void columnAdd(float *Md, float *Nd, float *Pd, int width)
{
    // Get our global thread ID
    int id_x = blockIdx.x*blockDim.x+threadIdx.x; // column
    int id_y = blockIdx.y*blockDim.y+threadIdx.y; // row
    
    float Pvalue;
    // Make sure we do not go out of bounds
    if(id_y < width && id_x < width){
        for(int i = 0; i < width; ++i){
            Pvalue += Md[id_y*width] + Nd[i*width];
        }
            
    }
       
    Pd[id_y*width + id_x] = Pvalue;
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void matrixAdd(float *Md, float *Nd, float *Pd, int width)
{
    // Get our global thread ID
    int id_x = blockIdx.x*blockDim.x+threadIdx.x; // column
    int id_y = blockIdx.y*blockDim.y+threadIdx.y; // row
    
    float Pvalue = 0;

    // Make sure we do not go out of bounds
    if(id_y < width && id_x < width){
        for(int i = 0; i< width; ++i)
            Pvalue += Md[id_y*width+i] + Nd[i*width+id_x];
    }
    Pd[id_y*width + id_x] = Pvalue;
    
    // print the thread's 2dim grid id
    // while(id_x < 5 && id_y < 5){
    //     printf("blk: (%d,%d) Thread: (%d,%d) -> Row/Col = (%d,%d)\n",
    //         blockIdx.x, blockIdx.y,
    //         threadIdx.x, threadIdx.y,
    //         id_x, id_y);
    // }
}

// to check if get valid arguments
__host__ bool isNumber(char number[])
{
    int i = 0;
    if(number[i] == '-')
        return false;
    
    for(; number[i]; ++i){
        if(!isdigit(number[i]))
            return false;
    }
    return true;
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 100000; //n
    int width = 1024;
 
    
    // Host input vectors
    float *h_a;
    float *h_b;
    //Host output vector
    float *h_c;
 
    // Device input vectors
    float *Md;
    float *Nd;
    //Device output vector
    float *Pd;
 
    // Size, in bytes, of each vector
    size_t bytes = width*width*sizeof(float);
 
    // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&Md, bytes);
    cudaMalloc(&Nd, bytes);
    cudaMalloc(&Pd, bytes);
 
    int i;
    int j;
    int row = 0;
    int col = 0;
    // Initialize vectors on host
    for( i = 0; i < width; i++ ) {
        for(j = 0; j < width; j++)
        {
            h_a[j+i*width] = sin(i)*sin(j)+i*width;
            h_b[j+i*width] = cos(i)*cos(j)+i*width; //j+i*width;
        }
        
    }

    // print input
    printf("top-left 5 x 5 matrix of input \n");
    for(i = 0; i < 25; i++){
        printf("(h_a, h_b) -> (%f, %f)\n", h_a[i], h_b[i]);        
    }
 
    // Copy host vectors to device
    cudaMemcpy( Md, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( Nd, h_b, bytes, cudaMemcpyHostToDevice);
 
    //int blockSize, gridSize;
    dim3 dimGrid(32,32);
    dim3 dimBlock(width/32, width/32);
 
    // Number of threads in each thread block
    //blockSize = 1024;
 
    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)n/dimBlock);
 
    // Determine to execute which kernel
    char* temp; 
    if(argc == 2)
        temp = argv[1];
    if(argc < 2)
    {
        printf("Please indicate which kernel to execute by following number: \n");
        printf("1: matrix Add \n 2: row Add \n 3: columnAdd");
        return 1;
    }

    int kernel;
    if(isNumber(temp))
        kernel = atoi(temp);
    else{
        printf("Please select one of integer from 1 to 3 for kernel: \n");
        printf("1: matrix Add \n 2: row Add \n 3: columnAdd");
        return 1;
    }

    // Execute the kernel
    switch (kernel){
        case 1:
            matrixAdd<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
            break;
        case 2:
            rowAdd<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
            break;
        case 3:
            columnAdd<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
            break;
    }
 
    // Copy array back to host
    cudaMemcpy( h_c, Pd, bytes, cudaMemcpyDeviceToHost );
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    float sum = 0;
    row = 0;
    col = 0;
    printf("\n top-left 5 x 5 matrix of output \n");
    for( i = 0; i < width; i++ ) {
        sum += h_c[i]; 

        // print top left 5 x 5 matrix
        if(i < 25){
            printf("(%d,%d) -> %f\n", row, col, h_c[i]);
            col++;
            if(col == 5){
                ++row;
                col = 0;
            }
        }        
    }
    printf("final result: %f\n", sum/(float)n);
 
    // Release device memory
    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
