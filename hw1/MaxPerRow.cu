#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

// CUDA kernel. Each thread takes care of one element of c
__global__ void maxElement(float *Md, float *Nd, float *Pd, int width, int height)
{
    // Get our global thread ID
    int id_x = blockIdx.x*blockDim.x+threadIdx.x; // column
    int id_y = blockIdx.y*blockDim.y+threadIdx.y; // row
    
    float Pvalue = 0;
    float temp;
    // Make sure we do not go out of bounds
    if(id_y < width && id_x < height){
        for(int i = 0; i< width; ++i)
            temp = Md[id_y*width+i] + Nd[i*height+id_x];
            Pvalue = max(Pvalue, temp);
    }
    Pd[id_y*width + id_x] = Pvalue;
}

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
    int n = 100000;

    // get width and height 
    char* temp1;
    char* temp2;
    if(argc == 3){
        temp1 = argv[1];
        temp2 = argv[2];
    }
    if(argc != 3){
        printf("Please indicate kernel width and height: \n");
        return 1;
    }
    int width;
    int height;
    if(isNumber(temp1) && isNumber(temp2)){
        width = atoi(temp1);
        height = atoi(temp2);
        //check if width and height is <= 1024
        if( width > 1024 || height > 1024)
        {
            printf("Please entry a less or equal to 1024 non-negative integer for width and height: \n");
            return 1;
        }
    }
    else{
        printf("Please entry non-negative integer for width and height: \n");
        return 1;
    }
 
    
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
    for( i = 0; i < height; i++ ) {
        for(j = 0; j < width; j++)
        {
            h_a[i*width+j] = sin(i)*sin(j)+i*width;
            h_b[i*width+j] = cos(i)*cos(j)+i*width;
        }
        
    }

    // print input
    printf("top-left 5 x 5 matrix of input \n");
    for(i = 0; i < 25; i++){
        printf("(%d,%d) -> (%f, %f)\n", row, col, h_a[i], h_b[i]);
        col++;
        if(col == 5){
            ++row;
            col = 0;
        }        
    }
 
    // Copy host vectors to device
    cudaMemcpy( Md, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( Nd, h_b, bytes, cudaMemcpyHostToDevice);
 
    

    //int blockSize, gridSize;
    dim3 dimGrid(32,32);
    dim3 dimBlock(width/32, height/32);
 
    // Number of threads in each thread block
    //blockSize = 1024;
 
    // Number of thread blocks in grid
    //gridSize = (int)ceil((float)n/dimBlock);

    // Execute the kernel:
    maxElement<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width, height);

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
  