#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"


#define DEFAULT_THRESHOLD  8000

#define DEFAULT_FILENAME "BWstop-sign.ppm"

#define MASK_LENGTH 3
#define MASK_OFFSET (MASK_LENGTH/2)
#define TILE_WIDTH 16


__constant__ int D_maskX[MASK_LENGTH][MASK_LENGTH];
__constant__ int D_maskY[MASK_LENGTH][MASK_LENGTH];
__global__ void sobelFilter(int* N, int* P, int width, int height);
void verify_result(int *h_res, int *result, int width, int height);

const int Gx[3][3] = {{-1,0,1}, 
					  {-2,0,2},
					  {-1,0,1}};
const int Gy[3][3] = {{-1,-2,-1}, 
					  {0,0,0},
					  {1,2,1}};

/**
 * @brief 
 * 
 * @param mask for mask 
 * @param a indicate gx or gy ------  1 for gx; 2 for gy
 */
void initMask(int *mask, int a){

	switch (a)
	{
	case 1:
		for(int i = 0; i < MASK_LENGTH; i++)
		{
			for (int j = 0; j < MASK_LENGTH; j++)
			{
				mask[i*MASK_LENGTH+j] = Gx[i][j];
			}
			//printf("mask x: %d\n", Gx[i][0]);
		}
		break;

	case 2:
		for(int i = 0; i < MASK_LENGTH; i++)
		{
			for (int j = 0; j < MASK_LENGTH; j++)
			{
				mask[i*MASK_LENGTH+j] = Gy[i][j];
				//printf("mask y: %d\n", mask[i*MASK_LENGTH+j]);
			}
		}
		break;
	}

}

unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){
  
	if ( !filename || filename[0] == '\0') {
		fprintf(stderr, "read_ppm but no file name\n");
		return NULL;  // fail
	}

	FILE *fp;

	fprintf(stderr, "read_ppm( %s )\n", filename);
	fp = fopen( filename, "rb");
	if (!fp) 
	{
		fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
		return NULL; // fail 
	}

	char chars[1024];
	//int num = read(fd, chars, 1000);
	int num = fread(chars, sizeof(char), 1000, fp);

	if (chars[0] != 'P' || chars[1] != '6') 
	{
		fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
		return NULL;
	}

	unsigned int width, height, maxvalue;


	char *ptr = chars+3; // P 6 newline
	if (*ptr == '#') // comment line! 
	{
		ptr = 1 + strstr(ptr, "\n");
	}

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
	*xsize = width;
	*ysize = height;
	*maxval = maxvalue;
  
	unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
	if (!pic) {
		fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
		return NULL; // fail but return
	}

	// allocate buffer to read the rest of the file into
	int bufsize =  3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) bufsize *= 2;
	unsigned char *buf = (unsigned char *)malloc( bufsize );
	if (!buf) {
		fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
		return NULL; // fail but return
	}

    fseek(fp, -bufsize, SEEK_END);
	long numread = fread(buf, sizeof(char), bufsize, fp);
	fprintf(stderr, "Texture %s   read %ld of %d bytes\n", filename, numread, bufsize); 

	fclose(fp);
	
	int pixels = (*xsize) * (*ysize);
	for (int i=0; i<pixels; i++) 
		pic[i] = (int) buf[3*i];  // red channel
	
	return pic; // success
}



void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
	FILE *fp;
	  
	fp = fopen(filename, "wb");
	if (!fp) 
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n", filename);
		exit(-1); 
	}
  
	fprintf(fp, "P6\n"); 
	fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
	int numpix = xsize * ysize;
	for (int i=0; i<numpix; i++) {
		unsigned char uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc); 
	}

	fclose(fp);
}


int main( int argc, char **argv )
{
	int thresh = DEFAULT_THRESHOLD;
	char *filename;
	filename = strdup( DEFAULT_FILENAME);
  
	if (argc > 1) {
		if (argc == 3)  { // filename AND threshold
			filename = strdup( argv[1]);
			thresh = atoi( argv[2] );
		}
		if (argc == 2) { // default file but specified threshhold
			thresh = atoi( argv[1] );
		}
		fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
	}

	int xsize, ysize, maxval;
	unsigned int *pic = read_ppm( filename, &xsize, &ysize, &maxval ); 
	
	int numbytes =  xsize * ysize * 3 * sizeof( int );
	int *result = (int *) malloc( numbytes );
	if (!result) { 
		fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
		exit(-1); // fail
	}

	int i, j, magnitude, sum1, sum2; 
	
	for (int col=0; col<xsize; col++) {
		for (int row=0; row<ysize; row++) { 
			*result++ = 0; 
		}
	}

	for (i = 1;  i < ysize - 1; i++) {
		for (j = 1; j < xsize -1; j++) {
      
			int offset = i*xsize + j;

			sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
			+ 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
			+     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];
      
			sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
				- pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];
      
			magnitude =  sum1*sum1 + sum2*sum2;

			if (magnitude > thresh)
				result[offset] = 255;
			else 
				result[offset] = 0;
		}
	}

	write_ppm( (char *) "result8000gold.ppm", xsize, ysize, 255, result);


	//device input
	int* D_n;
	int* D_res;

	// Allocate memory for each vector on host
	// output tile size: blockDim.x - MASK_WIDTH + 1
	size_t bytes = xsize*ysize*sizeof(int);
	int* h_res = (int*) malloc(bytes);
	int mask_size = MASK_LENGTH*MASK_LENGTH*sizeof(int); 
	
	// initialize Matrix mask and result on Device
	int* h_maskX = new int[MASK_LENGTH*MASK_LENGTH];
	initMask(h_maskX, 1);
	int* h_maskY = new int[MASK_LENGTH*MASK_LENGTH];
	initMask(h_maskY, 2);

    // Allocate memory for each vector on GPU
    cudaMalloc(&D_n, bytes);
	cudaMalloc(&D_res, bytes);

	// Copy host vectors to device 
    cudaMemcpy( D_n, pic, bytes, cudaMemcpyHostToDevice);

	// copy the mask directly to the symbol
	cudaMemcpyToSymbol(D_maskX, h_maskX, mask_size);
	cudaMemcpyToSymbol(D_maskY, h_maskY, mask_size);

	int gridSizeX = (int) ceil((float)xsize/TILE_WIDTH);
	int gridSizeY = (int) ceil((float)ysize/TILE_WIDTH);
	//#threads per block 
    dim3 dimBlock(TILE_WIDTH+MASK_LENGTH-1,TILE_WIDTH+MASK_LENGTH-1);
	// # blocks per grid
    dim3 dimGrid(gridSizeX, gridSizeY,1);
	printf("sobel kernel \n");
	sobelFilter<<<dimGrid, dimBlock>>>(D_n, D_res, xsize, ysize);

	// copy result from device to host
	cudaMemcpy(h_res, D_res, bytes, cudaMemcpyDeviceToHost);

	write_ppm( (char *) "result800Device.ppm", xsize, ysize, 255, h_res);

	fprintf(stderr, "sobel done\n"); 

	//verify_result(h_res, result, xsize, ysize);

    // TO-DO: de-allocate !!!!
	
	// Release device memory 
	cudaFree(D_n);
	cudaFree(D_res);

	// Release host memory
	free(pic);
	free(result);
	free(h_res);
	free(h_maskX);
	free(h_maskY);

	return 0;
}

__global__ void sobelFilter(int* N, int* P, int width, int height){
	// output row, column
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	// input row, column
	int row_i = row - MASK_OFFSET;
	int col_i = col - MASK_OFFSET;

	__shared__ int N_ds[TILE_WIDTH+MASK_LENGTH-1][TILE_WIDTH+MASK_LENGTH-1];

	int pValue = 0;
	float gx = 0.0;
	float gy = 0.0;

	N_ds[threadIdx.y][threadIdx.x] = N[row_i*width + col_i];

	__syncthreads();

	if(threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH)
	{
		for(int i = 0; i <MASK_LENGTH; i++)
		{
			for(int j = 0; j < MASK_LENGTH; j++)
			{
				gx += N_ds[i+threadIdx.y][j+threadIdx.x]*D_maskX[i][j];
				//printf("matrix:%d, mask:%d \n", N_ds[i+threadIdx.y][j+threadIdx.x], D_maskX[i][j]);
				gy += N_ds[i+threadIdx.y][j+threadIdx.x]*D_maskY[i][j];
			}
		}			
	}
	pValue = gx*gx + gy*gy;
	if(pValue > 8000) P[row*width+col] = 255;
	else P[row*width+col] = 0;
	
	//printf("res: %d\n",pValue);
}

/**
 * @brief 
 * 
 * @param h_res result from GPU
 * @param result result from CPU
 * @param width width
 * @param height height
 */
void verify_result(int *h_res, int *result, int width, int height) {
	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++) 
		{
			int temp = 0;
			unsigned int res_test = (abs(result[i * width + j] - h_res[i * width + j]) <= temp);
			//printf( "Test %s\n", (result[i * width + j] == h_res[i * width + j]) ? "PASSED" : "FAILED");
			printf( "device: %d  host: %d\n",  h_res[i * width + j], result[i * width + j]);
			//printf( "device: %d  host: %d\n",  h_res[i * width + j], result[i * width + j]);
		}
	}
}
