#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"


#define DEFAULT_THRESHOLD  8000
#define FILTER_WIDTH 3
#define TILE_WIDTH 32
#define HALF_FILTER_WIDTH (FILTER_WIDTH/2)
#define TILE_WIDTH_Nds (TILE_WIDTH + 2 * HALF_FILTER_WIDTH)
#define DEFAULT_FILENAME "BWstop-sign.ppm"

__constant__ int Gx[FILTER_WIDTH*FILTER_WIDTH];
__constant__ int Gy[FILTER_WIDTH*FILTER_WIDTH];

__global__ void sobel_convolution(int* res_img, const unsigned int* const input_img, const int xsize, const int ysize, const int th)
{
	int half_filter_width = FILTER_WIDTH / 2; 
	int input_tile_width = TILE_WIDTH + (2 * half_filter_width); 

	__shared__ int N_ds[TILE_WIDTH_Nds*TILE_WIDTH_Nds];
	

	// Get global thread ID
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int g_idy = blockIdx.y * blockDim.y + threadIdx.y;
	int g_id = g_idy * xsize + g_idx;

	// Apply offset = filter_width/2 to (x,y) in tile block to fill data from input image
	int tile_idx = threadIdx.x + half_filter_width; 
	int tile_idy = threadIdx.y + half_filter_width;


	if (g_idx >= xsize || g_idy >= ysize) {
		return;
	}

	// Load image to shared memory to cooresndoing postion and starting empty space for hola
	N_ds[tile_idy*input_tile_width + tile_idx] = input_img[g_id];


	
	// Deal with corner firstly
	// Top left 
	/*if (threadIdx.x < half_filter_width && threadIdx.y < half_filter_width) {
		N_ds[(threadIdx.y)*input_tile_width + threadIdx.x] =
			(g_idy == 0  || g_idx == 0) ? 0 : input_img[g_id - half_filter_width - xsize];
	}

	// Top right 
	if (threadIdx.y < half_filter_width && threadIdx.x + half_filter_width >= blockDim.x) {
		N_ds[(threadIdx.y)*input_tile_width + tile_idx + half_filter_width]
			= g_idy == 0 || g_idx + half_filter_width >= xsize ? 0 : input_img[g_id - xsize + half_filter_width];
	}

	// Bottom right 
	if (threadIdx.x + half_filter_width >= blockDim.x && threadIdx.y + half_filter_width >= blockDim.y) {
		N_ds[(tile_idy+ half_filter_width)*input_tile_width + tile_idx + half_filter_width]
			= g_idx + half_filter_width >= xsize || g_idy + half_filter_width >= ysize ? 0 : input_img[g_id + half_filter_width + xsize];
	}

	
	// Bottom left 
	if (threadIdx.x < half_filter_width) {
		N_ds[(tile_idy + half_filter_width)*input_tile_width + threadIdx.x]
			=  g_idx == 0 ||  g_idy + half_filter_width >= ysize ? 0 : input_img[g_id + xsize - half_filter_width];
	}

	// Start to deal with halo in four direction
	// Deal with halo in right
	if (threadIdx.x + half_filter_width >= blockDim.x) {
		N_ds[tile_idy*input_tile_width + tile_idx + half_filter_width]
			= g_idx + half_filter_width >= xsize ? 0 : input_img[g_id + half_filter_width];
	}

	// Deal with halo in top
	if (threadIdx.y < half_filter_width) {
		N_ds[(threadIdx.y)*input_tile_width + tile_idx] = g_idy == 0 ? 0 : input_img[g_id - xsize];
	}

	// Deal with halo in left
	if (threadIdx.x < half_filter_width) {
		N_ds[tile_idy * input_tile_width + threadIdx.x] = g_idx == 0 ? 0 : input_img[g_id - half_filter_width];
		
	}

	//Deal with halo in bottom
	if (threadIdx.y + half_filter_width >= blockDim.y) {
		N_ds[(tile_idy + half_filter_width)*input_tile_width + tile_idx]
			= g_idy +  half_filter_width >= ysize ? 0 : input_img[g_id + xsize];
		
	}*/

	__syncthreads();

	if (g_idx == 0 || g_idx + half_filter_width >= xsize ) {
		return;
	}

	if ( g_idy == 0 || g_idy  + half_filter_width >= ysize) {
		return;
	}

	// 1D example
	// float Pvalue = 0;
  	// for(int j = 0; j < Mask_Width; j++) {
    // 	Pvalue += N_ds[threadIdx.x + j]*M[j];
  	// }

	// 2D Convolution
	int Pvalue_x = 0, Pvalue_y = 0, cur_x = tile_idx - half_filter_width, cur_y = tile_idy - half_filter_width;

	for (int i = 0; i < FILTER_WIDTH; i++) { //row
		int row = cur_y + i;
		for (int j = 0; j < FILTER_WIDTH; j++) { // col
	
			int col = cur_x + j;
			Pvalue_x += N_ds[row*input_tile_width + col] * Gx[i*FILTER_WIDTH + j];
			Pvalue_y += N_ds[row*input_tile_width + col] * Gy[i*FILTER_WIDTH + j];
		}
	}
	
	int widow_value = Pvalue_x*Pvalue_x+ Pvalue_y*Pvalue_y;
	printf("v: %d\n",widow_value);
	if (widow_value > th) {
		res_img[g_id] = 255;
	}
	else {
		res_img[g_id] = 0;
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


void compareImages(int* res_cpu, int* res_cuda, int xsize, int ysize)
{
	bool res = true;
	for(int i=0;i<ysize;i++)
	{
		for(int j=0; j<xsize;j++)
		{
			if(res_cpu[i*xsize + j] !=res_cuda[i*xsize+j])
			{
				printf("Mismatch at pixel (%d, %d) and results of CPU and CUDA = %d/%d\n", i, j, res_cpu[i*xsize + j], res_cuda[i*xsize + j]);
				res = false;
			}
		}
	}
	
	res ? printf("Test PASSED!\n"): printf("Test FAILED!\n");
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
	
	printf("Start to compute convolution with CPU version\n");

	printf("width: %d, height: %d\n", xsize, ysize);

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
	//Gx = {	
	// 	-1,  0,  1,
	// 	-2,  0,  2,
	// 	-1,  0,  1 
	// };

	//GY= { 
	// 	1, 2, 1,
	// 	0,  0,  0,
	// 	-1,  -2,  -1
	// };

	for (i = 1;  i < ysize - 1; i++) {
		for (j = 1; j < xsize -1; j++) {
      
			int offset = i*xsize + j;
		//横向新值 = (-1)*[左上] + (-2)*[左] + (-1)*[左下] + 1*[右上] + 2*[右] + 1*[右下]
			sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
			+ 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
			+     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];

      //纵向新值 = (1)*[左上] + (2)*[上] + (1)*[右上] + (-1)*[左下] + (-2)*[下] + (-1)*[右下]
			sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
				- pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];
      
			magnitude =  sum1*sum1 + sum2*sum2;

			if (magnitude > thresh)
				result[offset] = 255;
			else 
				result[offset] = 0;
		}
	}

	write_ppm( (char *) "result_cpu.ppm", xsize, ysize, 255, result);
	printf("Save the result of CPU version\n");

	printf("Start to compute convolution with CUDA version\n");
	int h_Gx[FILTER_WIDTH*FILTER_WIDTH] =
	{	
		-1,  0,  1,
		-2,  0,  2,
		-1,  0,  1 
	};
	int h_Gy[FILTER_WIDTH*FILTER_WIDTH] =
	{ 
		1, 2, 1,
		0,  0,  0,
		-1,  -2,  -1
	};

	// Copy host memory to constant memory
	cudaMemcpyToSymbol(Gx, h_Gx, FILTER_WIDTH*FILTER_WIDTH * sizeof(int));
	cudaMemcpyToSymbol(Gy, h_Gy, FILTER_WIDTH *FILTER_WIDTH * sizeof(int));

	unsigned int *d_pic;
	int byes = xsize * ysize * sizeof(int);
	
	int *d_result;
	int byes_image = xsize*ysize * sizeof(unsigned int) ;
	cudaMalloc(&d_pic, byes_image);
	cudaMalloc(&d_result, byes);

	// Copy original image from cpu to gpu
	cudaMemcpy(d_pic, pic, xsize*ysize * sizeof(unsigned int), cudaMemcpyHostToDevice);

	dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
	dim3 gridSize((int)ceil((float)xsize / blockSize.x), (int)ceil((float)ysize / blockSize.y), 1);
	sobel_convolution<<<gridSize, blockSize>>>(d_result, d_pic, xsize, ysize, thresh);	

	int* h_result = (int*)malloc(numbytes);
	cudaMemcpy(h_result, d_result, byes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();	
	printf("Sobel convolution with CUDA Done!!\n");


	// Check cpu version and cuda version
	//compareImages(result, h_result, xsize, ysize);
	
	// Save result with cuda version
	write_ppm("result_cuda.ppm", xsize, ysize, 255, h_result);

	// Free memory in gpu
	cudaFree(d_pic);
	cudaFree(d_result);
	free(pic);
	free(result);
	free(h_result);
	fprintf(stderr, "sobel done\n"); 

}

