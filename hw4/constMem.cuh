#include <cuda_runtime.h>

#define NUM_ELEMENT 128
#define CHUCK_SIZE 128
#define TILE_SIZE 128
__constant__ float in1[CHUCK_SIZE];

