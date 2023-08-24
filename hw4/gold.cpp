#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "constMem.cuh"

void computeGold(float* result, float* input1, float* input2, float* temp);

/**
 * @brief 
 * 
 * @param result reference result res[128][128]
 * @param input1 input1[128]
 * @param input2 input2[128][128]
 * @param temp temp[128]
 */
void computeGold(float* result, float* input1, float* input2, float* temp){
    for(int i = 0; i < NUM_ELEMENT; i++)
    {
        temp[i] = 0;
        for(int j = 0; j < NUM_ELEMENT; j++)
        {
            temp[i] += input2[i,j];
            result[i,j] = temp[i];
            for( int k =0; k < NUM_ELEMENT; k++)
            {
                result[i,j] += input1[j] * input1[k];
            }
        }
    }
}
