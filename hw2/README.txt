Compilation example:
nvcc -I . matrixmul.cu file_io.cpp matrixmul_gold.cpp

nvcc -I . -arch=sm_37 -gencode=arch=compute_37,code=sm_37 MatMul/matrixmul.cu file_io.cpp MatMul/matrixmul_gold.cpp MatMul/matrixmul_kernel.cu -o matmul

Colab: 

Crashes can cause the drive to dismount. 

Force-stopping a cell can cause the drive to dismount. 

When in doubt just restart the instance, saves a lot of headaches and a lot of time.

