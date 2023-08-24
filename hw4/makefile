NVCC=nvcc
EXEC=main

all: $(EXEC)
main: sequential.o
      $(NVCC) -o main sequential.o

sequential.o: main.cu sequential.cu gold.cpp
      $(NVCC) -o sequential.o main.cu sequential.cu gold.cpp	  

clean:
      rm -rf *.o

mrproper: clean
      rm -rf main