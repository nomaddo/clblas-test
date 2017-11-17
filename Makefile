# need clblas (via `apt-get install clblas-dev`)
# need clblast (https://github.com/CNugteren/CLBlast)

CC=gcc
FLAG=-g -O2 -lOpenCL

all: clblast clblas

clblast: clblast-tuned.c
	$(CC) -o $@ $< $(FLAG) -lclblast

clblas: clblas.c
	$(CC) -o $@ $< $(FLAG) -lclBLAS

test: clblas clblast
	./clblas
	./clblast
