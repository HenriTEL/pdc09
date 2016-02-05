INCLUDE_DIR =
LIB_DIR =
LIBS = `pkg-config --libs opencv`

############################################################
# GPU
CC = /usr/local/cuda-7.5/bin/nvcc
LD = /usr/local/cuda-7.5/bin/nvcc
CPPFLAGS=-DGPU $(CCONFIG)
CPPFLAGS+=`pkg-config --cflags opencv`
LDFLAGS=-DGPU $(LCONFIG) `pkg-config opencv --libs` -lstdc++

simple:	sf1_gpu lab2rgb

testgpu: sf1_gpu
	./sf1_gpu simple-data/config.txt 6 simple-data/tree

sf1_gpu: GPU.o ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o main_test_simple.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

GPU.o: GPU.cu
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@
############################################################
# CPU
#CC = g++
#LD = g++
#WARNGCC= -Wno-sign-compare -Wno-reorder -Wno-unknown-pragmas -Wno-overloaded-virtual
#CPPFLAGS = -openmp -std=c++0x -DNDEBUG -O3 -msse2 -Wall $(WARNGCC)
#LDFLAGS = -DNDEBUG -O3 -msse2

## --- Debugging
#CPPFLAGS = -std=c++0x -g -Wall $(WARNGCC) 
#LDFLAGS = 

#simple:	sf1_cpu lab2rgb

#testcpu:
#	./sf1_cpu simple-data/config.txt 6 simple-data/tree

#sf1_cpu: ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o main_test_simple.o
#	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

############################################################
# GLOBAL
%.o: %.cpp 
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

main_test_simple.o: main_test_simple.cpp
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

lab2rgb: lab2rgb.o label.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

clean:
	rm -f *.o sf1_cpu lab2rgb

