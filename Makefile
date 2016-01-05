CC = g++
LD = g++

WARNGCC= -Wno-sign-compare -Wno-reorder -Wno-unknown-pragmas -Wno-overloaded-virtual

# --- With optimisation
CPPFLAGS = -fopenmp -std=c++11 -DNDEBUG -O3 -msse2 -Wall $(WARNGCC)
LDFLAGS = -DNEBUG -O3 -msse2

# --- Debugging
#CPPFLAGS = -std=c++0x -g -Wall $(WARNGCC) 
#LDFLAGS = 


INCLUDE_DIR =
LIB_DIR =-L /usr/lib/gcc/x86_64-redhat-linux/4.8.3 /usr/lib/gcc/x86_64-redhat-linux/4.8.3/libgomp.a -L /usr/lib64 /usr/lib64/libpthread.so
LIBS = `pkg-config --libs opencv`

simple:	sf1_cpu lab2rgb


testcpu:
	./sf1_cpu simple-data/config.txt 6 simple-data/tree
	rm simple-data/features/*

%.o: %.cpp 
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

main_test_simple.o: main_test_simple.cpp
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

sf1_cpu: ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o main_test_simple.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

lab2rgb: lab2rgb.o label.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

clean:
	rm -f *.o sf1_cpu lab2rgb

