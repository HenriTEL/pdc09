CC = g++
LD = g++

WARNGCC= -Wno-sign-compare -Wno-reorder -Wno-unknown-pragmas -Wno-overloaded-virtual

# --- With optimisation
CPPFLAGS = -openmp -std=c++0x -DNDEBUG -O3 -msse2 -Wall $(WARNGCC)
LDFLAGS = -DNEBUG -O3 -msse2

# --- Debugging
#CPPFLAGS = -std=c++0x -g -Wall $(WARNGCC) 
#LDFLAGS = 


INCLUDE_DIR =
LIB_DIR =
LIBS = `pkg-config --libs opencv`

simple:	sf1_cpu test_heap lab2rgb


testcpu:
	./sf1_cpu simple-data/config.txt 6 simple-data/tree
testheap: test_heap
	./test_heap
%.o: %.cpp 
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

main_test_simple.o: main_test_simple.cpp
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@
test_heap.o: test_heap.cpp
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

sf1_cpu: ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o main_test_simple.o utils.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

test_heap: ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o test_heap.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

lab2rgb: lab2rgb.o label.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

clean:
	rm -f *.o sf1_cpu lab2rgb

