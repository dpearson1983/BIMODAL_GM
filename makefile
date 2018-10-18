CXX = cuda-g++
ARCHS = sm_52
VXX = nvcc -arch=$(ARCHS) -ccbin=cuda-g++
CXXFLAGS = -lgsl -lgslcblas -lm -fopenmp
CXXOPTS = -march=native -mtune=native -O3
VXXFLAGS = -Xptxas="-v" -maxrregcount=64 --compiler-options "$(CXXFLAGS) $(CXXOPTS)"

build: harppi make_spline dewiggle main.cu
	$(VXX) $(VXXFLAGS) -O3 -o $(HOME)/bin/bimodal_gm main.cu obj/*.o
	
harppi: source/harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/harppi.o source/harppi.cpp
	
make_spline: source/make_spline.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/make_spline.o source/make_spline.cpp
	
dewiggle: source/dewiggle.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/dewiggle.o source/dewiggle.cpp
	
clean:
	rm obj/harppi.o obj/make_spline.o $(HOME)/bin/bimodal_gm
