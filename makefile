CXX = cuda-g++
ARCHS = -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61
VXX = nvcc $(ARCHS) -ccbin=cuda-g++
CXXFLAGS = -lgsl -lgslcblas -lm -fopenmp
CXXOPTS = -march=native -mtune=native -O3
VXXFLAGS = -Xptxas="-v" -maxrregcount=64 --compiler-options "$(CXXFLAGS) $(CXXOPTS)"

build: harppi make_spline dewiggle main.cu include/mcmc.h include/bispectrum_model.h
	$(VXX) $(VXXFLAGS) -O3 -o $(HOME)/bin/bimodal_gm main.cu obj/*.o
	
harppi: source/harppi.cpp include/harppi.h
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/harppi.o source/harppi.cpp
	
make_spline: source/make_spline.cpp include/make_spline.h
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/make_spline.o source/make_spline.cpp
	
dewiggle: source/dewiggle.cpp include/dewiggle.h
	mkdir -p obj
	$(CXX) $(CXXFLAGS) $(CXXOPTS) -c -o obj/dewiggle.o source/dewiggle.cpp
	
clean:
	rm obj/harppi.o obj/make_spline.o $(HOME)/bin/bimodal_gm
