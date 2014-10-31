run: raycast
	./raycast
	
raycast : bmp.cpp raycast.cu
	nvcc bmp.cpp raycast.cu -lcudart -lm -o raycast -arch=sm_20
