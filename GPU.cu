#include <iostream>
#include <string>

/* Cuda stuff */
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "GPU.h"
#include "TrainingSet.h"
#include "StrucClassSSF.h"

using namespace vision;

void check_cuda( cudaError_t ok, std::string message )
{
	if( ok!=cudaSuccess )
		std::cerr << ">>> Error: " << message << std::endl;
}

__device__
float gpuGetValueIntegral (float *gpuFeaturesIntegral, uint8_t channel, 
	int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t w, int16_t h)
{
	float res = (
			gpuFeaturesIntegral[y2 + x2*h + channel*w*h] -
			gpuFeaturesIntegral[y2 + x1*h + channel*w*h] -
			gpuFeaturesIntegral[y1 + x2*h + channel*w*h] +
			gpuFeaturesIntegral[y1 + x1*h + channel*w*h]);

	return res;
}

/***************************************************************************
 Prepare the kernel call:
 - Transfer the features to the GPU
 - Prepare an array for the results, initialized to zero (in parallel on the GPU)
 ***************************************************************************/
void preKernel(float *features, float *features_integral,
	float **_gpuFeatures, float **_gpuFeaturesIntegral, unsigned int **_gpuResult,
	int16_t w, int16_t h, int16_t w_integral, int16_t h_integral, int16_t noChannels, 
	int numLabels, int16_t numTries, StrucClassSSF<float> *forest, StrucClassSSF<float> **_gpuForest)
{
	cudaError_t ok;
	int size;

	// Init GPU memory for the features
	size = noChannels*w*h*sizeof(float);
	ok = cudaMalloc ((void**) _gpuFeatures, size);
	check_cuda(ok, err_alloc);

	ok = cudaMemcpy (*_gpuFeatures, features, size, cudaMemcpyHostToDevice);
	check_cuda(ok, err_cpy);

	size = noChannels*w_integral*h_integral*sizeof(float);
	ok = cudaMalloc ((void**) _gpuFeaturesIntegral, size);
	check_cuda(ok, err_alloc);

	ok = cudaMemcpy (*_gpuFeaturesIntegral, features_integral, size, cudaMemcpyHostToDevice);
	check_cuda(ok, err_cpy);
	
	// Forest load
	size = 0;
	for( int i=0; i < numTries; i++ )
		size += sizeof(forest[i]) - sizeof(int*) + forest[i].getHeapSize();
	//ok = cudaMalloc ((void**) _gpuForest, size);
	//check_cuda(ok, err_alloc);
	// TODO copy heap
	//ok = cudaMemcpy (*_gpuForest, forest, size, cudaMemcpyHostToDevice);
	//check_cuda(ok, err_cpy);

	// Allocate memory for the results
	size=w*h*numLabels*sizeof(unsigned int);
	ok=cudaMalloc ((void**) _gpuResult, size);
	check_cuda(ok, err_alloc);

}


/***************************************************************************
 After the kernel call:
 - Transfer the result back from the GPU to the _CPU
 - free the GPU memory related to a single image
 ***************************************************************************/
void postKernel(float *_gpuFeatures, float *_gpuFeaturesIntegral, unsigned int *_gpuResult,
	unsigned int *result, int16_t w, int16_t h, int numLabels)
{
	cudaError_t ok;
	int size;

	// Copy the results back to host memory
	size=w*h*numLabels*sizeof(unsigned int);
	ok=cudaMemcpy (result, _gpuResult, size, cudaMemcpyDeviceToHost);
	check_cuda(ok, err_cpy);

#ifdef GPU_DEBUG_SINGLE_PIXEL
	std::cerr << "Debug-error code (int)=" << std::dec << (int) *result << "\n";
	std::cerr << "Return values: ";
	for (int i=0; i<result[0]; ++i)
		std::cerr << result[i+1] << " ";
	std::cerr << "\n";
#endif  		

	// Free GPU memory.
	cudaFree(_gpuFeatures);
	cudaFree(_gpuFeaturesIntegral);
	cudaFree(_gpuResult);
	// TODO free the forest
}
