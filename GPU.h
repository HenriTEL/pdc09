#ifndef GPU_H
#define GPU_H

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "TrainingSet.h"
#include "StrucClassSSF.h"

const std::string err_alloc("could not alloc data in GPU");
const std::string err_cpy("could not transfert data to GPU");

void check_cuda( cudaError_t ok, std::string message );

__device__
float gpuGetValueIntegral (float *gpuFeaturesIntegral, uint8_t channel, 
	int16_t x1, int16_t y1, int16_t x2, int16_t y2, int16_t w, int16_t h);

void preKernel(float *features, float *features_integral,
	float **_gpuFeatures, float **_gpuFeaturesIntegral, unsigned int **_gpuResult,
	int16_t w, int16_t h, int16_t w_integral, int16_t h_integral, int16_t noChannels, 
	int numLabels, int16_t numTries, vision::StrucClassSSF<float> *forest, vision::StrucClassSSF<float> **_gpuForest);

void postKernel(float *_gpuFeatures, float *_gpuFeaturesIntegral, unsigned int *_gpuResult,
	unsigned int *result, int16_t w, int16_t h, int numLabels);
	

#endif
