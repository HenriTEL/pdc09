	// Quelques headers ...

	// Transfer the decision trees to the GPU.
	// Called once for the whole program
	void send_forest(
  		uint8_t  *cpuU8, int8_t  *cpuI8, int16_t  *cpuI16,	uint32_t  *cpuU32, float  *cpuF,
  		uint8_t **gpuU8, int8_t **gpuI8, int16_t **gpuI16,	uint32_t **gpuU32, float **gpuF,
  		int nodeCount, int gpuxLDim, int gpuyLDim);


  	void gpuPredictFullImageSingleTree (float *_gpuFeatures, float *_gpuFeaturesIntegral, 
		unsigned int *_gpuResult, 
		int16_t w, int16_t h, int16_t w_integral, int16_t h_integral, int16_t noChannels, 
		uint8_t *gpuU8, int8_t *gpuI8, int16_t *gpuI16,	uint32_t *gpuU32, float *gpuF,
		int nodeCount, int gpuxLDim, int gpuyLDim, int numLabels);
	

	
	getFlattenedFeatures(uint16_t imageId, FeatureType **out_features, int *out_nbChannels)


    // A Fragment of code to be put into the main file





