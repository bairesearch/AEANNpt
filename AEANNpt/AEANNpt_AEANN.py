"""AEANNpt_AEANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
AEANNpt_AEANN Autoencoder generated artificial neural network

"""

from ANNpt_globalDefs import *
import AEANNpt_AEANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	
	print("creating new model")
	config = AEANNpt_AEANNmodel.AEANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		hiddenLayerSize = hiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = AEANNpt_AEANNmodel.AEANNmodel(config)
	return model
	
