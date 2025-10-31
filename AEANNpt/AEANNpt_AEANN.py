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
AEANNpt_AEANN Autoencoder/Breakaway generated artificial neural network

"""

from ANNpt_globalDefs import *
from torchsummary import summary
import AEANNpt_AEANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	
	if(printAEANNmodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t datasetType = ", datasetType)
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetRepeatSize = ", datasetRepeatSize)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t numberOfLayers = ", numberOfLayers)
		print("\t hiddenLayerSize = ", hiddenLayerSize)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t ---")
		print("\t useAutoencoder = ", useAutoencoder)
		print("\t useBreakaway = ", useBreakaway)
		print("\t AEANNtrainGreedy = ", AEANNtrainGreedy)
		print("\t trainLocal = ", trainLocal)	#False typically indicates standard full backprop ANN training
		print("\t ---")
		print("\t supportSkipLayers = ", supportSkipLayers)
		print("\t supportSkipLayersResidual = ", supportSkipLayersResidual)
		print("\t autoencoderPrediction = ", autoencoderPrediction)
		print("\t supportSkipLayersF = ", supportSkipLayersF)
		print("\t supportSkipLayersB = ", supportSkipLayersB)
		print("\t ---")
		print("\t trainingUpdateImplementation = ", trainingUpdateImplementation)
		print("\t ---")
		print("\t useImageDataset = ", useImageDataset)
		if(useImageDataset):
			print("\t useCNNlayers = ", useCNNlayers)
			print("\t numberOfConvlayers = ", numberOfConvlayers)
			print("\t numberOfFFLayers = ", numberOfFFLayers)
		
	config = AEANNpt_AEANNmodel.AEANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		numberOfConvlayers = numberOfConvlayers,
		hiddenLayerSize = hiddenLayerSize,
		CNNhiddenLayerSize = CNNhiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = AEANNpt_AEANNmodel.AEANNmodel(config)
	
	print(model)
	#summary(model, input_size=(3, 32, 32))  # adjust input_size as needed

	return model
	
