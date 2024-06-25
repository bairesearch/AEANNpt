"""AEANNpt_AEANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
AEANNpt AEANN globalDefs

"""

#debug parameters:
debugSmallNetwork = False

useAutoencoder = True	#only disable for debug

autoencoderPrediction = "previousLayer"	#autoencoder (backwards connections) predicts previous layer	#orig AEANNtf/AEANNpt implementation
#autoencoderPrediction = "inputLayer" 	#autoencoder (backwards connections) predicts input layer 	#orig AEORtf autoencoder_simulation2 implementation
supportSkipLayers = True #fully connected skip layer network
if(supportSkipLayers):
	autoencoderPrediction = "allPreviousLayers"		#optional	#orig AEANNtf implementation
	pass
	
trainLocal = True
activationFunctionTypeForward = "relu"
activationFunctionTypeBackward = "sigmoid"	#CHECKTHSI #use sigmoid for consistency with AEANNtf


#loss function paramters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer paramters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False


#network hierarchy parameters: 
#override ANNpt_globalDefs default model parameters;
batchSize = 64
numberOfLayers = 4
hiddenLayerSize = 10

#CONSIDER: disable bias;

workingDrive = '/large/source/ANNpython/AEANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelAEANN'

