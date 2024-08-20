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
useBreakaway = True	#optional 	#also condition training of new hidden layers on directly connected output

#autoencoder architecture parameters:
autoencoderPrediction = "previousLayer"	#autoencoder (backwards connections) predicts previous layer	#orig AEANNtf/AEANNpt implementation
#autoencoderPrediction = "inputLayer" 	#autoencoder (backwards connections) predicts input layer 	#orig AEORtf autoencoder_simulation2 implementation
supportSkipLayers = False #fully connected skip layer network
if(supportSkipLayers):
	autoencoderPrediction = "allPreviousLayers"		#optional	#orig AEANNtf implementation
	supportSkipLayersF = True
	if(autoencoderPrediction == "allPreviousLayers"):
		supportSkipLayersB = False
	else:
		supportSkipLayersB = True	#optional	#add full connectivity to decoder	#not in orig AEANNtf/AEANNpt implementation
else:
	supportSkipLayersF = False
	supportSkipLayersB = False

if(useBreakaway):
	trainingUpdateImplementation = "backprop"
else:	
	#training update implementation parameters:
	symmetricalAEsubnetIOlayers = False	#initialise (dependent var);

	if(autoencoderPrediction=="previousLayer" and not supportSkipLayers):
		symmetricalAEsubnetIOlayers = True
	if(autoencoderPrediction=="allPreviousLayers" and supportSkipLayers and supportSkipLayersF):	#implied: and not supportSkipLayersB:
		symmetricalAEsubnetIOlayers = True

	if(symmetricalAEsubnetIOlayers):
		#trainingUpdateImplementation = "hebbian"	#custom bio plausible approximation of single hidden layer AEANN backprop
		trainingUpdateImplementation = "backprop"
	else:
		trainingUpdateImplementation = "backprop"	# single hidden layer AEANN backprop
	print("trainingUpdateImplementation = ", trainingUpdateImplementation)

trainLocal = True
activationFunctionTypeForward = "relu"
activationFunctionTypeBackward = "sigmoid"	#CHECKTHSI #use sigmoid for consistency with AEANNtf


#loss function parameters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer parameters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False


#network hierarchy parameters: 
#override ANNpt_globalDefs default model parameters;
batchSize = 64
numberOfLayers = 4	#default: 4
hiddenLayerSize = 10	#default: 10
trainNumberOfEpochs = 10	#default: 10


#CONSIDER: disable bias;

workingDrive = '/large/source/ANNpython/AEANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelAEANN'

