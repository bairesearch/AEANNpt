"""AEANNpt_AEANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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

#autoencoder architecture parameters:

useAutoencoder = True	#only disable for debug/benchmark
useBreakaway = True	#also condition training of new hidden layers on directly connected output
AEANNtrainGreedy = True	#only train weights for layer l1 (greedy training)

autoencoderPrediction = "previousLayer"	#autoencoder (backwards connections) predicts previous layer	#orig AEANNtf/AEANNpt implementation
#autoencoderPrediction = "inputLayer" 	#autoencoder (backwards connections) predicts input layer 	#orig AEORtf autoencoder_simulation2 implementation
supportSkipLayers = True #fully connected skip layer network
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

if(supportSkipLayers):
	trainingUpdateImplementation = "backprop"	#supportSkipLayers [autoencoderPrediction = "allPreviousLayers"] does not yet support hebbian
else:
	#training update implementation parameters:
	symmetricalAEsubnetIOlayers = False	#initialise (dependent var);

	if(autoencoderPrediction=="previousLayer" and not supportSkipLayers):
		symmetricalAEsubnetIOlayers = True
	if(autoencoderPrediction=="allPreviousLayers" and supportSkipLayers and supportSkipLayersF):	#implied: and not supportSkipLayersB:
		symmetricalAEsubnetIOlayers = True

	if(symmetricalAEsubnetIOlayers):
		#trainingUpdateImplementation = "hebbian"	#custom bio plausible approximation of single hidden layer AEANN backprop	#incomplete
		trainingUpdateImplementation = "backprop"
		if(trainingUpdateImplementation == "hebbian"):
			learningRate = 0.0001	#currently requires lower learning rate
	else:
		trainingUpdateImplementation = "backprop"	# single hidden layer AEANN backprop

trainLocal = True	#use custom AEANNpt_AEANNmodel training code, do not use general ANNpt_main training code
if(not useAutoencoder and not useBreakaway and not AEANNtrainGreedy):
	#trainLocal = False	#optional #use of general ANNpt_main training code	#only disable for debug/benchmark
	pass
	
activationFunctionTypeForward = "relu"
activationFunctionTypeBackward = "relu"	#default: relu	#orig: "sigmoid" (used sigmoid for consistency with AEANNtf)


#loss function parameters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer parameters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

if(useBreakaway):
	trainNumberOfEpochsHigh = False	#useAutoencoder+useBreakaway+AEANNtrainGreedy can require ~4x more epochs to train

#CONSIDER: disable bias;

workingDrive = '/large/source/ANNpython/AEANNpt/'
dataDrive = workingDrive	#'/datasets/'

modelName = 'modelAEANN'

