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

#autoencoder/breakaway architecture parameters:
useAutoencoder = False	#optional	#condition training of hidden layers on directly connected input (stacked autoencoder algorithm)
useBreakaway = True	#optional	#condition training of hidden layers on directly connected output (stacked breakaway algorithm)
AEANNtrainGreedy = True	#train weights for hidden layers separately (greedy training algorithm) #required for custom bio plausible approximation of multilayer backprop
if(useAutoencoder or useBreakaway or AEANNtrainGreedy):
	trainLocal = True	#mandatory	#execute training at each layer (AEANNpt_AEANN training code), do not execute training at final layer only (ANNpt_main training code)
else:
	trainLocal = False	#optional	#disable for debug/benchmark against standard full layer backprop
supportSkipLayers = True #optional	#fully connected skip layer network

#dataset parameters:
useImageDataset = False	#use CIFAR-10 dataset with CNN 
if(useImageDataset):
	useTabularDataset = False
	useCNNlayers = True		#mandatory:True
else:
	useTabularDataset = True
	useCNNlayers = False	 #default:False	#optional	#enforce different connection sparsity across layers to learn unique features with greedy training	#use 2D CNN instead of linear layers

#CNN parameters:
if(useCNNlayers):
	if(useImageDataset):
		#create CNN architecture, where network size converges by a factor of ~4 (or 2*2) per layer and number of channels increases by the same factor
		CNNkernelSize = 2
		CNNstride = 2
		CNNpadding = 0	#"same"
		useCNNlayers2D = True
		CNNkernelSizeTotal = CNNkernelSize*CNNkernelSize					
		CNNmaxInputPadding = True	#pad input with zeros such that CNN is applied to every layer
		debugCNN = False
	else:
		#create CNN architecture, where network size converges by a factor of ~2 (or 2*2 if useCNNlayers2D) per layer and number of channels increases by the same factor
		CNNkernelSize = 2
		CNNstride = CNNkernelSize
		CNNpadding = 0	#"same"
		useCNNlayers2D = False
		if(useCNNlayers2D):
			CNNkernelSizeTotal = CNNkernelSize*CNNkernelSize
		else:
			CNNkernelSizeTotal = CNNkernelSize
		CNNmaxInputPadding = True	#pad input with zeros such that CNN is applied to every layer
		debugCNN = False
	
#skip layer parameters:
autoencoderPrediction = "previousLayer"	#autoencoder (backwards connections) predicts previous layer	#orig AEANNtf/AEANNpt implementation
#autoencoderPrediction = "inputLayer" 	#autoencoder (backwards connections) predicts input layer 	#orig AEORtf autoencoder_simulation2 implementation
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

#training update implementation parameters:
if(supportSkipLayers):
	trainingUpdateImplementation = "backprop"	#supportSkipLayers [autoencoderPrediction = "allPreviousLayers"] does not yet support hebbian
else:
	symmetricalAEsubnetIOlayers = False	#initialise (dependent var);
	if(autoencoderPrediction=="previousLayer" and not supportSkipLayers):
		symmetricalAEsubnetIOlayers = True
	if(autoencoderPrediction=="allPreviousLayers" and supportSkipLayers and supportSkipLayersF):	#implied: and not supportSkipLayersB:
		symmetricalAEsubnetIOlayers = True
	if(symmetricalAEsubnetIOlayers):
		#trainingUpdateImplementation = "hebbian"	#custom bio plausible approximation of single hidden layer autoencoder/breakaway backprop		#experimental	#incomplete
		trainingUpdateImplementation = "backprop"
	else:
		trainingUpdateImplementation = "backprop"	# single hidden layer AEANN backprop
trainingUpdateImplementationBackpropAuto = True	#use auto gradient backprop (vs manual gradient calculations)
if(not trainingUpdateImplementationBackpropAuto):
	learningRate = 0.0001	#currently requires lower learning rate

#activation function parameters:
activationFunctionTypeForward = "relu"
activationFunctionTypeBackward = "relu"	#default: relu	#orig: "sigmoid" (used sigmoid for consistency with AEANNtf)	#for useAutoencoder

#loss function parameters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer parameters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

#training epoch parameters:
trainNumberOfEpochsHigh = False	#use ~4x more epochs to train

#data storage parameters:
workingDrive = '/large/source/ANNpython/AEANNpt/'
dataDrive = workingDrive	#'/datasets/'
modelName = 'modelAEANN'
