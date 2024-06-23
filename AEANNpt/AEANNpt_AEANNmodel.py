"""AEANNpt_AEANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
AEANNpt Autoencoder generated artificial neural network model

"""

import torch as pt
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers

class AEANNconfig():
	def __init__(self, batchSize, numberOfLayers, hiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.hiddenLayerSize = hiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples

class BiasLayer(nn.Module):
	def __init__(self, num_features):
		super(BiasLayer, self).__init__()
		self.bias = nn.Parameter(pt.zeros(num_features), requires_grad=True)

	def forward(self, x):
		# Adding bias to every element of the input tensor x
		return x + self.bias.unsqueeze(0)
			
class AEANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config
			
		layersLinearListF = []
		layersLinearListB = []
		layersListB = []
		layersLinearListF.append(None)	#input layer i=0
		layersLinearListB.append(None)	#input layer i=0
		layersListB.append(None)	#input layer i=0
		self.n_h = self.generateLayerSizeList()
		
		for l1 in range(1, config.numberOfLayers+1):	
			if(supportSkipLayers):
				layersLinearListF2 = []
				if(not supportSkipLayersAutoencodeLastLayerPredictionOnly):
					layersLinearListB2 = []
				for l2 in range(0, l1):
					if(l2 < l1):
						linearF2 = ANNpt_linearSublayers.generateLinearLayer(self, l2, config, forward=True, bias=False, layerIndex2=l1)	#need to add bias after skip layer connections have been added
						layersLinearListF2.append(linearF2)
						if(not supportSkipLayersAutoencodeLastLayerPredictionOnly):
							linearB2 = ANNpt_linearSublayers.generateLinearLayer(self, l2, config, forward=False, bias=True, layerIndex2=l1)	#orig AEANNtf:bias=False
							layersLinearListB2.append(linearB2)
				layersLinearF2 = nn.ModuleList(layersLinearListF2)
				layersLinearListF.append(layersLinearF2)
				if(not supportSkipLayersAutoencodeLastLayerPredictionOnly):
					layersLinearB2 = nn.ModuleList(layersLinearListB2)
					layersLinearListB.append(layersLinearB2)
				if(supportSkipLayersAutoencodeLastLayerPredictionOnly):
					linearB = ANNpt_linearSublayers.generateLinearLayer(self, self.getLayerIndex(l1), config, forward=False, bias=True)	#orig AEANNtf:bias=False
					layersLinearListB.append(linearB)
				B = BiasLayer(self.n_h[l1])	#need to add bias after skip layer connections have been added
				layersListB.append(B)
			else:
				linearF = ANNpt_linearSublayers.generateLinearLayer(self, self.getLayerIndex(l1), config, forward=True, bias=True)
				layersLinearListF.append(linearF)
				linearB = ANNpt_linearSublayers.generateLinearLayer(self, self.getLayerIndex(l1), config, forward=False, bias=True)	#orig AEANNtf:bias=False
				layersLinearListB.append(linearB)

		self.layersLinearF = nn.ModuleList(layersLinearListF)
		self.layersLinearB = nn.ModuleList(layersLinearListB)
		self.activationF = ANNpt_linearSublayers.generateActivationFunction(activationFunctionTypeForward)
		self.activationB = ANNpt_linearSublayers.generateActivationFunction(activationFunctionTypeBackward)
		if(supportSkipLayers):
			self.layersB = nn.ModuleList(layersListB)
		
		self.lossFunctionBackward = nn.MSELoss()	#lossFunctionAutoencoder
		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunctionFinal = nn.CrossEntropyLoss()
		else:
			self.lossFunctionFinal = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		if(supportSkipLayers):
			self.Ztrace = [None]*(config.numberOfLayers+1)
			self.Atrace = [None]*(config.numberOfLayers+1)
			for l1 in range(1, numberOfLayers+1):
				self.Ztrace[l1] = pt.zeros([batchSize, self.n_h[l1]], device=device)
				self.Atrace[l1] = pt.zeros([batchSize, self.n_h[l1]], device=device)

	def generateLayerSizeList(self):
		n_h = [None]*(self.config.numberOfLayers+1)
		for l1 in range(1, self.config.numberOfLayers+1):
			if(l1 == 0):
				n_h[l1] = self.config.inputLayerSize
			elif(l1 == numberOfLayers):
				n_h[l1] = self.config.outputLayerSize
			else:
				n_h[l1] = self.config.hiddenLayerSize
		return n_h
			
	def forward(self, trainOrTest, x, y, optim, layer=None):
	
		autoencoder = useAutoencoder
		
		outputPred = None
		outputTarget = None

		outputPred = x #in case layer=0

		AprevLayer = x
		if(supportSkipLayers):
			self.Atrace[0] = AprevLayer

		maxLayer = self.config.numberOfLayers
		
		for l1 in range(1, maxLayer+1):
			#print("l1 = ", l1)
			
			A, Z, outputTarget = self.neuralNetworkPropagationLayerForward(l1, AprevLayer, autoencoder)

			if(autoencoder):
				if(l1 == self.config.numberOfLayers):
					outputPred = Z	#activation function softmax is applied by self.lossFunctionFinal = nn.CrossEntropyLoss()
				else:
					outputPred = self.neuralNetworkPropagationLayerBackwardAutoencoder(l1, A)
					self.trainLayer(self.getLayerIndex(l1), outputPred, outputTarget, optim, self.lossFunctionBackward)	#first layer optimiser is defined at i=0 
			else:
				if(l1 == self.config.numberOfLayers):
					outputPred = Z	#activation function softmax is applied by self.lossFunctionFinal = nn.CrossEntropyLoss()
				else:
					outputPred = A
			if(l1 == self.config.numberOfLayers):
				outputTarget = y
				loss, accuracy = self.trainLayer(self.getLayerIndex(l1), outputPred, outputTarget, optim, self.lossFunctionFinal, calculateAccuracy=True)
			
			A = A.detach()	#only train weights for layer l1

			AprevLayer = A
			if(supportSkipLayers):
				self.Ztrace[l1] = Z
				self.Atrace[l1] = A

		return loss, accuracy

	def getLayerIndex(self, l1):
		layerIndex = l1-1
		return layerIndex
	
	def trainLayer(self, layerIndex, pred, target, optim, lossFunction, calculateAccuracy=False):
		accuracy = 0
		if(calculateAccuracy):
			accuracy = self.accuracyFunction(pred, target)
		loss = lossFunction(pred, target)
		opt = optim[layerIndex]
		opt.zero_grad()
		loss.backward()
		opt.step()
		return loss, accuracy


	def neuralNetworkPropagationLayerForward(self, l1, AprevLayer, autoencoder):

		outputTarget = None

		if(supportSkipLayers):
			if(autoencoder):
				outputTargetList = []
			Z = pt.zeros_like(self.Ztrace[l1])
			for l2 in range(0, l1):
				Zpartial = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), self.Atrace[l2], self.layersLinearF[l1][l2])
				Z = pt.add(Z, Zpartial)
				if(autoencoder):
					if(not supportSkipLayersAutoencodeLastLayerPredictionOnly):
						outputTargetListPartial = self.Atrace[l2]
						outputTargetList.append(outputTargetListPartial)
			Z = pt.add(Z, self.layersB[l1](Z))	#need to add bias after skip layer connections have been added
			if(autoencoder):
				if(not supportSkipLayersAutoencodeLastLayerPredictionOnly):
					outputTarget = pt.concat(outputTargetList, dim=1)
					#print("outputTarget.shape = ", outputTarget.shape)
				else:
					outputTarget = AprevLayer
		else:	
			if(autoencoder):
				outputTarget = AprevLayer
			Z = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), AprevLayer, self.layersLinearF[l1])
		A = ANNpt_linearSublayers.executeActivationLayer(self, self.getLayerIndex(l1), Z, self.activationF)	#relU	

		return A, Z, outputTarget

	def neuralNetworkPropagationLayerBackwardAutoencoder(self, l1, A):
		if(supportSkipLayers and not supportSkipLayersAutoencodeLastLayerPredictionOnly):
			outputPredList = []
			for l2 in range(0, l1):
				Zback = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), A, self.layersLinearB[l1][l2])
				outputPredPartial = self.activationB(Zback)
				outputPredList.append(outputPredPartial)
			outputPred = pt.concat(outputPredList, dim=1)
		else:
			Zback = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), A, self.layersLinearB[l1])
			outputPred = self.activationB(Zback)
		return outputPred
	
		


