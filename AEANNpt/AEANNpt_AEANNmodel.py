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
		if(useBreakaway):
			layersLinearListO = []
		layersListB = []
		layersLinearListF.append(None)	#input layer i=0
		layersLinearListB.append(None)	#input layer i=0
		layersListB.append(None)	#input layer i=0
		self.n_h = self.generateLayerSizeList()
		print("self.n_h = ", self.n_h)
		
		for l1 in range(1, config.numberOfLayers+1):	
			if(supportSkipLayers):
				layersLinearListF2 = []
				layersLinearListB2 = []
				for l2 in range(0, l1):
					if(supportSkipLayersF):
						linearF2 = ANNpt_linearSublayers.generateLinearLayer(self, l2, config, forward=True, bias=False, layerIndex2=l1)	#need to add bias after skip layer connections have been added
						layersLinearListF2.append(linearF2)
					if(autoencoderPrediction=="allPreviousLayers"):
						linearB2 = ANNpt_linearSublayers.generateLinearLayer(self, l2, config, forward=False, bias=True, layerIndex2=l1)	#orig AEANNtf:bias=False
						layersLinearListB2.append(linearB2)
					elif(supportSkipLayersB):
						if(autoencoderPrediction=="previousLayer"):
							l3 = self.getLayerIndex(l1)
						elif(autoencoderPrediction=="inputLayer"):
							l3 = 0
						linearB2 = ANNpt_linearSublayers.generateLinearLayer(self, l3, config, forward=False, bias=True, layerIndex2=l2+1)	#orig AEANNtf:bias=False
						layersLinearListB2.append(linearB2)
				layersLinearF2 = nn.ModuleList(layersLinearListF2)
				layersLinearListF.append(layersLinearF2)
				if(supportSkipLayersB or autoencoderPrediction=="allPreviousLayers"):
					layersLinearB2 = nn.ModuleList(layersLinearListB2)
					layersLinearListB.append(layersLinearB2)
				else:
					if(autoencoderPrediction=="previousLayer"):
						l2 = self.getLayerIndex(l1)
					elif(autoencoderPrediction=="inputLayer"):
						l2 = 0
					linearB = ANNpt_linearSublayers.generateLinearLayer(self, l2, config, forward=False, bias=True, layerIndex2=l1)	#orig AEANNtf:bias=False
					layersLinearListB.append(linearB)
				B = BiasLayer(self.n_h[l1])	#need to add bias after skip layer connections have been added
				layersListB.append(B)
			else:
				l2 = self.getLayerIndex(l1)
				linearF = ANNpt_linearSublayers.generateLinearLayer(self, l2, config, forward=True, bias=True, layerIndex2=l1)
				layersLinearListF.append(linearF)
				if(autoencoderPrediction=="previousLayer"):
					l2 = self.getLayerIndex(l1)
				elif(autoencoderPrediction=="inputLayer"):
					l2 = 0
				linearB = ANNpt_linearSublayers.generateLinearLayer(self, l2, config, forward=False, bias=True, layerIndex2=l1)	#orig AEANNtf:bias=False
				layersLinearListB.append(linearB)
			if(useBreakaway):
				l2 = self.getLayerIndex(l1)
				linearO = ANNpt_linearSublayers.generateLinearLayer2(self, l2, config.hiddenLayerSize, config.outputLayerSize, config.linearSublayersNumber, parallelStreams=False, sign=True, bias=True)
				layersLinearListO.append(linearO)				
		self.layersLinearF = nn.ModuleList(layersLinearListF)
		self.layersLinearB = nn.ModuleList(layersLinearListB)
		if(useBreakaway):
			self.layersLinearO = nn.ModuleList(layersLinearListO)
		self.activationF = ANNpt_linearSublayers.generateActivationFunction(activationFunctionTypeForward)
		self.activationB = ANNpt_linearSublayers.generateActivationFunction(activationFunctionTypeBackward)
		if(supportSkipLayers):
			self.layersB = nn.ModuleList(layersListB)
		
		if(useAutoencoder):
			self.lossFunctionBackward = nn.MSELoss()	#lossFunctionAutoencoder
		if(useBreakaway):
			self.lossFunctionOutput = nn.CrossEntropyLoss()
		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunctionFinal = nn.CrossEntropyLoss()
		else:
			self.lossFunctionFinal = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		self.Ztrace = [None]*(config.numberOfLayers+1)
		self.Atrace = [None]*(config.numberOfLayers+1)
		for l1 in range(1, numberOfLayers+1):
			self.Ztrace[l1] = pt.zeros([batchSize, self.n_h[l1]], device=device)
			self.Atrace[l1] = pt.zeros([batchSize, self.n_h[l1]], device=device)

	def generateLayerSizeList(self):
		n_h = [None]*(self.config.numberOfLayers+1)
		for l1 in range(0, self.config.numberOfLayers+1):
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
		self.Atrace[0] = AprevLayer
		self.Ztrace[0] = pt.zeros_like(AprevLayer)	#set to zero as not used (just used for shape initialisation)

		maxLayer = self.config.numberOfLayers
		
		for l1 in range(1, maxLayer+1):
			#print("l1 = ", l1)
			#print("self.config.numberOfLayers = ", self.config.numberOfLayers)
			
			A, Z, outputTarget = self.neuralNetworkPropagationLayerForward(l1, AprevLayer, autoencoder)
			if(trainingUpdateImplementation=="hebbian"):
				self.Ztrace[l1] = Z
				self.Atrace[l1] = A
			
			if(autoencoder):
				if(l1 == self.config.numberOfLayers):
					outputPred = Z	#activation function softmax is applied by self.lossFunctionFinal = nn.CrossEntropyLoss()
				else:
					if(trainOrTest):
						self.trainLayerHidden(l1, A, outputTarget, optim, Otarget=y)	#first layer optimiser is defined at i=0 
			else:
				if(l1 == self.config.numberOfLayers):
					outputPred = Z	#activation function softmax is applied by self.lossFunctionFinal = nn.CrossEntropyLoss()
				else:
					outputPred = A
			if(l1 == self.config.numberOfLayers):
				outputTarget = y
				if(trainOrTest):
					loss, accuracy = self.trainLayerFinal(l1, outputPred, outputTarget, optim, calculateAccuracy=True)
				else:
					loss, accuracy = self.calculateLossAccuracy(outputPred, outputTarget, self.lossFunctionFinal, calculateAccuracy=True)
			
			if(AEANNtrainGreedy):
				A = A.detach()	#only train weights for layer l1 (greedy training)

			AprevLayer = A
			self.Ztrace[l1] = Z
			self.Atrace[l1] = A
				
		return loss, accuracy

	def getLayerIndex(self, l1):
		layerIndex = l1-1
		return layerIndex
	
	def calculateActivationDerivative(self, A):
		Aactive = (A > 0).float()	#derivative of relu
		return Aactive
	
	def calculateMSE(self, pred, target):
		mse_per_neuron = (pred - target).mean(dim=0)
		#squared_differences = (pred - target) ** 2
		#mse_per_neuron = squared_differences.mean(dim=0)
		return mse_per_neuron

	def trainLayerFinal(self, l1, pred, target, optim, calculateAccuracy=False):
		lossFunction = self.lossFunctionFinal
		layerIndex = self.getLayerIndex(l1)
		loss, accuracy = self.calculateLossAccuracy(pred, target, lossFunction, calculateAccuracy)
		opt = optim[layerIndex]
		opt.zero_grad()
		loss.backward()
		opt.step()
		return loss, accuracy
				
	def trainLayerHidden(self, l1, Ahidden, Itarget, optim, calculateAccuracy=False, Otarget=None):
		
		if(useAutoencoder):
			Ipred = self.neuralNetworkPropagationLayerBackwardAutoencoder(l1, Ahidden)
		if(useBreakaway):
			Opred = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), Ahidden, self.layersLinearO[l1])
		
		if(trainingUpdateImplementation=="backprop"):
			layerIndex = self.getLayerIndex(l1)
			if(useAutoencoder):
				IlossFunction = self.lossFunctionBackward
				Iloss, Iaccuracy = self.calculateLossAccuracy(Ipred, Itarget, IlossFunction, calculateAccuracy)
				loss = Iloss
				accuracy = Iaccuracy
			if(useBreakaway):
				assert useAutoencoder
				OlossFunction = OlossFunction=self.lossFunctionOutput
				Oloss, Oaccuracy = self.calculateLossAccuracy(Opred, Otarget, OlossFunction, calculateAccuracy)
				loss = Iloss + Oloss
				accuracy = (Iaccuracy + Oaccuracy)/2
			opt = optim[layerIndex]
			opt.zero_grad()
			loss.backward()
			opt.step()
		elif(trainingUpdateImplementation=="hebbian"):
			#TODO: support supportSkipLayers
			print("l1 = ", l1)
			# backpropagation approximation notes:
				# error_L = (y_L - A_L) [sign reversal]
				# error_l = (W_l+1 * error_l+1) . activationFunctionPrime(z_l) {~A_l}
				# dC/dB = error_l
				# dC/dW = A_l-1 * error_l
				# Bnew = B+dC/dB [sign reversal]
				# Wnew = W+dC/dW [sign reversal]
			IOerror = self.calculateMSE(Ipred, Itarget)	#error_L = (A_L - y_L)
			print("IOerror = ", IOerror)
			inputA =  self.Atrace[l1-1]
			hiddenA = self.Atrace[l1]
			hiddenAactive = self.calculateActivationDerivative(hiddenA)
			#print("self.layersLinearB[l1] = ", self.layersLinearB[l1])
			ihWeight = self.layersLinearF[l1].weight
			hoWeight = self.layersLinearB[l1].weight
			print("ihWeight = ", ihWeight)
			print("hoWeight = ", hoWeight)
			#print("IOerror = ", IOerror)
			#print("hiddenAactive = ", hiddenAactive)
			hiddenError = pt.matmul(IOerror.unsqueeze(0), hoWeight)*hiddenAactive	 #backprop: error_l = (W_l+1 * error_l+1) . activationFunctionPrime(z_l)	#hidden neuron error must be passed back from axons to dendrites
			#print("hiddenError = ", hiddenError)
			ihdCdW = pt.matmul(inputA.transpose(0, 1), hiddenError)	#dC/dW = A_l-1 * error_l
			#print("hiddenA = ", hiddenA)	#64, 10
			#print("IOerror = ", IOerror)
			IOerrorExpanded = IOerror.unsqueeze(0).expand(batchSize, -1)	#expand to batch dim	#64,5
			#print("IOerrorExpanded = ", IOerrorExpanded)
			hodCdW = pt.matmul(hiddenA.transpose(0, 1), IOerrorExpanded)	#dC/dW = A_l-1 * error_l
			ihdCdB = pt.mean(hiddenError, dim=0)	#error_l
			hodCdB = IOerror	#error_l
			
			ihdCdW = ihdCdW.transpose(0, 1)*learningRate
			hodCdW = hodCdW.transpose(0, 1)*learningRate
			ihdCdB = ihdCdB*learningRate
			hodCdB = hodCdB*learningRate
			print("ihdCdW = ", ihdCdW)
			print("hodCdW = ", hodCdW)
			print("ihdCdB = ", ihdCdB)
			print("hodCdB = ", hodCdB)
			ihB = self.layersLinearF[l1].bias
			ihW = self.layersLinearF[l1].weight
			hoB = self.layersLinearB[l1].bias
			hoW = self.layersLinearB[l1].weight
			#print("ihW = ", ihW)
			#print("hoW = ", hoW)
			#print("ihB = ", ihB)
			#print("hoB = ", hoB)
			ihBnew = nn.Parameter(ihB-ihdCdB)	#Bnew = B-dC/dB
			ihWnew = nn.Parameter(ihW-ihdCdW)	#Wnew = W-dC/dW
			hoBnew = nn.Parameter(hoB-hodCdB)	#Bnew = B-dC/dB
			hoWnew = nn.Parameter(hoW-hodCdW)	#Wnew = W-dC/dW
			self.layersLinearF[l1].bias = ihBnew
			self.layersLinearF[l1].weight = ihWnew
			self.layersLinearB[l1].bias = hoBnew
			self.layersLinearB[l1].weight = hoWnew
			#if(supportSkipLayers):
			#	self.layersB = nn.ModuleList(layersListB)	#bias cannot be calculated using this method
			loss = None
			accuracy = None
		return loss, accuracy

	def calculateLossAccuracy(self, pred, target, lossFunction, calculateAccuracy=False):
		accuracy = 0
		if(calculateAccuracy):
			accuracy = self.accuracyFunction(pred, target)
		loss = lossFunction(pred, target)
		return loss, accuracy

	def neuralNetworkPropagationLayerForward(self, l1, AprevLayer, autoencoder):

		outputTarget = None

		if(autoencoder):
			if(autoencoderPrediction=="allPreviousLayers"):
				outputTargetList = []
				for l2 in range(0, l1):
					outputTargetListPartial = self.Atrace[l2]
					outputTargetList.append(outputTargetListPartial)
				outputTarget = pt.concat(outputTargetList, dim=1)
				#print("outputTarget.shape = ", outputTarget.shape)
			elif(autoencoderPrediction=="previousLayer"):
				outputTarget = AprevLayer
			elif(autoencoderPrediction=="inputLayer"):
				outputTarget = self.Atrace[0]
		
		if(supportSkipLayersF):
			Z = pt.zeros_like(self.Ztrace[l1])
			for l2 in range(0, l1):
				Zpartial = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), self.Atrace[l2], self.layersLinearF[l1][l2])
				Z = pt.add(Z, Zpartial)
			Z = pt.add(Z, self.layersB[l1](Z))	#need to add bias after skip layer connections have been added
		else:
			Z = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), AprevLayer, self.layersLinearF[l1])
		A = ANNpt_linearSublayers.executeActivationLayer(self, self.getLayerIndex(l1), Z, self.activationF)	#relU	

		return A, Z, outputTarget

	def neuralNetworkPropagationLayerBackwardAutoencoder(self, l1, A):
		if(autoencoderPrediction=="allPreviousLayers"):
			outputPredList = []
			for l2 in range(0, l1):
				Zback = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), A, self.layersLinearB[l1][l2])
				outputPredPartial = self.activationB(Zback)
				outputPredList.append(outputPredPartial)
			outputPred = pt.concat(outputPredList, dim=1)
		elif(supportSkipLayersB):
			if(autoencoderPrediction=="inputLayer"):
				Zback = pt.zeros_like(self.Ztrace[0])
			elif(autoencoderPrediction=="previousLayer"):
				Zback = pt.zeros_like(self.Ztrace[l1-1])
			for l2 in range(0, l1):
				if(l2 == l1-1):
					Apartial = A #allow backprop through current encoder layer only
				else:
					Apartial = self.Atrace[l1]
				ZbackPartial = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), Apartial, self.layersLinearB[l1][l2])
				Zback = pt.add(Zback, ZbackPartial)
			outputPred = self.activationB(Zback)
		else:
			Zback = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), A, self.layersLinearB[l1])
			outputPred = self.activationB(Zback)
		return outputPred
	
		


