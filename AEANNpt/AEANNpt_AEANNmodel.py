"""AEANNpt_AEANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
AEANNpt Autoencoder/Breakaway generated artificial neural network model

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
			if(trainingUpdateImplementation == "hebbian"):
				layersLinearListOB = []
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
						if(useCNNlayers and l2==l1-1 and l1 < config.numberOfLayers):	#currently only use CNN for non-skip connections	#do not use CNN for final layer
							linearF2 = ANNpt_linearSublayers.generateLinearLayerCNN(self, l2, config, forward=True, bias=False, layerIndex2=l1)	#need to add bias after skip layer connections have been added
						else:
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
				if(l1 == config.numberOfLayers or not useImageDataset):
					featureBaseChannels = 1
				else:
					featureBaseChannels = numberInputImageChannels
				B = BiasLayer(self.n_h[l1]*featureBaseChannels)	#need to add bias after skip layer connections have been added
				layersListB.append(B)
			else:
				l2 = self.getLayerIndex(l1)
				if(useCNNlayers and l1 < config.numberOfLayers):	#do not use CNN for final layer
					linearF = ANNpt_linearSublayers.generateLinearLayerCNN(self, l2, config, forward=True, bias=True, layerIndex2=l1)
				else:
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
				linearO = ANNpt_linearSublayers.generateLinearLayer2(self, l2, config.hiddenLayerSize, config.outputLayerSize, config.linearSublayersNumber, parallelStreams=False, sign=True, bias=True, layerIndex2=l1)
				layersLinearListO.append(linearO)	
				if(trainingUpdateImplementation == "hebbian"):
					linearOB = ANNpt_linearSublayers.generateLinearLayer2(self, l1, config.hiddenLayerSize, config.outputLayerSize, config.linearSublayersNumber, parallelStreams=False, sign=True, bias=True)
					layersLinearListOB.append(linearOB)		
		self.layersLinearF = nn.ModuleList(layersLinearListF)
		self.layersLinearB = nn.ModuleList(layersLinearListB)
		if(useBreakaway):
			self.layersLinearO = nn.ModuleList(layersLinearListO)
			if(trainingUpdateImplementation == "hebbian"):
				self.layersLinearOB = nn.ModuleList(layersLinearListOB)
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

		batch_size = x.shape[0]
		if(useImageDataset):
			#model code always assumes data dimensions are flattened;
			x = x.reshape(batch_size, -1)

		for l1 in range(1, numberOfLayers+1):
			if(l1 == self.config.numberOfLayers or not useImageDataset):
				featureBaseChannels = 1
			else:
				featureBaseChannels = numberInputImageChannels
			self.Ztrace[l1] = pt.zeros([batch_size, self.n_h[l1]*featureBaseChannels], device=device)
			self.Atrace[l1] = pt.zeros([batch_size, self.n_h[l1]*featureBaseChannels], device=device)

		outputPred = x #in case layer=0

		AprevLayer = x
		self.Atrace[0] = AprevLayer
		self.Ztrace[0] = pt.zeros_like(AprevLayer)	#set to zero as not used (just used for shape initialisation)
		
		maxLayer = self.config.numberOfLayers
		
		for l1 in range(1, maxLayer+1):
			#print("l1 = ", l1)
			#print("self.config.numberOfLayers = ", self.config.numberOfLayers)
			
			A, Z, inputTarget = self.neuralNetworkPropagationLayerForward(l1, AprevLayer, autoencoder)
			if(trainingUpdateImplementation=="hebbian"):
				self.Ztrace[l1] = Z
				self.Atrace[l1] = A
			
			if(autoencoder or useBreakaway):
				if(l1 == self.config.numberOfLayers):
					outputPred = Z	#activation function softmax is applied by self.lossFunctionFinal = nn.CrossEntropyLoss()
				else:
					if(trainOrTest):
						self.trainLayerHidden(l1, AprevLayer, A, inputTarget, y, optim)	#first layer optimiser is defined at i=0 
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
				Z = Z.detach()	#not required; added for robustness
				
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
	
	def calcMSEDerivative(self, pred, target):
		mse_per_neuron = (pred - target)
		return mse_per_neuron

	def trainLayerFinal(self, l1, pred, target, optim, calculateAccuracy=False):
		lossFunction = self.lossFunctionFinal
		layerIndex = self.getLayerIndex(l1)
		loss, accuracy = self.calculateLossAccuracy(pred, target, lossFunction, calculateAccuracy)
		if(trainLocal):
			opt = optim[layerIndex]
			opt.zero_grad()
			loss.backward()
			opt.step()
		return loss, accuracy
				
	def trainLayerHidden(self, l1, AprevLayer, Ahidden, Itarget, Otarget, optim, calculateAccuracy=False):
		
		Ipred = None
		Opred = None
		if(useAutoencoder):
			Ipred = self.neuralNetworkPropagationLayerBackwardAutoencoder(l1, Ahidden)
		if(useBreakaway):
			Opred = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), Ahidden, self.layersLinearO[l1])
		
		if(trainingUpdateImplementation=="backprop"):
			if(trainingUpdateImplementationBackpropAuto):
				loss, accuracy = self.trainLayerHiddenBackpropAuto(l1, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy)
			else:
				loss, accuracy = self.trainLayerHiddenBackpropManual(l1, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy)
		elif(trainingUpdateImplementation=="hebbian"):
			loss, accuracy = self.trainLayerHiddenHebbian(l1, AprevLayer, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy)
			
		return loss, accuracy

	def trainLayerHiddenBackpropAuto(self, l1, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy=False):
		layerIndex = self.getLayerIndex(l1)
		if(useAutoencoder):
			IlossFunction = self.lossFunctionBackward
			Iloss, Iaccuracy = self.calculateLossAccuracy(Ipred, Itarget, IlossFunction, calculateAccuracy)
			loss = Iloss
			accuracy = Iaccuracy
		if(useBreakaway):
			OlossFunction = OlossFunction=self.lossFunctionOutput
			Oloss, Oaccuracy = self.calculateLossAccuracy(Opred, Otarget, OlossFunction, calculateAccuracy)
			if(useAutoencoder):
				loss = Iloss + Oloss
				accuracy = (Iaccuracy + Oaccuracy)/2
			else:
				loss = Oloss
				accuracy = Oaccuracy
		opt = optim[layerIndex]
		opt.zero_grad()
		loss.backward()
		opt.step()	
		
		return loss, accuracy
		
	def trainLayerHiddenBackpropManual(self, l1, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy=False):
		#print("l1 = ", l1)

		# backpropagation approximation notes:
			# error_L = (A_L - y_L)
			# error_l = (W_l+1 * error_l+1) . activationFunctionPrime(z_l) {~A_l}
			# dC/dB = error_l
			# dC/dW = A_l-1 * error_l
			# Bnew = B-dC/dB
			# Wnew = W-dC/dW

		iA = self.Atrace[l1-1]
		hA = self.Atrace[l1]	#or hZ = self.Ztrace[l1]

		#hi[+ho] activation deltas:
		hZderivative = self.calculateActivationDerivative(hA)	#or hZ
		if(useAutoencoder):
			iError2 = self.calcMSEDerivative(Ipred, Itarget)	#shape: batSize*iSize
			hiWeight = self.layersLinearB[l1].weight	#shape: iSize*hSize
			hErrorI = pt.matmul(iError2, hiWeight)*hZderivative	#backprop algorithm: error_l = (W_l+1 * error_l+1) . activationFunctionPrime(z_l)	#hidden neuron error must be passed back from axons to dendrites	#shape: batSize*hSize
			hError = hErrorI	#shape: batSize*hSize
		if(useBreakaway):
			oTargetOneHot = pt.nn.functional.one_hot(Otarget, num_classes=self.config.numberOfClasses)
			oError2 = self.calcMSEDerivative(Opred, oTargetOneHot)	#shape: batSize*iSize
			hoWeight = self.layersLinearO[l1].weight	# shape: (oSize, hSize)
			hErrorO = pt.matmul(oError2, hoWeight)*hZderivative
			if(useAutoencoder):
				hError = (hErrorI+hErrorO)	#OLD: /2
			else:
				hError = hErrorO

		#ih activation deltas:
		iZderivative = self.calculateActivationDerivative(iA)	#or iZ
		ihWeight = self.layersLinearF[l1].weight	#shape: hSize*iSize
		iError1 = pt.matmul(hError, ihWeight)*iZderivative	#backprop algorithm: error_l = (W_l+1 * error_l+1) . activationFunctionPrime(z_l)	#hidden neuron error must be passed back from axons to dendrites	#shape: batSize*iSize

		#hi[+ho] parameter deltas:
		if(useAutoencoder):
			hidCdW = pt.matmul(iError2.transpose(0, 1), hA) / batchSize	#backprop algorithm: dC/dW = A_l-1 * error_l
			hidCdB = pt.mean(iError2, dim=0)	#error_l
		if(useBreakaway):
			hodCdW = pt.matmul(oError2.transpose(0, 1), hA) / batchSize	#backprop algorithm: dC/dW = A_l-1 * error_l	
			hodCdB = pt.mean(oError2, dim=0)	#error_l

		#ih parameter deltas:
		ihdCdW = pt.matmul(hError.transpose(0, 1), iA) / batchSize	#backprop algorithm: dC/dW = A_l-1 * error_l
		ihdCdB = pt.mean(hError, dim=0)	#error_l

		ihdCdW = ihdCdW*learningRate
		ihdCdB = ihdCdB*learningRate
		if(useAutoencoder):
			hidCdW = hidCdW*learningRate
			hidCdB = hidCdB*learningRate
		if(useBreakaway):
			hodCdW = hodCdW*learningRate
			hodCdB = hodCdB*learningRate

		with pt.no_grad():
			self.layersLinearF[l1].weight -= ihdCdW
			self.layersLinearF[l1].bias -= ihdCdB
			if(useAutoencoder):
				self.layersLinearB[l1].weight -= hidCdW
				self.layersLinearB[l1].bias -= hidCdB
			if(useBreakaway):
				self.layersLinearO[l1].weight -= hodCdW
				self.layersLinearO[l1].bias   -= hodCdB

		loss = None
		accuracy = None
		return loss, accuracy

	def trainLayerHiddenHebbian(self, l1, AprevLayer, Ahidden, Itarget, Ipred, optim, Otarget, Opred, calculateAccuracy=False):
		if(useAutoencoder):
			printe("trainLayerHiddenHebbian does not currently support useAutoencoder")
		if(useBreakaway):
			'''
			experimental;
			1. apply hebbian learning of connections between between hidden and output layer [step not required if not sending back error in step 2]
			2. output layer [dependent on error calc?] sends reverse connections back to both input (ie prev hidden) and hidden layers
			3. apply hebbian learning of connections between output and input/hidden layers
				use activations from input layer through forward to highlight input/hidden neurons for hebbian learning
				use activations from output layer target (ie ideal output) to highlight output layer neurons for hebbian learning
			4. in a second artifically stimulated [by output layer] forward prop round calculate the coincidence of the input and hidden layer firings
			5. strengthen those input->hidden connections that exhibit high coincidence [hebbian learning]
			'''
			#2-3;
			performHebbianLearning(Otarget, AprevLayer, self.layersLinearOB[l1-1])	#note these are backwards connections from o to i
			performHebbianLearning(Otarget, Ahidden, self.layersLinearOB[l1])	#note these are backwards connections from o to h
			#4-5;
			AprevLayerArtificial =  ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1-1), Otarget, self.layersLinearOB[l1-1])
			AhiddenArtificial = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), Otarget, self.layersLinearOB[l1])
			performHebbianLearning(AprevLayerArtificial, AhiddenArtificial, self.layersLinearF[l1])
			
		loss = None
		accuracy = None
		return loss, accuracy
		
	def performHebbianLearning(inputActivations, outputActivations, layer):
		coincidenceMatrix = calculateCoincidenceMatrix(inputActivations, outputActivations)	#note pytorch linear weights are defined by o*i
		strength = coincidenceMatrix*learningRate
		#TODO; apply negative error
		layer.weight = layer.weight + strength 
	
	def calculateCoincidenceMatrix(inputActivations, outputActivations):
		coincidenceMatrix = torch.matmul(outputActivations.transpose(0, 1), inputActivations)		#note pytorch linear weights are defined by o*i
		return coincidenceMatrix

	def calculateLossAccuracy(self, pred, target, lossFunction, calculateAccuracy=False):
		accuracy = 0
		if(calculateAccuracy):
			accuracy = self.accuracyFunction(pred, target)
		loss = lossFunction(pred, target)
		return loss, accuracy

	def neuralNetworkPropagationLayerForward(self, l1, AprevLayer, autoencoder):

		inputTarget = None

		if(autoencoder):
			if(autoencoderPrediction=="allPreviousLayers"):
				inputTargetList = []
				for l2 in range(0, l1):
					inputTargetListPartial = self.Atrace[l2]
					inputTargetList.append(inputTargetListPartial)
				inputTarget = pt.concat(inputTargetList, dim=1)
				#print("inputTarget.shape = ", inputTarget.shape)
			elif(autoencoderPrediction=="previousLayer"):
				inputTarget = AprevLayer
			elif(autoencoderPrediction=="inputLayer"):
				inputTarget = self.Atrace[0]
		
		if(supportSkipLayersF):
			Z = pt.zeros_like(self.Ztrace[l1])
			for l2 in range(0, l1):
				cnn = False
				if(useCNNlayers and l2==l1-1 and l1 < self.config.numberOfLayers):	#currently only use CNN for non-skip connections #do not use CNN for final layer
					cnn = True
				Zpartial = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), self.Atrace[l2], self.layersLinearF[l1][l2], cnn=cnn)
				Z = pt.add(Z, Zpartial)
			Z = pt.add(Z, self.layersB[l1](Z))	#need to add bias after skip layer connections have been added
		else:
			cnn = False
			if(useCNNlayers and l1 < self.config.numberOfLayers):	#do not use CNN for final layer
				cnn = True
			Z = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l1), AprevLayer, self.layersLinearF[l1], cnn=cnn)
		A = ANNpt_linearSublayers.executeActivationLayer(self, self.getLayerIndex(l1), Z, self.activationF)	#relU	

		return A, Z, inputTarget

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
	
		


