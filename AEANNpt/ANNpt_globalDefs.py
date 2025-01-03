"""ANNpt_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2023-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
ANNpt globalDefs

"""

#algorithm selection
useAlgorithmVICRegANN = False
useAlgorithmAUANN = False
useAlgorithmLIANN = False
useAlgorithmLUANN = False
useAlgorithmLUOR = False
useAlgorithmSANIOR = False
useAlgorithmEIANN = False
useAlgorithmEIOR = False
useAlgorithmAEANN = True

#initialise (dependent vars);
usePairedDataset = False
datasetNormalise = False
datasetRepeat = False
datasetShuffle = False	#automatically performed by generateVICRegANNpairedDatasets
datasetOrderByClass = False	#automatically performed by generateVICRegANNpairedDatasets
dataloaderShuffle = True
dataloaderMaintainBatchSize = True
dataloaderRepeat = False

optimiserAdam = True

#initialise (dependent vars);
useCustomWeightInitialisation = False
useCustomBiasInitialisation = False
useCustomBiasNoTrain = False
useSignedWeights = False
usePositiveWeightsClampModel = False

debugSmallBatchSize = False	#small batch size for debugging matrix output
debugDataNormalisation = False

trainLocal = False
trainGreedy = False
trainIndividialSamples = False

useLinearSublayers = False	#use multiple independent sublayers per linear layer	#optional
if(useLinearSublayers):
	linearSublayersNumber = 10
else:
	linearSublayersNumber = 1
	

#default network hierarchy parameters (overwritten by specific dataset defaults): 
learningRate = 0.005	#0.005	#0.0001
batchSize = 64	 #default: 64	#debug: 2
numberOfLayers = 4	#default: 4	#counts hidden and output layers (not input layer)
hiddenLayerSize = 10	#default: 10
trainNumberOfEpochs = 10	#default: 10

#initialise (dependent vars);
debugSmallNetwork = False
trainNumberOfEpochsHigh = False
inputLayerInList = True
outputLayerInList = True
useCNNlayers = False
thresholdActivations = False
debugPrintActivationOutput = False
simulatedDendriticBranches = False
activationFunctionType = "relu"
trainLastLayerOnly = False
normaliseActivationSparsity = False
debugUsePositiveWeightsVerify = False
datasetNormaliseMinMax = True	#normalise between 0.0 and 1.0
datasetNormaliseStdAvg = False	#normalise based on std and mean (~-1.0 to 1.0)
		
useInbuiltCrossEntropyLossFunction = True	#required
if(useSignedWeights):
	usePositiveWeightsClampModel = True	#clamp entire model weights to be positive (rather than per layer); currently required

useTabularDataset = False
useImageDataset = False
if(useAlgorithmVICRegANN):
	from VICRegANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmAUANN):
	from LREANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLIANN):
	from LIANNpt_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLUANN):
	from LUANNpt_LUANN_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmLUOR):
	from LUANNpt_LUOR_globalDefs import *
	useImageDataset = True
elif(useAlgorithmSANIOR):
	from LUANNpt_SANIOR_globalDefs import *
	useImageDataset = True
elif(useAlgorithmEIANN):
	from EIANNpt_EIANN_globalDefs import *
	useTabularDataset = True
elif(useAlgorithmEIOR):
	from EIANNpt_EIOR_globalDefs import *
	useImageDataset = True
elif(useAlgorithmAEANN):
	from AEANNpt_AEANN_globalDefs import *
	useTabularDataset = True
	
import torch as pt

useLovelyTensors = False
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	pt.set_printoptions(profile="full")
	pt.set_printoptions(sci_mode=False)
	
#pt.autograd.set_detect_anomaly(True)

stateTrainDataset = True
stateTestDataset = True

if(useCustomWeightInitialisation):
	Wmean = 0.0
	WstdDev = 0.05	#stddev of weight initialisations

#initialise (dependent vars);
datasetReplaceNoneValues = False
datasetConvertClassValues = False	#reformat class values from 0.. ; contiguous (will also convert string to int)
datasetConvertFeatureValues = False	#reformat feature values from 0.. ; contiguous (will also convert string to int)
datasetLocalFile = False
datasetSpecifyDataFiles = True	#specify data file names in dataset (else automatically selected by huggingface)
datasetHasTestSplit = True
datasetHasSubsetType = False

datasetCorrectMissingValues = False	#initialise (dependent var)
datasetConvertClassTargetColumnFloatToInt = False	#initialise (dependent var)
dataloaderRepeatSampler = False	#initialise (dependent var)
dataloaderRepeatLoop = False	#initialise (dependent var)	#legacy (depreciate)
if(useTabularDataset):
	#datasetName = 'tabular-benchmark'	#expected test accuracy: ~58%
	#datasetName = 'blog-feedback'	#expected test accuracy: ~64%
	#datasetName = 'titanic'	#expected test accuracy: ~88%
	#datasetName = 'red-wine'	#expected test accuracy: ~33%
	#datasetName = 'breast-cancer-wisconsin'	#expected test accuracy: ~33%
	#datasetName = 'diabetes-readmission'	#expected test accuracy: ~58%
	#datasetName = 'banking-marketing'	#expected test accuracy: ~74%	#third party benchmark accuracy: ~79.1%
	#datasetName = 'adult_income_dataset'	#expected test accuracy: ~75% 	#third party benchmark accuracy: ~85.8%
	#datasetName = 'covertype'	#expected test accuracy: ~%89 	#third party benchmark accuracy: ~97.1%
	#datasetName = 'higgs'	#expected test accuracy: 85%	#third party benchmark accuracy: 73.8%	#https://archive.ics.uci.edu/dataset/280/higgs	#untested
	datasetName = 'new-thyroid'	#expected test accuracy: ~50%
	if(datasetName == 'tabular-benchmark'):
		datasetNameFull = 'inria-soda/tabular-benchmark'
		classFieldName = 'class'
		trainFileName = 'clf_cat/albert.csv'
		datasetHasTestSplit = False
		datasetNormalise = True
		#datasetShuffle = True	#raw dataset is not shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 10	#default: 10
		trainNumberOfEpochs = 1	#default: 1
	elif(datasetName == 'blog-feedback'):
		datasetNameFull = 'wwydmanski/blog-feedback'
		classFieldName = 'target'
		datasetSpecifyDataFiles = False
		datasetNormalise = True
		#datasetConvertClassValues = True	#int: not contiguous	#alternate method (slower)
		datasetConvertClassTargetColumnFloatToInt = True	#int: not contiguous
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 128	#default: 128	#orig: 800
		trainNumberOfEpochs = 1	#default: 1
	elif(datasetName == 'titanic'):
		datasetNameFull = 'victor/titanic'
		classFieldName = '2urvived'
		datasetSpecifyDataFiles = False
		datasetReplaceNoneValues = True
		datasetNormalise = True
		datasetCorrectMissingValues = True
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 100	#default: 100
		trainNumberOfEpochs = 10	#default: 10
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
	elif(datasetName == 'red-wine'):
		datasetNameFull = 'lvwerra/red-wine'
		classFieldName = 'quality'
		trainFileName = 'winequality-red.csv'
		datasetHasTestSplit = False
		datasetConvertClassValues = True	#int: not start at 0
		datasetNormalise = True
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4	#external benchmark: 4
		hiddenLayerSize = 100	#default: 100	#external benchmark; 64/128
		trainNumberOfEpochs = 10	#default: 10	#note train accuracy continues to increase (overfit) with increasing epochs
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
	elif(datasetName == 'breast-cancer-wisconsin'):
		datasetNameFull = 'scikit-learn/breast-cancer-wisconsin'
		classFieldName = 'diagnosis'
		trainFileName = 'breast_cancer.csv'
		datasetHasTestSplit = False
		datasetReplaceNoneValues = True
		datasetConvertClassValues = True	#string: B/M
		datasetNormalise = True
		datasetCorrectMissingValues = True
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 20	#default: 20	#old: 100
		trainNumberOfEpochs = 10	#default: 10
		datasetRepeat = True
		if(datasetRepeat):
			datasetRepeatSize = 10
	elif(datasetName == 'diabetes-readmission'):
		datasetNameFull = 'imodels/diabetes-readmission'
		classFieldName = 'readmitted'
		datasetSpecifyDataFiles = False
		datasetNormalise = True
		learningRate = 0.005
		numberOfLayers = 4
		hiddenLayerSize = 10
		trainNumberOfEpochs = 1
	elif(datasetName == 'banking-marketing'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetNameFull = 'Andyrasika/banking-marketing'
		classFieldName = 'y'
		datasetConvertClassValues = True	#string: yes/no
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 5
		hiddenLayerSize = 128
		trainNumberOfEpochs = 1
	elif(datasetName == 'adult_income_dataset'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetHasTestSplit = False
		datasetNameFull = 'meghana/adult_income_dataset'
		classFieldName = 'income'
		datasetConvertClassValues = True	#string: <=50K/>50K
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 4
		hiddenLayerSize = 256
		trainNumberOfEpochs = 10	
	elif(datasetName == 'covertype'):
		datasetSpecifyDataFiles = False
		datasetConvertFeatureValues = True	#required if they contain strings
		datasetHasTestSplit = False
		datasetHasSubsetType = True
		datasetSubsetName = 'covertype'
		datasetNameFull = 'mstz/covertype'
		classFieldName = 'cover_type'
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 6
		hiddenLayerSize = 512
		trainNumberOfEpochs = 10		#train performance increases with higher epochs
	elif(datasetName == 'higgs'):
		datasetLocalFile = True		#manually download from https://archive.ics.uci.edu/dataset/280/higgs
		#datasetNameFull = 'mstz/higgs'	#jxie/higgs
		classFieldName = 'y'
		trainFileName = 'HIGGS.csv'	#CHECKTHIS
		testFileName = 'HIGGS.csv'	#CHECKTHIS
		datasetNormalise = True
		learningRate = 0.001
		numberOfLayers = 5
		hiddenLayerSize = 256	#300
		trainNumberOfEpochs = 1		
	elif(datasetName == 'new-thyroid'):
		classFieldName = 'class'
		trainFileName = 'new-thyroid.csv'
		datasetHasTestSplit = False
		#datasetShuffle = True	#raw dataset is not completely shuffled	#not required with dataloaderShuffle 
		datasetLocalFile = True	
		datasetNormalise = True
		datasetConvertClassValues = True
		learningRate = 0.005	#default: 0.001	#orig: 0.005
		numberOfLayers = 4	#default: 4	#orig: 2
		hiddenLayerSize = 10	#default: 10	#orig: 4
		#numberOfLayers = 2
		#hiddenLayerSize = 4
		#batchSize = 1
		trainNumberOfEpochs = 1	#default: 1
		datasetRepeat = True	#enable better sampling by dataloader with high batchSize (required if batchSize ~= datasetSize)
		if(datasetRepeat):
			datasetRepeatSize = 100	#for batchSize ~= 64
	#elif ...

	if(dataloaderRepeat):
		dataloaderRepeatSize = 10	#number of repetitions
		dataloaderRepeatLoop = False	#legacy (depreciate)
		dataloaderRepeatSampler = True
		if(dataloaderRepeatSampler):
			dataloaderRepeatSamplerCustom = False	#no tqdm visualisation
			assert not dataloaderShuffle	#dataloaderShuffle is not supported by dataloaderRepeatSampler
	
if(trainNumberOfEpochsHigh):
	trainNumberOfEpochs = trainNumberOfEpochs*4
	
if(debugSmallBatchSize):
	batchSize = 10
if(debugSmallNetwork):
	batchSize = 2
	numberOfLayers = 4
	hiddenLayerSize = 5	
	trainNumberOfEpochs = 1
	
printAccuracyRunningAverage = True
if(printAccuracyRunningAverage):
	runningAverageBatches = 10

datasetSplitNameTrain = 'train'
datasetSplitNameTest = 'test'
if(not datasetHasTestSplit):
	datasetTestSplitSize = 0.1

relativeFolderLocations = False
userName = 'user'	#default: user
tokenString = "INSERT_HUGGINGFACE_TOKEN_HERE"	#default: INSERT_HUGGINGFACE_TOKEN_HERE
import os
if(os.path.isdir('user')):
	from user.user_globalDefs import *

modelSaveNumberOfBatches = 100	#resave model after x training batches

dataFolderName = 'data'
modelFolderName = 'model'
if(relativeFolderLocations):
	dataPathName = dataFolderName
	modelPathName = modelFolderName
else:	
	dataPathName = '/media/' + userName + dataDrive + dataFolderName
	modelPathName = '/media/' + userName + workingDrive + modelFolderName

def getModelPathNameFull(modelPathNameBase, modelName):
	modelPathNameFull = modelPathNameBase + '/' + modelName + '.pt'
	return modelPathNameFull
	
modelPathNameBase = modelPathName
modelPathNameFull = getModelPathNameFull(modelPathNameBase, modelName)
	
def printCUDAmemory(tag):
	print(tag)
	
	pynvml.nvmlInit()
	h = pynvml.nvmlDeviceGetHandleByIndex(0)
	info = pynvml.nvmlDeviceGetMemoryInfo(h)
	total_memory = info.total
	memory_free = info.free
	memory_allocated = info.used
	'''
	total_memory = pt.cuda.get_device_properties(0).total_memory
	memory_reserved = pt.cuda.memory_reserved(0)
	memory_allocated = pt.cuda.memory_allocated(0)
	memory_free = memory_reserved-memory_allocated  # free inside reserved
	'''
	print("CUDA total_memory = ", total_memory)
	#print("CUDA memory_reserved = ", memory_reserved)
	print("CUDA memory_allocated = ", memory_allocated)
	print("CUDA memory_free = ", memory_free)

def printe(str):
	print(str)
	exit()

device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')


	
