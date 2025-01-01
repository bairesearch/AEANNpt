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
batchSize = 64
numberOfLayers = 4	#default: 4
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
datasetNormaliseClassValues = False	#reformat class values from 0.. ; contiguous (will also convert string to int)
datasetLocalFile = False


learningRate = 0.005	#0.005	#0.0001

datasetCorrectMissingValues = False	#initialise (dependent var)
datasetConvertClassTargetColumnFloatToInt = False	#initialise (dependent var)
dataloaderRepeatSampler = False	#initialise (dependent var)
dataloaderRepeatLoop = False	#initialise (dependent var)	#legacy (depreciate)
if(useTabularDataset):
	#datasetName = 'tabular-benchmark'	#expected test accuracy: 100%
	#datasetName = 'blog-feedback'	#expected test accuracy: ~72%
	#datasetName = 'titanic'	#expected test accuracy: ~87%
	#datasetName = 'red-wine'	#expected test accuracy: ~90%
	#datasetName = 'breast-cancer-wisconsin'	#expected test accuracy: ~55%
	#datasetName = 'diabetes-readmission'	#expected test accuracy: ~%
	datasetName = 'new-thyroid'	#expected test accuracy: 100%
	if(datasetName == 'tabular-benchmark'):
		datasetNameFull = 'inria-soda/tabular-benchmark'
		classFieldName = 'class'
		trainFileName = 'clf_cat/albert.csv'
		testFileName = 'clf_cat/albert.csv'
		datasetNormalise = True
		datasetShuffle = True	#raw dataset is not shuffled
		numberOfLayers = 4
		hiddenLayerSize = 10
		trainNumberOfEpochs = 1
	elif(datasetName == 'blog-feedback'):
		datasetNameFull = 'wwydmanski/blog-feedback'
		classFieldName = 'target'
		trainFileName = 'train.csv'
		testFileName = 'test.csv'
		datasetNormalise = True
		#datasetNormaliseClassValues = True	#int: not contiguous	#alternate method (slower)
		datasetConvertClassTargetColumnFloatToInt = True	#int: not contiguous
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 800	#default: 800
		trainNumberOfEpochs = 1	#default: 1
	elif(datasetName == 'titanic'):
		datasetNameFull = 'victor/titanic'
		classFieldName = '2urvived'
		trainFileName = 'train_and_test2.csv'	#train
		testFileName = 'train_and_test2.csv'	#test
		datasetReplaceNoneValues = True
		datasetNormalise = True
		datasetCorrectMissingValues = True
		datasetShuffle = True	#raw dataset is not completely shuffled
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 100	#default: 100
		trainNumberOfEpochs = 10	#default: 10
	elif(datasetName == 'red-wine'):
		datasetNameFull = 'lvwerra/red-wine'
		classFieldName = 'quality'
		trainFileName = 'winequality-red.csv'
		testFileName = 'winequality-red.csv'
		datasetNormaliseClassValues = True	#int: not start at 0
		datasetNormalise = True
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 6	#default: 6
		hiddenLayerSize = 100	#default: 100
		trainNumberOfEpochs = 100	#default: 100
	elif(datasetName == 'breast-cancer-wisconsin'):
		datasetNameFull = 'scikit-learn/breast-cancer-wisconsin'
		classFieldName = 'diagnosis'
		trainFileName = 'breast_cancer.csv'
		testFileName = 'breast_cancer.csv'
		datasetReplaceNoneValues = True
		datasetNormaliseClassValues = True	#string: B/M	#requires conversion of target string B/M to int
		datasetNormalise = True
		datasetCorrectMissingValues = True
		datasetShuffle = True	#raw dataset is not completely shuffled
		learningRate = 0.001	#default:  0.001
		numberOfLayers = 4	#default: 4
		hiddenLayerSize = 20	#default: 100
		trainNumberOfEpochs = 10	#default: 10
	elif(datasetName == 'diabetes-readmission'):
		datasetNameFull = 'imodels/diabetes-readmission'
		classFieldName = 'readmitted'
		trainFileName = 'train.csv'
		testFileName = 'test.csv'	
		datasetNormalise = True
		numberOfLayers = 4
		hiddenLayerSize = 10
		trainNumberOfEpochs = 1
	elif(datasetName == 'new-thyroid'):
		classFieldName = 'class'
		trainFileName = 'new-thyroid.csv'
		testFileName = 'new-thyroid.csv'
		datasetLocalFile = True	
		datasetNormalise = True
		datasetNormaliseClassValues = True
		numberOfLayers = 4
		hiddenLayerSize = 10
		trainNumberOfEpochs = 10
		datasetRepeat = True	#enable better sampling by dataloader with high batchSize (required if batchSize ~= datasetSize)
		if(datasetRepeat):
			datasetRepeatSize = 10
		dataloaderRepeat = True
	#elif ...

	if(dataloaderRepeat):
		dataloaderRepeatSize = 10	#number of repetitions
		dataloaderRepeatSampler = True
		dataloaderRepeatLoop = False	#legacy (depreciate)
		if(dataloaderRepeatSampler):
			dataloaderRepeatSamplerCustom = False	#no tqdm visualisation
		
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


	
