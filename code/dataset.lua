require 'torch'
require 'xlua'
require 'image'
require 'csv'

dataset = {}

DATA_PATH = '/home/bruno/workspace/lua/age estimation/scripts/faces_small/'
classLabels_PATH = '/home/bruno/workspace/lua/age estimation/scripts/labels/'

classLabels_FILE_NAME = 'labels.csv'

dataset.IMAGE_CHANELS = 1
dataset.IMAGE_WIDTH = 128
dataset.IMAGE_HEIGHT = 128

dataset.TRAIN_SPLIT = 0.8

dataset.AGE_SPLITTING = {20, 24, 27, 31, 36, 43, 50, 56, 62}	-- determined manually
dataset.SPLITTING_AGE = 36	-- default value, usually changed during preprocessing

function string.ends(String, End)
   return End == '' or string.sub(String, -string.len(End)) == End
end

function generateOneHotVector(age)
	age = tonumber(age)
	local oneHotVector = torch.zeros(#dataset.AGE_SPLITTING + 1)
	local classLabel = 1
	for i,v in ipairs(dataset.AGE_SPLITTING) do
		if age < v then
			oneHotVector[classLabel] = 1
			return oneHotVector
		else
			classLabel = classLabel + 1
		end
	end
	oneHotVector[classLabel] = 1
	return oneHotVector
end

function determineClassLabel(age)
	local classLabel = 1
	for i,v in ipairs(dataset.AGE_SPLITTING) do
		if age < v then
			return classLabel
		else
			classLabel = classLabel + 1
		end
	end
	return classLabel
end

function determineClassLabelSimple(age)
	age = tonumber(age)
	if age <= dataset.SPLITTING_AGE then return 1 else return 2 end
end

function dataset.loadDatasetTwoClass(percent2Load)
	local csv = require('csv')
	local f = csv.open(classLabels_PATH .. classLabels_FILE_NAME)
	local fileLines = f:lines()

	local dataNames = {}
	local labels = {}

	for fields in f:lines() do
		local fileName = fields[1]
		local age = fields[3]
--		print(fileName .. ' ' .. age .. ' years old')
		if fileName:ends('png') then
			table.insert(labels, tonumber(age))
			table.insert(dataNames, DATA_PATH .. fileName)
		end
	end
	
	local labels2Sort = {}
	local labels2Use = {}
	local dataNames2Use = {}
	numOfExamples = math.floor(#labels * percent2Load)
	numOfExamples = numOfExamples - (numOfExamples % 5)

	for i = 1, numOfExamples do
		labels2Sort[i] = labels[i]
		labels2Use[i] = labels[i]
		dataNames2Use[i] = dataNames[i]
	end

	table.sort(labels2Sort)
	medianIndex = math.floor(#labels2Sort/2)
	print(medianIndex)
	dataset.SPLITTING_AGE = labels2Sort[medianIndex]		-- median
	print('using ' .. dataset.SPLITTING_AGE .. ' as splitting age..')

	trainNumOfExamples = dataset.TRAIN_SPLIT * #labels2Use	-- assuming number of examples is divisible by 5
	testNumOfExamples = #labels2Use - trainNumOfExamples

	torch.setdefaulttensortype('torch.ByteTensor')
	local trainData = torch.Tensor(trainNumOfExamples, dataset.IMAGE_CHANELS, dataset.IMAGE_WIDTH, dataset.IMAGE_HEIGHT)
	local testData = torch.Tensor(testNumOfExamples, dataset.IMAGE_CHANELS, dataset.IMAGE_WIDTH, dataset.IMAGE_HEIGHT)

	local trainLabels = torch.Tensor(trainNumOfExamples)
	local testLabels = torch.Tensor(testNumOfExamples)

	local iTrain = 1
	local iTest = 1

	for i = 1, #labels2Use do
		if (i % 100 == 0) then
			print(i)
		end
		imageName = dataNames2Use[i]
		if (i % 5 == 0) then
			testData[iTest] = image.load(imageName, 1, 'byte')
			testLabels[iTest] = determineClassLabelSimple(labels2Use[i])
			iTest = iTest + 1
		else
			trainData[iTrain] = image.load(imageName, 1, 'byte')
			trainLabels[iTrain] = determineClassLabelSimple(labels2Use[i])
			iTrain = iTrain + 1
		end
	end

	return trainData, testData, trainLabels, testLabels
end


function dataset.loadDatasetTenClass(percent2Load, classification)
	local csv = require('csv')
	local f = csv.open(classLabels_PATH .. classLabels_FILE_NAME)
	local fileLines = f:lines()

	local dataNamesAll = {}
	local labelsAll = {}

	for fields in f:lines() do
		local fileName = fields[1]
		local age = fields[3]
--		print(fileName .. ' ' .. age .. ' years old')
		if fileName:ends('png') then
			table.insert(labelsAll, tonumber(age))
			table.insert(dataNamesAll, DATA_PATH .. fileName)
		end
	end
	
	numOfExamples = math.floor(#labelsAll * percent2Load)
	local dataNames2Keep = {}
	local labels2Keep = {}

	if classification then
		numOfExamplesPerClass = math.floor(numOfExamples / 10)
		numOfExamples = 10 * numOfExamplesPerClass	-- rounding
		print(numOfExamples)
		print(#labelsAll)

		local counterMap = {}	-- to make sure each class has equal number of examples
		for i = 1, 10 do
			table.insert(counterMap, numOfExamplesPerClass)
		end

		if percent2Load == 1 then		-- no need to keep track of distribution
			for i = 1, #labelsAll do
				classLabel = determineClassLabel(labelsAll[i])
				table.insert(dataNames2Keep, dataNamesAll[i])
				table.insert(labels2Keep, classLabel)
			end
		else
			print('making sure data is evenly distributed..')
			for i = 1, #labelsAll do
				classLabel = determineClassLabel(labelsAll[i])
				if counterMap[classLabel] > 0 then
					table.insert(dataNames2Keep, dataNamesAll[i])
					table.insert(labels2Keep, classLabel)
					counterMap[classLabel] = counterMap[classLabel] - 1
				end
				if #labels2Keep == numOfExamples then break end
			end
		end
	else
		print(numOfExamples)
		print(#labelsAll)
		torch.setdefaulttensortype('torch.FloatTensor')
		local shuffle = torch.randperm(#labelsAll)
		for i = 1, #labelsAll do
			classLabel = labelsAll[shuffle[i]]
			table.insert(dataNames2Keep, dataNamesAll[shuffle[i]])
			table.insert(labels2Keep, classLabel)
		end
	end

	trainNumOfExamples = dataset.TRAIN_SPLIT * numOfExamples

	local shuffle = torch.randperm(numOfExamples)

	torch.setdefaulttensortype('torch.ByteTensor')
	local data = torch.Tensor(numOfExamples, dataset.IMAGE_CHANELS, dataset.IMAGE_WIDTH, dataset.IMAGE_HEIGHT)
	local targets = torch.Tensor(numOfExamples)

	for i = 1, numOfExamples do
		if (i % 100 == 0) then
			print(i)
		end
		imageName = dataNames2Keep[i]
		data[shuffle[i]] = image.load(imageName, 1, 'byte')
		targets[shuffle[i]] = labels2Keep[i]
	end

	return  data[{ {1, trainNumOfExamples}, {}, {}, {} }],
			data[{ {trainNumOfExamples+1, -1}, {}, {}, {} }],
			targets[{{1, trainNumOfExamples}}],
			targets[{{trainNumOfExamples+1, -1}}]
end

function dataset.loadDatasetCustom(numOfExamples)
	local youngCounter = numOfExamples / 2
	local oldCounter = numOfExamples - youngCounter

	local csv = require('csv')
	local f = csv.open(classLabels_PATH .. classLabels_FILE_NAME)
	local fileLines = f:lines()

	local dataNames = {}
	local labels = {}

	for fields in f:lines() do
		local fileName = fields[1]
		local age = tonumber(fields[3])
		if fileName:ends('png') then
			if age <= 25 and youngCounter > 0 then
				table.insert(labels, age)
				table.insert(dataNames, DATA_PATH .. fileName)
				youngCounter = youngCounter - 1
			elseif age >= 55 and oldCounter > 0 then
				table.insert(labels, age)
				table.insert(dataNames, DATA_PATH .. fileName)
				oldCounter = oldCounter - 1
			end
			if youngCounter == 0 and oldCounter == 0 then
				break
			end
		end
	end

	trainNumOfExamples = dataset.TRAIN_SPLIT * #labels	-- assuming number of examples is divisible by 5
	testNumOfExamples = #labels - trainNumOfExamples

	torch.setdefaulttensortype('torch.ByteTensor')
	local trainData = torch.Tensor(trainNumOfExamples, dataset.IMAGE_CHANELS, dataset.IMAGE_WIDTH, dataset.IMAGE_HEIGHT)
	local testData = torch.Tensor(testNumOfExamples, dataset.IMAGE_CHANELS, dataset.IMAGE_WIDTH, dataset.IMAGE_HEIGHT)

	local trainLabels = torch.Tensor(trainNumOfExamples)
	local testLabels = torch.Tensor(testNumOfExamples)

	local iTrain = 1
	local iTest = 1

	torch.setdefaulttensortype('torch.FloatTensor')
	local testShuffle = torch.randperm(testNumOfExamples)
	local trainShuffle = torch.randperm(trainNumOfExamples)
	torch.setdefaulttensortype('torch.ByteTensor')

	-- examples seem to be almost sorted already, making split more less balanced	
	for i = 1, #labels do
		if (i % 100 == 0) then
--			print(i)
		end
		imageName = dataNames[i]
		if (i % 5 == 0) then
			testData[testShuffle[iTest]] = image.load(imageName, 1, 'byte')
			testLabels[testShuffle[iTest]] = determineClassLabelSimple(labels[i])
			iTest = iTest + 1
		else
			trainData[trainShuffle[iTrain]] = image.load(imageName, 1, 'byte')
			trainLabels[trainShuffle[iTrain]] = determineClassLabelSimple(labels[i])
			iTrain = iTrain + 1
		end
	end

	return trainData, testData, trainLabels, testLabels
end

--function dataset.loadTrainSet(percent2Load)
--	dataNames, classLabels = loadLabelsDataNames(percent2Load)
--	numOfExamples = math.floor(#classLabels * percent2Load * dataset.TRAIN_SPLIT)
--	data, labels = loadSet(dataNames, classLabels, numOfExamples, 0)
--	return data, labels
--end
--
--function dataset.loadTestSet(percent2Load)
--	dataNames, classLabels = loadLabelsDataNames(percent2Load)
--	numOfExamples = math.floor(#classLabels * percent2Load * (1-dataset.TRAIN_SPLIT))
--	offset = math.floor(#classLabels * percent2Load * dataset.TRAIN_SPLIT)
--	data, labels = loadSet(dataNames, classLabels, numOfExamples, offset)
--	return data, labels
--end

-- torch.ByteTensor does not implement the torch.mean() function
function dataset.normalize(data)
	data = data:type(torch.getdefaulttensortype())
	local mean = torch.mean(data)
	local std = torch.std(data)
	data:add(-mean)
	data:mul(1/std)
	return mean, std
end

--trainData, testData, trainLabels, testLabels = dataset.loadDatasetCustom(1000)