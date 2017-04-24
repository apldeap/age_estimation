require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset'
require 'pl'
require 'paths'
require 'cutorch'
require 'cunn'
require 'cudnn'


-- set customs seed
torch.manualSeed(1)
--math.randomseed(1)

cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = false

-- network hyper-parameters
PARAM_LEARNING_RATE = 0.0003
PARAM_BATCH_SIZE = 32
PARAM_MOMENTUM = 0
PARAM_MAX_ITER = 10000
PARAM_L1_COEF = 0
PARAM_L2_COEF = 0
PARAM_DROUPOUT = 0.35

USE_STAND = false
USE_NORM = true

NUM_OF_THREADS = 4

PERCENT_OF_EXAMPLES = 1  -- portion of examples to be used, 1 to use whole database
RUN_ON_GPU = true
CLASSIFICATION = false


function generateRandomName()
   math.randomseed(os.time())
   local fileName = 'network-'
   local rest = {}
   for i = 1, 20 do
      rest[i] = string.char(math.random(97,122))
   end
   math.randomseed(1)
   return fileName .. table.concat(rest) .. '.net'
end

RANDOM_NAME = generateRandomName()

LOGS_PATH = '/home/bruno/workspace/lua/age estimation/logs'
NETWORKS_PATH = '/home/bruno/workspace/lua/age estimation/networks/' .. generateRandomName()
print('random name: ' .. NETWORKS_PATH)

-- set threads
torch.setnumthreads(NUM_OF_THREADS)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

classes = {'1','2','3','4','5','6','7','8','9','10'}


-- define model to train                                                -- OUTPUT DIMENSIONS
model = nn.Sequential()                                                 -- (1 x 128 x 128)

-- stage 1
model:add(nn.SpatialConvolutionMM(1, 16, 15, 15, 1, 1, 7, 7))           -- (16 x 128 x 128)
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(4, 4, 2, 2, 2, 2))                       -- (16 x 65 x 65)
--model:add(nn.Dropout(PARAM_DROUPOUT))
-- stage 2
model:add(nn.SpatialConvolutionMM(16, 32, 13, 13, 1, 1, 6, 6))          -- (32 x 65 x 65)
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3, 3, 4, 4, 1, 1))                        -- (32 x 17 x 17)
--model:add(nn.Dropout(PARAM_DROUPOUT))
-- stage 3
model:add(nn.SpatialConvolutionMM(32, 64, 5, 5, 1, 1, 2, 2))           -- (64 x 17 x 17)
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))                       -- (64 x 9 x 9)
--model:add(nn.Dropout(PARAM_DROUPOUT))
-- stage 4
model:add(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 1, 1))            -- (128 x 9 x 9)
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))                       -- (128 x 5 x 5)
-- stage 5 : standard 2-layer MLP:
model:add(nn.Reshape(128*5*5))                                           -- (1600)
model:add(nn.Linear(128*5*5, 1000))                                       -- (200)
model:add(nn.Dropout(PARAM_DROUPOUT))
model:add(nn.Tanh())
if CLASSIFICATION then
   model:add(nn.Linear(1000, 1))                                            -- (10)
   model:add(nn.LogSoftMax())
else
   model:add(nn.Linear(1000, 1))                                            -- (1)
end

---- define model to train                                                -- OUTPUT DIMENSIONS
--model = nn.Sequential()                                                 -- (1 x 256 x 256)
--
---- stage 1
--model:add(nn.SpatialConvolutionMM(1, 32, 15, 15, 1, 1, 7, 7))           -- (32 x 256 x 256)
--model:add(nn.Tanh())
--model:add(nn.SpatialMaxPooling(4, 4, 4, 4, 2, 2))                       -- (32 x 65 x 65)
--model:add(nn.Dropout(PARAM_DROUPOUT))
---- stage 2
--model:add(nn.SpatialConvolutionMM(32, 32, 13, 13, 1, 1, 6, 6))          -- (32 x 65 x 65)
--model:add(nn.Tanh())
--model:add(nn.SpatialMaxPooling(3, 3, 4, 4, 1, 1))                        -- (32 x 17 x 17)
--model:add(nn.Dropout(PARAM_DROUPOUT))
---- stage 3
--model:add(nn.SpatialConvolutionMM(32, 32, 5, 5, 1, 1, 2, 2))           -- (32 x 17 x 17)
--model:add(nn.Tanh())
--model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))                       -- (32 x 9 x 9)
--model:add(nn.Dropout(PARAM_DROUPOUT))
---- stage 4
--model:add(nn.SpatialConvolutionMM(32, 64, 3, 3, 1, 1, 1, 1))            -- (64 x 9 x 9)
--model:add(nn.Tanh())
--model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))                       -- (64 x 5 x 5)
--model:add(nn.Dropout(PARAM_DROUPOUT))
---- stage 5 : standard 2-layer MLP:
--model:add(nn.Reshape(64*5*5))                                           -- (1600)
--model:add(nn.Linear(64*5*5, 1000))                                       -- (200)
--model:add(nn.Tanh())
--model:add(nn.Dropout(PARAM_DROUPOUT))
--model:add(nn.Linear(1000, #classes))                                            -- (10)
--model:add(nn.LogSoftMax())

if CLASSIFICATION then
   criterion = nn.ClassNLLCriterion()
else
   criterion = nn.MSECriterion()
end

if RUN_ON_GPU then
   model = model:cuda()
   criterion = criterion:cuda()

   cudnn.convert(model, cudnn)

end

print(model)

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- load data
--print('Loading train data..')
--trainData, trainLabels = dataset.loadTrainSet(PERCENT_OF_EXAMPLES)
--print(trainLabels:size())
--print('Loading test data..')
--testData, testLabels = dataset.loadTestSet(PERCENT_OF_EXAMPLES)

print('Loading data..')
trainData, testData, trainLabels, testLabels = dataset.loadDatasetTenClass(PERCENT_OF_EXAMPLES, CLASSIFICATION)

confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(LOGS_PATH, 'train-' .. RANDOM_NAME .. '.log'))
testLogger = optim.Logger(paths.concat(LOGS_PATH, 'test-' .. RANDOM_NAME .. '.log'))

inputs = torch.Tensor(PARAM_BATCH_SIZE, dataset.IMAGE_CHANELS, dataset.IMAGE_WIDTH, dataset.IMAGE_HEIGHT):type('torch.DoubleTensor')
targets = torch.Tensor(PARAM_BATCH_SIZE):type('torch.DoubleTensor')
if RUN_ON_GPU then
   inputs = inputs:cuda()
   targets = targets:cuda()
end

-- training function
function train(data, labels)
   -- epoch tracker
   epoch_num = epoch_num or 1

   -- local vars
   local time = sys.clock()

   torch.setdefaulttensortype('torch.FloatTensor')
   local shuffle = torch.randperm(data:size(1))
   torch.setdefaulttensortype('torch.ByteTensor')

   -- do one epoch
   for t = 1, data:size(1), PARAM_BATCH_SIZE do

      if (data:size(1) - t + 1) < PARAM_BATCH_SIZE then
         break    -- for now..
      end

      -- create mini batch
      local k = 1
      for i = t, math.min(t + PARAM_BATCH_SIZE-1, data:size(1)) do
         local input = data[shuffle[i]]:clone()
         local label = labels[shuffle[i]]
         inputs[k] = input
         targets[k] = label
         k = k + 1
      end

      -- standardization
      if USE_STAND then
         local mean = inputs:mean()
         local std = inputs:std()
         inputs:add(-mean)
         inputs:mul(1/std)
      end

      -- normalization
      if USE_NORM then
--         local min_value = inputs:min()
--         local span = inputs:max() - min_value
--         inputs:add(-min_value)
--         inputs:mul(1/span)
         inputs:mul(1/256)    -- as these are all bytes
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)

         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if PARAM_L1_COEF ~= 0 or PARAM_L2_COEF ~= 0 then
            -- locals:
            local norm, sign = torch.norm, torch.sign

            -- Loss:
            f = f + PARAM_L1_COEF * norm(parameters,1)
            f = f + PARAM_L2_COEF * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(PARAM_L1_COEF) + parameters:clone():mul(PARAM_L2_COEF) )
         end

--         -- update confusion
--         for i = 1,PARAM_BATCH_SIZE do
--            confusion:add(outputs[i], targets[i])
--         end

         -- return f and df/dX
         return f, gradParameters
      end

      -- Perform SGD step:
      sgdState = sgdState or {
         learningRate = PARAM_LEARNING_RATE,
         momentum = PARAM_MOMENTUM,
         learningRateDecay = 5e-7
      }

      optim.sgd(feval, parameters, sgdState)

      -- disp progress
      xlua.progress(t, data:size(1))

   end
   
   -- time taken
   time = sys.clock() - time
   time = time / data:size(1)
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   collectgarbage()
   print('Computing train error..')

   local trainError = 0.0

   -- compute train error
   for t = 1, data:size(1), PARAM_BATCH_SIZE do
      if (data:size(1) - t + 1) < PARAM_BATCH_SIZE then
         break    -- for now..
      end
      -- create mini batch
      local k = 1
      for i = t, math.min(t + PARAM_BATCH_SIZE-1, data:size(1)) do
         local input = data[i]:clone() -- memory problems?
         local label = labels[i]
         inputs[k] = input
         targets[k] = label
         k = k + 1
      end

      -- standardization
      if USE_STAND then
         local mean = inputs:mean()
         local std = inputs:std()
         inputs:add(-mean)
         inputs:mul(1/std)
      end

      -- normalization
      if USE_NORM then
--         local min_value = inputs:min()
--         local span = inputs:max() - min_value
--         inputs:add(-min_value)
--         inputs:mul(1/span)
         inputs:mul(1/256)    -- as these are all bytes
      end

      -- evaluate function for complete mini batch
      --print(inputs)
      local outputs = model:forward(inputs)

      -- update confusion
      if CLASSIFICATION then
         for i = 1, PARAM_BATCH_SIZE do
            confusion:add(outputs[i], targets[i])
         end      
      else
         trainError = trainError + criterion:forward(outputs, targets)
      end

      -- disp progress
      xlua.progress(t, data:size(1))
   end

   -- print confusion matrix
   if CLASSIFICATION then
      print(confusion)
      print('>>> current train result: ' .. confusion.totalValid .. '\n')
      trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
      confusion:zero()
   else
      trainError = trainError / data:size(1)
      print('\n>>> train error: ' .. trainError .. '\n')
      trainLogger:add{['% mean class accuracy (train set)'] = trainError}
   end

   -- save/log current net
-- local filename = paths.concat(opt.save, 'mnist.net')
-- os.execute('mkdir -p ' .. sys.dirname(filename))
-- if paths.filep(filename) then
--    os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
-- end
-- print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

end

-- test function
function test(data, labels)
   -- local vars
   local time = sys.clock()

   local testError = 0.0

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1, data:size(1), PARAM_BATCH_SIZE do
      if (data:size(1) - t + 1) < PARAM_BATCH_SIZE then
         break    -- for now..
      end

      -- disp progress
      xlua.progress(t, data:size(1))
      local k = 1
      for i = t, math.min(t+PARAM_BATCH_SIZE-1, data:size(1)) do
         local input = data[i]:clone() -- memory problems?
         local label = labels[i]
         inputs[k] = input
         targets[k] = label
         k = k + 1
      end

      -- standardization
      if USE_STAND then
         local mean = inputs:mean()
         local std = inputs:std()
         inputs:add(-mean)
         inputs:mul(1/std)
      end

      -- normalization
      if USE_NORM then
--         local min_value = inputs:min()
--         local span = inputs:max() - min_value
--         inputs:add(-min_value)
--         inputs:mul(1/span)
         inputs:mul(1/256)    -- as these are all bytes
      end

      -- test samples
      local preds = model:forward(inputs)

      -- update confusion
      if CLASSIFICATION then
         for i = 1, PARAM_BATCH_SIZE do
            confusion:add(preds[i], targets[i])
         end      
      else
         testError = testError + criterion:forward(preds, targets)
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / data:size(1)
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   if CLASSIFICATION then
      print(confusion)
      if confusion.totalValid > bestTotalValid then
         bestTotalValid = confusion.totalValid
         print('\n>>> NEW BEST TEST RESULT: ' .. bestTotalValid .. '\n')
         torch.save(NETWORKS_PATH, model)
      else
         print('>>> current test result: ' .. confusion.totalValid ..' | best test result: ' .. bestTotalValid .. '\n')
      end
      testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
      confusion:zero()
   else
      testError = testError / data:size(1)
      if testError < bestTotalValid then
         bestTotalValid = testError
         print('\n>>> NEW BEST TEST ERROR: ' .. bestTotalValid .. '\n')
         torch.save(NETWORKS_PATH, model)
      else
         print('>>> current test error: ' .. testError ..' | best test error: ' .. bestTotalValid .. '\n')
      end
      testLogger:add{['% mean class accuracy (test set)'] = testError}
   end
end

-- write params and model to logs folder
params = {
   PARAM_LEARNING_RATE = PARAM_LEARNING_RATE,
   PARAM_BATCH_SIZE = PARAM_BATCH_SIZE,
   PARAM_MOMENTUM = PARAM_MOMENTUM,
   PARAM_MAX_ITER = PARAM_MAX_ITER,
   PARAM_L1_COEF = PARAM_L1_COEF,
   PARAM_L2_COEF = PARAM_L2_COEF,
   USE_STAND = USE_STAND,
   USE_NORM = USE_NORM,
   NUM_OF_THREADS = NUM_OF_THREADS,
   PERCENT_OF_EXAMPLES = PERCENT_OF_EXAMPLES,
   RUN_ON_GPU = RUN_ON_GPU,
   PARAM_DROUPOUT = PARAM_DROUPOUT,
   CLASSIFICATION = CLASSIFICATION,
   model = model
}
paramsString = ''
for i,v in pairs(params) do
   paramsString = paramsString .. tostring(i) .. ' = ' .. tostring(v) .. '\n'
end
paramsOutput = io.open(LOGS_PATH .. '/params-' .. RANDOM_NAME .. '.txt', 'w')
paramsOutput:write(paramsString)
paramsOutput:close()


epoch_num = 1
if CLASSIFICATION then
   bestTotalValid = 0.0
else
   bestTotalValid = 1000000
end
while true do

   print('_________________________________________________\nepoch num: ' .. epoch_num .. '\n')

   -- train/test
   train(trainData, trainLabels)
   test(testData, testLabels)

   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
--   trainLogger:plot()    as it keeps crashing
--   testLogger:plot()

   if epoch_num == PARAM_MAX_ITER then break end
   epoch_num = epoch_num + 1
end