require 'pl'
require 'trepl'
require 'torch' 
require 'image'
require 'nn'
require 'qtwidget'
require 'sys'

NETWORKS_PATH = '/home/bruno/lua/age_estimation/networks'
MODEL_FILE_NAME = 'network-gsanmeotkgnhqvkitxwg.net'
TEST_FILE_PATH = 'test_images_paths.txt'
TEST_IMAGES_PATH = '/home/bruno/lua/age_estimation/test data'

args = lapp[[
   -r,--resultsDir      (default results)       	directory with training results
   -t,--testSet			(default test_set.txt)  	txt file containing image paths
   -m,--modelName		(default model.net)			torch model
]]

PARAM_Z = 10.0

function get_lines(file)
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end 

function round(num, numDecimalPlaces)
  local mult = 10^(numDecimalPlaces or 0)
  return math.floor(num * mult + 0.5) / mult
end

--wind = qtwidget.newwindow(180, 120)
wind = qtwidget.newwindow(128, 128)

--model = torch.load(NETWORKS_PATH .. '/' .. MODEL_FILE_NAME):float()	-- float?
model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1, 16, 15, 15, 1, 1, 7, 7))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(4, 4, 2, 2, 2, 2))
model:add(nn.Reshape(16*65*65))
model:add(nn.Linear(16*65*65, 1))

print(model)
torch.setnumthreads(1)

--input = torch.Tensor(1, 128, 128)

imgpaths = get_lines(TEST_IMAGES_PATH .. '/' .. TEST_FILE_PATH)
for i, p in ipairs(imgpaths) do

	local img = image.load(TEST_IMAGES_PATH .. '/' .. p, 1, 'double')
	local h = img:size()[2]
	local w = img:size()[3]

--	print(img:size())

	local time = sys.clock()
	local output = model:forward(img)
	time = sys.clock() - time

--	print(output)
	print('age: ' .. output[1] .. ', ms: ' .. time * 1000.0)

	image.display{image=img, win=wind} 
	sys.sleep(3)
end