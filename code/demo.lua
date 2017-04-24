require 'pl'
require 'trepl'
require 'torch' 
require 'image'
require 'nn'
require 'qtwidget'
require 'sys'

args = lapp[[
   -r,--resultsDir      (default results)       	directory with training results
   -t,--testSet			(default test_set.txt)  	txt file containing image paths
   -m,--modelName		(default model.net)			torch model
]]

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

wind = qtwidget.newwindow(180, 120)

print(sys.COLORS.green .. 'Testing model ' .. args.modelName .. ':\n') 

net = torch.load(args.resultsDir .. '/' .. args.modelName):float()
torch.setnumthreads(1)

imgpaths = get_lines(args.testSet)
for i, p in ipairs(imgpaths) do

	img = image.load(p):float()
	h = img:size()[2]
	w = img:size()[3]
	z = 10.0

	print (img:size())
	print (net)

	local time = sys.clock()
	output = net:forward(img)
	time = sys.clock() - time

	print('x: ' .. output[1] .. ', y: ' .. output[2] .. ', ms: ' .. time * 1000.0)

	bigImg = image.scale(img, w * z, h * z) 

	bigColorImg = torch.Tensor(3, h * z, w * z)
	bigColorImg[1] = bigImg
	bigColorImg[2] = bigImg
	bigColorImg[3] = bigImg

	x = (output[1] * w * z)
	y = (output[2] * h * z)
	resultImg = image.drawRect(bigColorImg, x, y, x, y, {lineWidth = 3, color = {0, 255, 0}})
	image.display{image=resultImg, win=wind} 
	sys.sleep(1)
end