require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator

----------------------------------------------------------------------
--Helper functions

function split(s, delimiter)
    result = {};
    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match);
    end
    return result;
end

----------------------------------------------------------------------
--Local vars
local imgDir = './faces'
local imgExt = 'png'
local gtTxt = '_gt.txt'
local imChannels = 1
local imWidth = 256
local imHeight = 256
local numOutputs = 2
local sampleCnt = 0
local opt = opt or {
   visualize = true,
   size = 'small',
   patches = 'all'
}

--os.execute('mkdir ' .. opt.save)

----------------------------------------------------------------------
-- Loading file list from directory

-- Create empty table to store file names:
files = {}
print('Getting list of images..')
-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(imgDir) do
   -- We only load files that match the extension
   if file:find(imgExt .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(imgDir, file))
      sampleCnt = sampleCnt + 1
   end
end

-- Check files
if #files == 0 then
   error('given directory doesnt contain any files of type: ' .. imgExt)
end

print(#files)

-- We sort files alphabetically, it's quite simple with table.sort()
table.sort(files, function (a,b) return a < b end)


----------------------------------------------------------------------
-- Create data tensors
torch.setdefaulttensortype('torch.ByteTensor')
local imagesAll = torch.Tensor(sampleCnt, imChannels, imHeight, imWidth)
local gtAll = torch.Tensor(sampleCnt, numOutputs)

----------------------------------------------------------------------
-- Loading images

-- Go over the file list:
for i,file in ipairs(files) do
   -- load each image
   print(i)
   imagesAll[i] = image.load(file, 1, 'byte')
end