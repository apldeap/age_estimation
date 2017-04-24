require 'optim'

LOGS_PATH = '/home/bruno/workspace/lua/age estimation/logs'
--CURRENT_LOG = '/lr_5e-2__l2_5e-2__poe_1e-2__norm__randBatch_ffn1000'
CURRENT_LOG = ''

TRAIN_LOG_PATH = LOGS_PATH .. CURRENT_LOG .. '/train.log'
TEST_LOG_PATH = LOGS_PATH .. CURRENT_LOG .. '/test.log'

TRAIN_TITLE = 'mean class accuracy (train)'
TEST_TITLE = 'mean class accuracy (test)'

-- see if the file exists
function fileExists(file)
	local f = io.open(file, "rb")
	if f then f:close() end
	return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function linesFrom(file)
	lines = {}
	for line in io.lines(file) do 
		table.insert(lines, line)
	end
	return lines
end

function loadLogger(file, title)
	if not fileExists(file) then
		print('log file missing!')
		return nil
	end

	currentLogger = optim.Logger()

	print('opening file ' .. file)
	lines = linesFrom(file)
	for k,v in pairs(lines) do
		if k ~= 1 then
			currentLogger:add{[title] = tonumber(v)}
		end
	end

	return currentLogger
end


trainLogger = loadLogger(TRAIN_LOG_PATH, TRAIN_TITLE)
testLogger = loadLogger(TEST_LOG_PATH, TEST_TITLE)

trainLogger:style{[TRAIN_TITLE] = '-'}
testLogger:style{[TEST_TITLE] = '-'}
trainLogger:plot()
testLogger:plot()