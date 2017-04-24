import math
import numpy

def fpFromRow(group, index, w, h, row):
    
    fp_x = row['p' + str(group) + '_' + str(index) + '_x']
    fp_y = row['p' + str(group) + '_' + str(index) + '_y']

    if fp_x != None and fp_x != 0 and fp_y != None and fp_y != 0:
        return [fp_x * w, fp_y * h], True
    else:
        return [0, 0], False

def fpDistance(fp1, fp2):
    return math.sqrt((math.pow((fp1[0] - fp2[0]), 2) + math.pow((fp1[1] - fp2[1]), 2)))


def findBoundingBoxFromDBRows(row, image):

	width = image.size[0]
	height = image.size[1]

	leye1, leye1_found = fpFromRow(3,11,width,height,row)
	leye2, leye2_found = fpFromRow(3,7,width,height,row)
	reye1, reye1_found = fpFromRow(3,12,width,height,row)
	reye2, reye2_found = fpFromRow(3,8,width,height,row)
	nose1, nose1_found = fpFromRow(9,1,width,height,row)
	nose2, nose2_found = fpFromRow(9,2,width,height,row)
	mouth, mouth_found = fpFromRow(8,1,width,height,row)

	if not (leye1_found and leye2_found and reye1_found and reye2_found and nose1_found and nose2_found and mouth_found):
		return [-1,-1], -1

	leye = [(leye1[0] + leye2[0])/2, (leye1[1] + leye2[1])/2]
	reye = [(reye1[0] + reye2[0])/2, (reye1[1] + reye2[1])/2]
	nose = [(nose1[0] + nose2[0])/2, (nose1[1] + nose2[1])/2]
	centerEyes = [(leye[0] + reye[0])/2, (leye[1] + reye[1])/2]

	distance = max(fpDistance(leye, mouth), max(fpDistance(reye, mouth), fpDistance(leye, reye)))

	return[nose[1] * 0.8 + centerEyes[1] * 0.2, nose[0] * 0.8 + centerEyes[0] * 0.2], distance * 1.8
