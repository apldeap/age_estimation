import sqlite3
import argparse
from PIL import Image, ImageDraw
from BoundingBox import findBoundingBoxFromDBRows

INPUT_FOLDER_PATH = '/home/bruno/workspace/lua/age estimation/database/age_db/'
DATABASE_NAME = 'db_FaceAnalyser'
OUTPUT_FOLDER_PATH = 'faces_small/'

AGE_MIN = 12
AGE_MAX = 80

ROTATION_X_MAX = 20
ROTATION_Y_MAX = 20
ROTATION_Z_MAX = 20

SCALING_FACTOR = 1.4
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# simple auxiliary method for printing info
def debug(data, message):
	print('>>> ' + str(data) + message)

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='source folder', default=INPUT_FOLDER_PATH, dest='src')
parser.add_argument('--db', help='database', default=DATABASE_NAME, dest='db')

args = parser.parse_args()

folder_src = args.src
dbase = args.db

# DATABASE QUERY
query = 'SELECT * FROM t'
conn = sqlite3.connect(folder_src + dbase)
conn.text_factory = str
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute(query)
conn.commit()
rows = c.fetchall()

debug(len(rows), ' rows fetched')

age_missing = 0
age_out_of_bounds = 0
rotation_x_out_of_bounds = 0
rotation_y_out_of_bounds = 0
rotation_z_out_of_bounds = 0

images_not_opened = []
images_without_face = []

rows_kept = []
i = 0
for r in rows:
	i += 1
	if not i % 100:
		print(i)

	row = {}
	for idx, col in enumerate(c.description):
		row[col[0]] = r[idx]

	row['path_name'] = row['path_name'].replace('\\', '/')	# linux notation

	if not row['Age']:
		age_missing += 1
		continue

	if row['Age'] <= AGE_MIN or row['Age'] >= AGE_MAX:
		age_out_of_bounds += 1
		continue

	if not row['Head_Rotation_value_x'] or not row['Head_Rotation_value_x'] == 'NULL':
		row['Head_Rotation_value_x'] = 0
	if not row['Head_Rotation_value_y'] or not row['Head_Rotation_value_y'] == 'NULL':
		row['Head_Rotation_value_y'] = 0
	if not row['Head_Rotation_value_z'] or not row['Head_Rotation_value_z'] == 'NULL':
		row['Head_Rotation_value_z'] = 0

	if (row['Head_Rotation_value_x'] > ROTATION_X_MAX or row['Head_Rotation_value_x'] < -ROTATION_X_MAX):
		rotation_x_out_of_bounds += 1
		continue
	if (row['Head_Rotation_value_y'] > ROTATION_Y_MAX or row['Head_Rotation_value_y'] < -ROTATION_Y_MAX):
		rotation_y_out_of_bounds += 1
		continue
	if (row['Head_Rotation_value_z'] > ROTATION_Z_MAX or row['Head_Rotation_value_z'] < -ROTATION_Z_MAX):
		rotation_z_out_of_bounds += 1
		continue

	try:
		image = Image.open(folder_src + row['path_name'])

#		if (image.size[0] != IMAGE_WIDTH) or (image.size[1] != IMAGE_HEIGHT):
#			image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

		if max(image.size) > 512:
			width = int(image.size[0] * 512 / max(image.size))
			height = int(image.size[1] * 512 / max(image.size))
			image = image.resize((width, height))

		position, scale = findBoundingBoxFromDBRows(row, image)
		if scale <= 0:
			images_without_face.append(row['path_name'])
			continue
		row['position'], row['scale'] = position, scale
	except:
		images_not_opened.append(row['path_name'])
		continue

	rows_kept.append(row)

debug(age_missing, ' rows are missing age')
debug(age_out_of_bounds, ' rows\' age is out of bounds')
debug(rotation_x_out_of_bounds, ' rows\' x rotation out of bounds')
debug(rotation_y_out_of_bounds, ' rows\' y rotation is out of bounds')
debug(rotation_z_out_of_bounds, ' rows\' z rotation is out of bounds')
debug(len(images_not_opened), ' images not opened')
debug(len(images_without_face), ' images without face')

debug(len(rows_kept), ' rows kept')

csv_file_content = []
csv_file_content.append('file name,path name,age')

csv_file = open('labels/labels.csv', 'w')

try:
	for index in range(len(rows_kept)):
		if not index % 100:
			print(index)

		row = rows_kept[index]
		image = Image.open(folder_src + row['path_name'])

		if max(image.size) > 512:
			width = int(image.size[0] * 512 / max(image.size))
			height = int(image.size[1] * 512 / max(image.size))
			image = image.resize((width, height))
			resized = True

		# crop image
		position = row['position']
		scale = row['scale'] * SCALING_FACTOR
		image = image.crop((position[1] - scale/2, position[0] - scale/2, position[1] + scale/2, position[0] + scale/2)) # prvo y onda x
		image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

		file_name = 'face_' + str(index+1) + '.png'		# indexing file names from 1 as defined in lua
		try:
			image.convert('L').save(OUTPUT_FOLDER_PATH + file_name)
		except IOError as e:
			print('Could not save image ' + file_name)
			continue
#			image.convert('RGB').save(OUTPUT_FOLDER_PATH + file_name, "PNG", optimize=True)

		row['path_name'] = row['path_name'].replace(',', '(comma)')	# dummy escaping as it's not that important
		csv_file_content.append(','.join([file_name, row['path_name'], str(row['Age'])]))
except Exception as ignorable:
	raise ignorable
finally:
	csv_file.write('\n'.join(csv_file_content))
	csv_file.close()