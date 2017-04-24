from PIL import Image, ImageDraw

image = Image.open('/home/bruno/workspace/lua/age estimation/database/age_db//Age-imgs-detector/16/16+year+old+girl184.jpg')
image.convert('RGB').save(file_name, "PNG", optimize=True)