import os
import shutil

x = os.listdir('.')
x.remove('a.py')

for i in x:
	shutil.move(i+'/content_stylized_style.jpg', i+'.jpg')
