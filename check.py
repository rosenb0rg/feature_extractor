import os
import glob
from char_dir import *


name = 'judgeC'
scene = 'raupach'
targ = '00'

from_n = 1
to_n = 10

# check to see if there are the same number of OBJ and PNG files
for i in reversed(range(from_n, to_n + 1)):
	i = '{0:03d}'.format(i)
	char = Character(name, scene, i, targ)
	objDir = char.align_obj_dir
	cropDir = char.align_crop_dir
	convDir = char.align_conv_dir
	compedDir = char.align_comped_dir
	swapDir = char.swap_comp_dir

	objCounter = len(glob.glob1(objDir, '*.obj'))
	pngCounter = len(glob.glob1(objDir, '*.png'))
	cropCounter = len(glob.glob1(cropDir, '*.png'))
	convCounter = len(glob.glob1(convDir, '*.png'))
	compedCounter = len(glob.glob1(compedDir, '*.png'))
	swapCounter = len(glob.glob1(swapDir, '*.png'))

	# if not objCounter == pngCounter == cropCounter == convCounter == compedCounter == swapCounter:
	print "\n", objDir
	print "obj:", objCounter
	print "png:", pngCounter
	print "cropped:", cropCounter
	print "converted:", convCounter
	print "comped:", compedCounter
	print "swapped:", swapCounter