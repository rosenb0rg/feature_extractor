import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import json
import glob
import os
from math import atan2, degrees, radians
import math
from tqdm import tqdm
import shutil
from utils import *
from char_dir import *
import shutil
from mtcnn.mtcnn import MTCNN

# facial Landmarks dictionary
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

# define the area of the face to extract
face_part = "mouth"

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='extractor!!!')

# parser.add_argument("-i", "--iDir", type=str, required=False, default='in',
# 	help="input directory")
# parser.add_argument("-o", "--oDir", type=str, required=False, default='out',
# 	help="output directory")

parser.add_argument("-s", "--Sce", type=str, required=False, default='in',
	help="scene name")
parser.add_argument("-c", "--Char", type=str, required=False, default='out',
	help="character name")
parser.add_argument("-S", "--Src", type=str, required=False, default='in',
	help="source number")
parser.add_argument("-t", "--Targ", type=str, required=False, default='out',
	help="target number")
parser.add_argument("-m", "--Mode", type=int, required=False, default=0,
	help="mode 0 runs normally, mode 1 will crop faces for training given a character only")
args = parser.parse_args()
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
#detector = dlib.get_frontal_face_detector()


# path for the shape predictor
#predictor = dlib.shape_predictor("C:/local/src/feature_extractor/shape_predictor_68_face_landmarks.dat")

#input and output directories
scene_name = args.Sce
character_name = args.Char
source_number = args.Src
target_number = args.Targ
mode_num = args.Mode



if mode_num == 0:
	character_info = Character(character_name, scene_name, source_number, target_number)
	in_dir = character_info.align_png_dir
	out_dir = character_info.align_crop_dir
else:
	character_info = Character(character_name, scene="00", source=00, target=00)
	in_dir = character_info.imgB_dir
	out_dir = character_info.imgB_crop_dir

if os.path.exists(out_dir):
	shutil.rmtree(out_dir, ignore_errors=True)
	os.makedirs(out_dir)

if not os.path.exists(in_dir):
	os.makedirs(in_dir)

if not os.path.exists(out_dir):
	os.makedirs(out_dir)

save_name = character_info.full_name

print (in_dir, '\n', out_dir)

# in_dir = args.iDir
# out_dir = args.oDir

extensions = ['.png', '.jpg', '.jpeg']

image_path_list = []
# 00100.png
# create a list of all the PNG or JPG images in the input directory
# if file lengh is less than 10 (i.e. hasnt been renamed), then give it a new name w/ correct format
# (renaming function turned off for now - not useful anymore)
for extension in extensions:
	for i, file in enumerate(glob.glob('%s/*%s' % (in_dir, extension))):
		head, tail = os.path.split(file)
		print (head, tail)
		img_num = '{0:03d}'.format(i+1)
		#print (img_num)
		renamed = head + '\\' + save_name + '_' + img_num + extension
		#print ('file:', file, "\n")
		print ('renamed', renamed, '\n')
		shutil.move(file, renamed)
		#image_path_list.append(renamed)

for extension in extensions:
	for i, file in enumerate(glob.glob('%s/*%s' % (in_dir, extension))):
		image_path_list.append(file)

image_path_list = sorted(image_path_list)

print (image_path_list)

# create an empty dictionary for filename, coordinate info
# to be written to a json file for the replacer script on the other side
info_dict = {}

# walk the list of input images, detect images
detector = MTCNN()

for i, image_path in enumerate(image_path_list):
	try:
		# load image and crop out the face with some padding
		image = cv2.imread(str(image_path))	
		results = detector.detect_faces(image)[0]
		#print (results['box'])
		x, y, w, h = results['box'] 

		pad = int(.2*h)
		x -= pad
		w += 2*pad
		y -= pad
		h += 2*pad

		image = image[y:y+h, x:x+w]
		
		# write over the exiting image with the cropped one
		cv2.imwrite(image_path, image)

		# redetect mouth points to extract just the mouth
		mleft = detector.detect_faces(image)[0]['keypoints']['mouth_left']
		mright = detector.detect_faces(image)[0]['keypoints']['mouth_right']
		print (image_path)

		# determine degrees needed to rotate the image to be aligned
		degrees = get_rot_angle(mleft, mright)

		# rotate original image to align
		image_r = rotate_image(image, degrees)

		# detect new mouth points (could replace with math)
		mleft = detector.detect_faces(image_r)[0]['keypoints']['mouth_left']
		mright = detector.detect_faces(image_r)[0]['keypoints']['mouth_right']

		# create coordinates for bounding box
		w = mright[0] - mleft[0]
		h = w
		x = mleft[0]
		y = mleft[1] - h/2
		print (x, y, w, h)

		# add pixels of padding as a percentage of the width
		pad = int(round(.5*w))
		x-=pad
		x=int(x)
		y-=pad
		y=int(y)
		h+=2*pad
		h=int(h)
		w+=2*pad
		w=int(w)

		coords = [x,y,h,w]

		#basename for files
		filename = os.path.basename(image_path)

		# location of the cropped face and source image
		out_path = '%s/%s' % (out_dir, filename)
		in_path = '%s/%s' % (in_dir, filename)

		# add cropped output file path, coordinates and input 
		# file path to dictionary
		info_dict[filename] = coords, degrees, in_path

		# add padding and resize to 128 pixels
		roi = image_r[y:y + h,x:x + w]
		roi = cv2.resize(roi, (128,128))
		#roi = imutils.resize(roi, 128, inter=cv2.INTER_CUBIC)

		# write image
		cv2.imwrite(out_path, roi)
	except:
		pass

try:
	with open('%s/alignments.json' % out_dir, 'w') as outfile:
		json.dump(info_dict, outfile, indent=4)
		outfile.write("\n")
except:
	pass