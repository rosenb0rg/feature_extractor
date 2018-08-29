

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import json
import os
from os import walk
import glob
from utils import rect_to_bb, shape_to_np, rotate_image, rotate
from char_dir import *
import shutil
# from mtcnn.mtcnn import MTCNN
# import face_recognition

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='replacer!!!')

# parser.add_argument("-m", "--mouthDir", type=str, required=True,
# 	help="directory with the json file and cropped images")
# # parser.add_argument("-f", "--faceDir", type=str, required=True,
# # 	help="target head image directory")
# parser.add_argument("-c", "--compDir", type=str, required=True,
# 	help="recomposited output directory")


parser.add_argument("-s", "--Sce", type=str, required=False, default='in',
	help="scene name")
parser.add_argument("-c", "--Char", type=str, required=False, default='out',
	help="character name")
parser.add_argument("-S", "--Src", type=str, required=False, default='in',
	help="source number")
parser.add_argument("-t", "--Targ", type=str, required=False, default='out',
	help="target number")
args = parser.parse_args()

scene_name = args.Sce
character_name = args.Char
source_number = args.Src
target_number = args.Targ

character_info = Character(character_name, scene_name, source_number, target_number)

mouth_dir = character_info.align_conv_dir
out_dir = character_info.align_comped_dir
alignemnts_dir = character_info.align_crop_dir

print("\nmouths:", mouth_dir, "\nsave to:", out_dir, "\nalignments:", alignemnts_dir)

if os.path.exists(out_dir):
	shutil.rmtree(out_dir, ignore_errors=True)
	os.makedirs(out_dir)

if not os.path.exists(out_dir):
	os.makedirs(out_dir)

#load alignments file
with open("%s\\alignments.json" % alignemnts_dir) as handle:
    info_dict = json.loads(handle.read())


# print (mouth_path_list)

mouth_path_list = []

# for key in info_dict:
# 	mouth_path_list.append(key)

for file in glob.glob('%s/*.png' % mouth_dir):
	mouth_path_list.append(file)
for file in glob.glob('%s/*.jpg' % mouth_dir):
	mouth_path_list.append(file)

#print ('ping', mouth_path_list)

# walk the list if cropped mouth images, detect images
for i, mouth_path in enumerate(mouth_path_list):
	mouth_path_base = os.path.basename(mouth_path)

	#print ('%s/%s' % (face_dir, os.path.basename(mouth_path)))

	coords = info_dict[(mouth_path_base)][0]
	degrees = info_dict[(mouth_path_base)][1]
	face_path = info_dict[(mouth_path_base)][2]
	x,y,h,w = coords

	mouth_img = cv2.imread(mouth_path)
	face_img = cv2.imread(face_path)

	#rotate face image to
	face_img = rotate_image(face_img, degrees)

	#resize the added mouth to fit back into it's slot
	mouth_img = cv2.resize(mouth_img, (w, h), interpolation = cv2.INTER_CUBIC)

	#replace the corresponding region of the big image with the small image
	face_img[y:y+mouth_img.shape[0], x:x+mouth_img.shape[1]] = mouth_img
	face_img = rotate_image(face_img, -1 * degrees)
	# gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

	# face_locations = face_recognition.face_locations(face_img)
	# top, right, bottom, left = face_locations[0]
	# height = top - bottom
	# pad = int(.2*height)
	# top += pad
	# bottom -= pad 
	# right -= pad
	# print (face_locations)
	# face_img = face_img[top:bottom, left:right]
	
	out_file = str('%s/%s' % (out_dir,os.path.basename(mouth_path)))
	out_file = os.path.abspath(out_file)

	print (mouth_path, '\nis being composited and put here\n', out_file, '\n')

	cv2.imwrite(out_file, face_img)

# for i in glob.glob(out_dir + )
	# result = detector.detect_faces(face_img)
	# print (result)
	# face_box = result[0]['box']
	# fx, fy, fw, fh = face_box
	# print (fy, fy+fh, fx, fx+fw)
	# pad = .2*fh
	# pad = int(pad/2)
	# print ('pad', pad)
	# fy -= pad
	# fh += 2*pad
	# print (fy, fy+fh, fx, fx+fw)
	# face_img = face_img[fy:fy+fh, fx:fx+fw]