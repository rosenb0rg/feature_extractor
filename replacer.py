

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import json
from os import walk
import glob
import os
from utils import rect_to_bb, shape_to_np, rotate_image, rotate

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='replacer!!!')

parser.add_argument("-m", "--mouthDir", type=str, required=True,
	help="directory with the json file and cropped images")
# parser.add_argument("-f", "--faceDir", type=str, required=True,
# 	help="target head image directory")
parser.add_argument("-c", "--compDir", type=str, required=True,
	help="recomposited output directory")


args = parser.parse_args()

mouth_dir = args.mouthDir
out_dir = args.compDir

#load alignments file
with open("%s/alignments.json" % mouth_dir) as handle:
    info_dict = json.loads(handle.read())



# print (mouth_path_list)

mouth_path_list = []

for key in info_dict:
	mouth_path_list.append(key)

# for file in glob.glob('%s/*.png' % mouth_dir):
# 	mouth_path_list.append(file)
# for file in glob.glob('%s/*.jpg' % mouth_dir):
# 	mouth_path_list.append(file)


# walk the list if cropped mouth images, detect images
for i, mouth_path in enumerate(mouth_path_list):
	print (mouth_path)
	#print ('%s/%s' % (face_dir, os.path.basename(mouth_path)))

	coords = info_dict[(mouth_path)][0]
	degrees = info_dict[(mouth_path)][1]
	face_path = info_dict[(mouth_path)][2]
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
	
	cv2.imwrite('%s/%s' % (out_dir,os.path.basename(mouth_path)), face_img)