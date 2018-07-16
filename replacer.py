

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

 
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='replacer!!!')
parser.add_argument("-m", "--mouthDir", type=str, required=True,
	help="cropped mouth image directory")
parser.add_argument("-f", "--faceDir", type=str, required=True,
	help="target head image directory")
parser.add_argument("-o", "--outDir", type=str, required=True,
	help="recomposited output directory")

args = parser.parse_args()

mouth_dir = args.mouthDir
face_dir = args.faceDir
out_dir = args.outDir


mouth_path_list = []
for file in glob.glob('%s/*.png' % mouth_dir):
	mouth_path_list.append(file)
for file in glob.glob('%s/*.jpg' % mouth_dir):
	mouth_path_list.append(file)

#load alignments file
with open("%s/alignments.json" % mouth_dir) as handle:
    info_dict = json.loads(handle.read())

# walk the list if input images, detect images
for i, mouth_path in enumerate(mouth_path_list):
	print (mouth_path)
	#print ('%s/%s' % (face_dir, os.path.basename(mouth_path)))

	mouth_img = cv2.imread(mouth_path)
	face_img = cv2.imread("%s/%s" % (face_dir, os.path.basename(mouth_path)))

	#extractio coordiinate from original kim 
	# figure out how to pull these from the data output of extractor

	coords = info_dict[os.path.basename(mouth_path)][0]
	x,y,h,w = coords

	#resize the added mouth to fit back into it's slot
	mouth_img = cv2.resize(mouth_img, (w, h), interpolation = cv2.INTER_CUBIC)

	#replace the corresponding region of the big image with the small image
	face_img[y:y+mouth_img.shape[0], x:x+mouth_img.shape[1]] = mouth_img
	
	cv2.imwrite('%s/%s' % (out_dir,os.path.basename(mouth_path)), face_img)