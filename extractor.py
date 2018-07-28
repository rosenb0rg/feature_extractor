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
from utils import *


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

parser.add_argument("-i", "--iDir", type=str, required=False, default='in',
	help="input directory")
parser.add_argument("-o", "--oDir", type=str, required=False, default='out',
	help="output directory")
args = parser.parse_args()
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()

# path for the shape predictor
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

#input and output directories
in_dir = args.iDir
out_dir = args.oDir

extensions = ['.png', '.jpg', '.jpeg']

image_path_list = []
# create a list of all the PNG or JPG images in the input directory
for extension in extensions:
	for file in glob.glob('%s/*%s' % (in_dir, extension)):
		image_path_list.append(file)

# create an empty dictionary for filename, coordinate info
# to be written to a json file for the replacer script on the other side
info_dict = {}

# walk the list of input images, detect images
for i, image_path in enumerate(image_path_list):

	# load the input image, and convert it to grayscale (dont resize)
	image = cv2.imread(str(image_path))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	 
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# identifier for Nth face in an image
		id = i

		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)
		
		# determine rotation angle needed to align mouth
		degrees = get_rot_angle(shape[48], shape[54])

		# rotate original image to align
		image_r = rotate_image(image , degrees)

		# # eventually: get the center of the enlarged, rotated image
		# # to realign the x,y corner pointer for cropping (bypass second face detection)
		# image_center = get_img_center(image_r)

		# detect facial landmarks in rotated image
		# needs to be edited to work on multi-face images
		gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY) 
		rects = detector(gray, 1)
		shape = predictor(gray, rects[0])
		shape = shape_to_np(shape)

		#just looking at the mouth region (i.e. points 48 to 68)
		q = FACIAL_LANDMARKS_IDXS[face_part][0]
		t = FACIAL_LANDMARKS_IDXS[face_part][1]
		# print ('q', q, 't', t)

		# extract the ROI of the face region as a separate image
		# make it a square based on widht of the mouth
		(x, y, w, h) = cv2.boundingRect(np.array([shape[q:t]]))
		

		# # eventually: rotate x,y points around the image center to reorient ROI box
		# point = [x,y]
		# x,y = rotate(image_center, point, math.radians(degrees))

		# print ('point', point)
		# print ('degrees', degrees)
		# print ('rotated point', x, y)


		boop = (w-h)/2
		# make the box around the mouth a square
		y = y - boop
		h = w

		# pixels of padding as a percentage of the width
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
		info_dict['%s' % (out_path)] = coords, degrees, in_path

		# add padding and resize to 128 pixels
		roi = image_r[y:y + h,x:x + w]
		roi = imutils.resize(roi, 128, inter=cv2.INTER_CUBIC)

		# write image
		cv2.imwrite(out_path, roi)

try:
	with open('%s/alignments.json' % out_dir, 'w') as outfile:
		json.dump(info_dict, outfile, indent=4)
		outfile.write("\n")
except:
	pass
