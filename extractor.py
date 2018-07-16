

# import the necessary packages
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict
import json
import glob
import os


# Facial Landmarks dictionary
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

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

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
parser = argparse.ArgumentParser(description='extractor!!!')
parser.add_argument("-i", "--iDir", type=str, required=True,
	help="input directory")
parser.add_argument("-o", "--oDir", type=str, required=True,
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

image_path_list = []
# create a list of all the PNG or JPG images in the input directory
for file in glob.glob('%s/*.png' % in_dir):
	image_path_list.append(file)
for file in glob.glob('%s/*.jpg' % in_dir):
	image_path_list.append(file)

# create an empty dictionary for filename, coordinate info
# to be written to a json file for the replacer script on the other side
info_dict = {}

# walk the list if input images, detect images
for i, image_path in enumerate(image_path_list):

	# load the input image, and convert it to grayscale (dont resize)
	image = cv2.imread(str(image_path))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	 
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		#just looking at the mouth region (i.e. points 48 to 68)
		i = FACIAL_LANDMARKS_IDXS[face_part][0]
		j = FACIAL_LANDMARKS_IDXS[face_part][1]

		# extract the ROI of the face region as a separate image
		# make it a square based on widht of the mouth
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
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

		coords =[x,y,h,w]
		filename = os.path.basename(image_path)
		out_path='%s/crop_%s' % (out_dir,os.path.basename(image_path))

		#add filenae, coordinates and outputh to dictionary
		info_dict['%s' % filename] = coords, out_path

		# add padding and resize to 256 pixels
		roi = image[y:y + h,x:x + w]
		roi = imutils.resize(roi, 256, inter=cv2.INTER_CUBIC)

		# write image
		cv2.imwrite(out_path, roi)
		cv2.waitKey(0)


with open('%s/alignments.json' % out_dir, 'w') as outfile:
   	json.dump(info_dict, outfile)
