

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

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width, color = mat.shape
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser(description='replacer!!!')
parser.add_argument("-m", "--mouthDir", type=str, required=True,
	help="cropped mouth image directory")
# parser.add_argument("-f", "--faceDir", type=str, required=True,
# 	help="target head image directory")
parser.add_argument("-o", "--outDir", type=str, required=True,
	help="recomposited output directory")

args = parser.parse_args()

mouth_dir = args.mouthDir
# face_dir = args.faceDir
out_dir = args.outDir

#load alignments file
with open("%s/alignments.json" % mouth_dir) as handle:
    info_dict = json.loads(handle.read())

mouth_path_list = []
for key in info_dict:
	mouth_path_list.append(key)

print (mouth_path_list)

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