#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

TR notes: I added in a loop so this takes input directory A and B and output directory as arguments.

"""

import cv2
import dlib
import numpy
import os
import argparse
import sys
from glob import glob
from os import walk
import glob
from tqdm import tqdm
from char_dir import *
import shutil
from mtcnn.mtcnn import MTCNN


PREDICTOR_PATH = "C:/local/src/PRnet/shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 5

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.  
#ALIGN_POINTS = (MOUTH_POINTS + NOSE_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS)

# All
ALIGN_POINTS = (FACE_POINTS + RIGHT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_POINTS)
#ALIGN_POINTS = (RIGHT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_POINTS)
#ALIGN_POINTS = (MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS + FACE_POINTS + JAW_POINTS
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 1

detector = dlib.get_frontal_face_detector()
# detectorb = MTCNN()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
	points=points.astype(numpy.int32)
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def read_im_and_landmarks(image):
    # im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = image
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

# im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
# im2, landmarks2 = read_im_and_landmarks(sys.argv[2])

# M = transformation_from_points(landmarks1[ALIGN_POINTS],
#                                landmarks2[ALIGN_POINTS])

# mask = get_face_mask(im2, landmarks2)
# warped_mask = warp_im(mask, M, im1.shape)
# combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
#                           axis=0)

# warped_im2 = warp_im(im2, M, im1.shape)
# warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

# output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

# cv2.imwrite('%s/o_%s' % (sys.argv[3], sys.argv[1]), output_im)


def main(args):
    # assign command line arguments to variables
    scene_name = args.scene
    character_name = args.character
    source_number = args.source
    target_number = args.target

    # get character info from arguments
    character_info = Character(character_name, scene_name, source_number, target_number)

    # assign input and output directories
    head_folder = character_info.swap_head_dir
    face_folder = character_info.align_comped_dir
    save_folder = character_info.swap_comp_dir
    print ('\nhead folder', head_folder, '\nface folder:', face_folder, '\nsave_folder:', save_folder)

    # make sure there is a place to save the output!
    # if os.path.exists(save_folder):
    #     shutil.rmtree(save_folder, ignore_errors=True)
    #     os.makedirs(save_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # make an ordered list of all the target head files
    head_path_list = []
    for file in glob.glob("%s\\*.png" % head_folder):
        head_path_list.append(file)
    for file in glob.glob("%s\\*.jpg" % head_folder):
        head_path_list.append(file)
    head_path_list=sorted(head_path_list)
    #print (head_path_list)

    # make an ordered list of all the faces to composite
    face_path_list = []
    for file in glob.glob("%s\\*.png" % face_folder):
        face_path_list.append(file)
    for file in glob.glob("%s\\*.jpg" % face_folder):
        face_path_list.append(file)
    face_path_list=sorted(face_path_list)
    # print (face_path_list)
    b = len(face_path_list)

    for i, image_path in tqdm(enumerate(head_path_list)):
        if i<b:
            try:
                print (i)
                head_file = head_path_list[i]
                face_file = face_path_list[i]
                head_base = (head_file.split('\\')[-1:][0]).strip()
                face_base = (face_file.split('\\')[-1:][0]).strip()
                print ('\nhead:', head_file,'\nface:', face_file,'\n')

                # read the image files for opencv
                head_img = cv2.imread(head_file, cv2.IMREAD_COLOR)
                face_img = cv2.imread(face_file, cv2.IMREAD_COLOR)

                # use MTCNN to pre-crop face                
                # result = detectorb.detect_faces(face_img)
                # print (result)
                # face_box = result[0]['box']
                # fx, fy, fw, fh = face_box
                # print (fy, fy+fh, fx, fx+fw)
                # pad = .1*fh
                # pad = int(pad/2)
                # print ('pad', pad)
                # fy -= pad
                # fh += 2*pad
                # print (fy, fy+fh, fx, fx+fw)
                # face_img = face_img[fy:fy+fh, fx:fx+fw]

                # perform the transformation 
                im1, landmarks1 = read_im_and_landmarks(head_img)
                im2, landmarks2 = read_im_and_landmarks(face_img)
                M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
                mask = get_face_mask(im2, landmarks2)
                warped_mask = warp_im(mask, M, im1.shape)
                combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)
                warped_im2 = warp_im(im2, M, im1.shape)
                warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
                output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
                # uncomment for no color correction at all
                # output_im = im1 * (1.0 - combined_mask) + warped_im2 * combined_mask
                cv2.imwrite('%s/%s' % (save_folder, face_base), output_im)
            except Exception as e:
                print('exception', e)
        else:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toms face swappp mod')

    # parser.add_argument('-H', '--headDir', default='in_head', type=str,
    #                     help='path to the input directory, where input images are stored.')
    # parser.add_argument('-f', '--faceDir', default='in_face', type=str,
    #                     help='path to the output directory, where results(obj,txt files) will be stored.')
    # parser.add_argument('-o', '--outDir', default='output', type=str,
    #                     help='path to the target vertices directory for mactching orientation')

    # parser.add_argument('-b', '--baseDir', default='D:\\characters', type=str,
    #                     help='where all this shit is located')
    parser.add_argument('-s', '--scene', default='raupach', type=str,
                        help='who is in the witness stand')
    parser.add_argument('-c', '--character', default='raupach', type=str,
                        help='who is talking')
    parser.add_argument('-S', '--source', default='output', type=str,
                        help='source number (which line)')
    parser.add_argument('-t', '--target', default='output', type=str,
                        help='target number (which shot does the mouth go onto)')    
   
    main(parser.parse_args())