# feature_extractor
Extract and re-composite individual facial features from a photograph

I wrote this script to generate images of facial parts to supply to faceswap-GAN and Deepfakes to experiment with training specific facial regions (i.e. mouth only)

There are two compononents:

extractor.py uses dlib's facial recognition to locate and crop out specificed facial region as a 256x256 square in a folder of face images. It will also output a json file with the coordinates to realign the cropped image with the original file.

    extractor.py -i [input directory] -o [output directory]

replacer.py takes a folder of cropped mouth images and recomposites them back into the region of the source images from which they were taken.

    replacer.py -m [directory with cropped images and json file] -f [directory with original faces] -o [output directory]
