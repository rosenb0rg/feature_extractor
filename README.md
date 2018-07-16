# feature_extractor
Extract and re-composite individual facial features from a photograph

I wrote this script to generate images of facial parts to supply to faceswap-GAN and Deepfakes to experiment with training specific facial regions (i.e. mouth only)

There are two compononents:

extractor.py uses dlib's facial recognition to locate and crop out specificed facial region as a 256x256 square. It will also output a json file in the output directory with the coordinates to realign the image with the original file.

    extractor.py -i [input directory] -o [output directory]

replacer.py takes a cropped mouth image and recomposites it back into the region of the source image from which it was taken.

    replacer.py -m [directory with cropped images and json file] -f [directory with original faces] -o [output directory]
