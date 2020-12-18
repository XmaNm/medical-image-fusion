'''
	Create and Collapse Steerable Pyramid.

	Usage:
		python create_collapse_pyramid.py -i saucer-mono256.png -x 256 -y 256
			-i : input image path
			-o : path for output
			-n : depth of steerable pyramid (default:5)
			-k : num of orientations of bandpass filters(default:4)
			-x : resolution axis x (axis=1)
			-y : resolution axis y (axis=0)
		Attn.
		gray scale image (1 channel) only.
		orientation : 4, 6, 8, 10, 12, 15, 18, 20, 30, 60

'''

from PIL import Image
import numpy as np
import sys, os
import argparse
import logging

import steerable_pyramid as steerable

SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
    	description='Steerable Pyramid')
	parser.add_argument('--xres', '-x', default=256, type=int,
                    help='resolution of width (axis=1)')
	parser.add_argument('--yres', '-y', default=256, type=int,
                    help='resolution of height (axis=0)')
	parser.add_argument('--depth', '-n', default=4, type=int,
                    help='Depth of Pyramid. Integer')
	parser.add_argument('--orientation', '-k', default=4, type=int,
                    help='Orientation. Integer')
	parser.add_argument('--input_file', '-i', default='CT.jpg',
	                help='Iuput File Name')
	parser.add_argument('--input_file', '-i', default='PET.jpg',
	                help='Iuput File Name')
	parser.add_argument('--image_name', '-j', default='img', 
                    help='Name of the input image')
	parser.add_argument('--out_dir', '-o', default='outp',
                    help='Output directory')
	parser.add_argument('--verbose', '-v', default=1, type=int,
                    help='verbose')
	args = parser.parse_args()


	im = Image.open(args.input_file).convert('L')
	# create steerable pyramids
	image = np.asarray(im)
	sp = steerable.SteerablePyramid(image, args.xres, args.yres, args.depth, args.orientation,
								 args.image_name, args.out_dir, args.verbose)


	sp.create_pyramids()

# 	create steerable pyramids
	recon = sp.collapse_pyramids()

	f_ishift = np.fft.ifftshift(recon)
	image_recon = np.fft.ifft2(f_ishift)
	image_recon = np.abs(image_recon)
	Image.fromarray(np.uint8(image_recon), mode='L').save('{}/{}-recon-full.png'.format(args.out_dir, args.image_name))

	# Caliculate Frobenius distance between original image and image reconstucted
	dist = np.sqrt(np.sum((image - image_recon)**2))

	LOGGER.info(str(dist))

	sys.exit()
