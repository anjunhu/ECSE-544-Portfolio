import argparse
import os
import numpy as np

from Grey import *
from Gamut import *

parser = argparse.ArgumentParser(description='ECSE 544 Portfolio')
# Datasets
parser.add_argument('-i', '--input-image', default='assets/slides_example/slides_sky.JPG', type=str)
parser.add_argument('-o', '--output', default="vault/wb_{}_{}_{}_{}", type=str)
parser.add_argument('-t', '--task', default="gamut", type=str)



def main():
    args = parser.parse_args()
    im = cv2.imread(args.input_image)
    output_filename = args.output

    # cv2.imwrite(output_filename.format(args.input_image[:-4] + '.jpg'),
    #             shades_of_weighed_blocks(im))

    canonical_gamut = Gamut()
    canonical_gamut.train(directory_assets="assets/gamut_canonical", pretrained_hull="vault/hull.pkl")
    canonical_gamut.test(directory_assets="assets/gamut_test")

    # for d in range(3):
    #     for sigma in [0, 5, 15]:
    #         for p in [1, 6, 15, np.inf]:
    #             cv2.imwrite(output_filename.format(d, p, sigma, args.input_image[:-4]+'.jpg'),
    #                         shades_of_grey(im, d, p, sigma))

    # for (d, p, sigma) in [(0, 1, 0), (0, np.inf, 0), (0, 13, 3), (1, 1, 5), (2, 1, 5)]:
    #     cv2.imwrite(output_filename.format(d, p, sigma, args.input_image[:-4]+'.jpg'),
    #                         shades_of_grey(im, d, p, sigma))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
