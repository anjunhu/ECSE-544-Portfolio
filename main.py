import argparse
import os
import numpy as np

from Grey import *
from Gamut import *

parser = argparse.ArgumentParser(description='ECSE 544 Portfolio')
# Datasets
parser.add_argument('-i', '--input-image', default='portrait.jpg', type=str)
parser.add_argument('-o', '--output', default="vault/wb_{}_{}_{}_{}", type=str)
parser.add_argument('-t', '--task', default="gamut", type=str)



def main():
    args = parser.parse_args()
    im = cv2.imread(args.input_image)
    output_filename = "wbgrids{}" #args.output

    cv2.imwrite(output_filename.format(args.input_image[:-4] + '.jpg'),
                shades_of_weighed_blocks(im))

    canonical_gamut = Gamut()
    canonical_gamut.train()

    for d in range(3):
        for sigma in range(1, 21, 6):
            for p in [1, 6, 15, np.inf]:
                cv2.imwrite(output_filename.format(d, p, sigma, args.image[:-4]+'.jpg'), shades_of_grey(im, d, p, sigma))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
