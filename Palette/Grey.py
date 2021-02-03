import argparse
import numpy as np
import cv2
import os


def holder_mean(inputs, p=1):
    assert p > 0
    return np.linalg.norm(inputs, ord=p) / len(inputs) ** (1. / p)


def gradient(image, d=1, verbose=False):
    if d == 0:
        return image
    if d == 2:
        return cv2.Laplacian(image, cv2.CV_32F)
    assert len(image.shape) == 2
    for i in range(d):
        x = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=-1))
        y = cv2.convertScaleAbs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=-1))
        d_img = cv2.addWeighted(x, 0.5, y, 0.5, 0)
    if verbose:
        cv2.imwrite("derivativeimg{}.jpg".format(d), d_img)
    return d_img


def angular_error(v1, v2):
    return np.arccos(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))


# np.inf for max
def shades_of_grey(image, derivative=0, lp_order=6, gaussian_size=5, verbose=False, ref_img=None):
    assert lp_order > 0 and len(image.shape) == 3 and image.shape[2] == 3
    smoothed_im = cv2.GaussianBlur(image, (gaussian_size, gaussian_size), 0) if (gaussian_size > 1) else image
    corrected_im = np.zeros_like(image).astype(float)
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    for c in range(3):
        mean = holder_mean(np.ndarray.flatten(gradient(smoothed_im[:, :, c], d=derivative)), p=lp_order)
        corrected_im[:, :, c] = image[:, :, c] / mean
    if verbose and ref_img is not None:
        ange = angular_error(np.ndarray.flatten(ref_img), np.ndarray.flatten(corrected_im))
        print("Param d={}, p={}, sig={} | Angular error = {}".format(derivative, lp_order, gaussian_size, ange))

    corrected_im = cv2.normalize(corrected_im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return corrected_im.astype(np.uint8)


def shades_of_weighed_blocks(image, derivative=0, lp_order=1, gaussian_size=0, grid=(2, 3), verbose=False):
    corrected_im = np.zeros_like(image, dtype=float)
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    smoothed_im = cv2.GaussianBlur(image, (gaussian_size, gaussian_size), 0) if (gaussian_size > 1) else image
    im_height, im_width,  _ = image.shape
    step_y = int(im_height // grid[1] + 1)
    step_x = int(im_width // grid[0] + 1)
    means = np.zeros((grid[0], grid[1], 3))
    stds = np.zeros((grid[0], grid[1], 3))
    a_factors = np.zeros(3)
    for c in range(3):
        for y in range(0, grid[1]):
            for x in range(0, grid[0]):
                tile = smoothed_im[y * step_y:min((y + 1) * step_y, im_height),
                       x * step_x:min((x + 1) * step_x, im_width), c]
                if verbose:
                    cv2.imwrite("tile{}{}.jpg".format(x, y),
                                cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                means[x, y, c] = holder_mean(np.ndarray.flatten(gradient(tile, d=derivative)), p=lp_order)
                stds[x, y, c] = np.std(np.ndarray.flatten(tile))
    stds = np.reshape(stds, (-1, 3))
    means = np.reshape(means, (-1, 3))
    for c in range(3):
        sum_std = sum(stds[:, c])
        a_factors[c] = np.dot(stds[:, c], means[:, c]) / sum_std
    a_sum = sum(a_factors)
    for c in range(3):
        corrected_im[:, :, c] = image[:, :, c] * a_sum / 3 / a_factors[c]

    corrected_im = cv2.normalize(corrected_im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return corrected_im.astype(np.uint8)


parser = argparse.ArgumentParser(description='RGB Colour Constancy - Generalized Grey World')
parser.add_argument('-i', '--input', default='../assets/wrong_white_balance', type=str)
parser.add_argument('-o', '--output', default="../vault/grey", type=str)


def run_grey():
    args = parser.parse_args()
    directory_assets = args.input
    assets = [f for f in os.listdir(directory_assets) if os.path.isfile(os.path.join(directory_assets, f))]

    for f in assets:
        im = cv2.imread(os.path.join(args.input, f))
        for (d, p, sigma) in [(0, 1, 0), (0, np.inf, 0), (0, 13, 3), (1, 1, 5), (2, 1, 5)]:
        # for d in range(3):
        #     for sigma in [0, 5, 15]:
        #         for p in [1, 6, 15, np.inf]:
                    out_name = f[:-4] + '_grey_d{}p{}s{}.jpg'.format(d, p, sigma)
                    cv2.imwrite(os.path.join(args.output, out_name),
                                shades_of_weighed_blocks(im, d, p, sigma))


if __name__ == '__main__':
    run_grey()
