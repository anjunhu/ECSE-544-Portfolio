import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pickle
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import argparse


def get_perspective_colors(train_img):
    perspective_image = np.einsum("ijk, ij->ijk", train_img[:, :, 1:], np.reciprocal(train_img[:, :, 0] + 0.005))
    points = np.reshape(perspective_image, (-1, 2))
    good_points = np.where(np.max(points, axis=1) < 10)
    points = points[good_points]
    perspective_image = np.stack(
        (np.ones_like(train_img[:, :, 0]), perspective_image[:, :, 0], perspective_image[:, :, 1]), axis=2)
    perspective_image[perspective_image > 4.] = 0.
    cv2.normalize(perspective_image, perspective_image, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    return points, perspective_image


def marinate(obj, filename):
    with open(filename, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def relish(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def sample_within_polygon(poly):
    min_x, min_y, max_x, max_y = poly.bounds
    while True:
        point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if poly.contains(point):
            return np.array([point.x, point.y])


class Gamut(object):
    def __init__(self, verbose=True):
        self.hull = None
        self.verbose = verbose

    def train(self, directory_assets='assets/gamut_canonical', pretrained_hull="vault/hull.pkl"):
        assets = [f for f in os.listdir(directory_assets) if os.path.isfile(os.path.join(directory_assets, f))]
        if pretrained_hull is None:
            collective_points = np.ones((1, 2), dtype=float)  # points in perspective colour space
            for f in assets:
                train_img = cv2.imread(os.path.join(directory_assets, f)).astype(float)
                train_img = cv2.normalize(train_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
                train_img = cv2.GaussianBlur(train_img, (5, 5), 0)
                train_img = cv2.medianBlur(train_img, 5)
                points, _ = get_perspective_colors(train_img)
                collective_points = np.concatenate((collective_points, points))
            self.hull = ConvexHull(collective_points)  # ndarray of floats, shape (npoints, ndim)
            if self.verbose:
                convex_hull_plot_2d(self.hull)
                plt.savefig(fname="perspective_gamut_sfu.jpg")
            marinate(self.hull, "vault/hull.pkl")
        else:
            self.hull = relish(pretrained_hull)
        return self.hull

    def test(self, directory_assets="../assets/gamut_test",
             pretrained_hull="../vault/gamut/hull.pkl", directory_output="../vault/gamut"):
        if pretrained_hull is not None:
            self.hull = relish(pretrained_hull)
        assets = [f for f in os.listdir(directory_assets) if os.path.isfile(os.path.join(directory_assets, f))]
        canonical_verticies = self.hull.points[self.hull.vertices]
        canonical_polygon = Polygon(canonical_verticies)

        for f in assets:
            test_img = cv2.imread(os.path.join(directory_assets, f)).astype(float)
            test_img = cv2.normalize(test_img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            test_img = cv2.medianBlur(test_img, 5)
            test_img = cv2.GaussianBlur(test_img, (5, 5), 0)
            test_img_avg_rgb = np.mean(test_img, axis=(0, 1))
            test_img_max_rgb = np.max(test_img, axis=(0, 1))
            test_img_points, perspective_img = get_perspective_colors(test_img)
            cv2.imwrite(os.path.join(directory_output, "{}_perspective_img.jpg".format(f[:-4])), perspective_img)
            test_hull = ConvexHull(test_img_points)

            observed_vertices = test_hull.points[test_hull.vertices]
            feasible_ks = []
            while len(feasible_ks) < 10:
                vo = observed_vertices[random.randint(0, len(observed_vertices) - 1)]
                A = np.array([[vo[0], 0], [0, vo[1]]])
                if np.linalg.det(A) == 0:
                    continue

                feasible = True
                b = sample_within_polygon(canonical_polygon)
                k_pair = np.linalg.solve(A, b)

                j = 0
                while feasible and j < len(observed_vertices):
                    transform = np.array([[k_pair[0], 0], [0, k_pair[1]]])
                    transformed_point = np.matmul(transform, observed_vertices[j])
                    transformed_point = Point(transformed_point)
                    j += 1
                    if not canonical_polygon.buffer(0.1).contains(transformed_point):
                        feasible = False
                        # print('failed')
                if feasible:
                    feasible_ks.append(k_pair)

            best_ks = max(feasible_ks, key=sum)
            # 0.5545236325390757
            # 0.6279112069184949
            c = 0.5 / np.max(test_img[:, :, 0])
            change = 0.01
            error = 1e9
            keep_going = True
            best_transform = np.array([c, best_ks[0] / c, best_ks[1] / c])
            corrected = np.einsum("ijk, k->ijk", test_img, best_transform)
            while keep_going:
                best_transform = np.array([c, best_ks[0] / c, best_ks[1] / c])
                corrected = np.einsum("ijk, k->ijk", test_img, best_transform)
                corrected_img_avg_rgb = np.mean(corrected, axis=(0, 1))
                corrected_img_max_rgb = np.max(corrected, axis=(0, 1))

                checksum_corrected = sum(corrected_img_max_rgb)
                checksum_test = sum(test_img_max_rgb)
                if np.isclose(checksum_test, checksum_corrected, rtol=1.e-2, atol=1.e-2) or change < 1e-4:
                    print(c)
                    keep_going = False

                new_error = abs(sum(corrected_img_max_rgb - test_img_max_rgb))
                if new_error > error:
                    change *= -1
                if abs(error - new_error) < 0.01:
                    change /= 10
                c += change
                error = new_error

            corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            test_img = cv2.normalize(test_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            before_after = np.concatenate((test_img, corrected))
            cv2.imwrite(os.path.join(directory_output, "{}_gamut_result.jpg".format(f[:-4])), before_after)

            if self.verbose:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                transformed_observed_verticies = np.einsum("ij, j -> ij", observed_vertices, best_ks)
                plt.fill(self.hull.points[self.hull.vertices, 0], self.hull.points[self.hull.vertices, 1],
                         'k', alpha=0.1, label="canonical hull")
                plt.fill(test_hull.points[test_hull.vertices, 0], test_hull.points[test_hull.vertices, 1],
                         'c', alpha=0.2, label="observed hull")
                plt.fill(transformed_observed_verticies[:, 0], transformed_observed_verticies[:, 1],
                         'g', alpha=0.2, label="transformed observed hull")
                plt.legend()
                plt.savefig(fname=os.path.join(directory_output, "gamut_transform_{}.jpg".format(f[:-4])))

        return self.hull


parser = argparse.ArgumentParser(description='Gamut Mapping')
parser.add_argument('-i', '--input', default="../assets/wrong_white_balance", type=str)
parser.add_argument('-o', '--output', default="../vault/gamut", type=str)


def run_gamut():
    args = parser.parse_args()

    gamut = Gamut()
    # gamut.train(directory_assets="assets/gamut_canonical", pretrained_hull=None)
    gamut.test(directory_assets=args.input, directory_output=args.output)


if __name__ == '__main__':
    run_gamut()
