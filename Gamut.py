import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pickle
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def get_perspective_colors(train_img):
    perspective_image = np.einsum("ijk, ij->ijk", train_img[:, :, :-1], np.reciprocal(train_img[:, :, -1]))
    points = np.reshape(perspective_image, (-1, 2))
    return points[np.isfinite(points).all(axis=1)]


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

    def train(self, directory_assets='assets/gamut_canonical', pretrained_hull=None):
        assets = [f for f in os.listdir(directory_assets) if os.path.isfile(os.path.join(directory_assets, f))]
        if pretrained_hull is None:
            collective_points = np.zeros((1, 2))  # points in perspective colour space
            for f in assets:
                try:
                    train_img = cv2.imread(os.path.join(directory_assets, f)).astype(float)
                    points = get_perspective_colors(train_img)
                    collective_points = np.concatenate((collective_points, points))
                except:
                    pass
            self.hull = ConvexHull(collective_points)  # ndarray of floats, shape (npoints, ndim)
            if self.verbose:
                convex_hull_plot_2d(self.hull)
                plt.savefig(fname="perspective_gamut_sfu.jpg")
            marinate(self.hull, "hull.pkl")
        else:
            self.hull = relish(pretrained_hull)
        return self.hull

    def test(self, directory_assets="assets/gamut_test"):
        assets = [f for f in os.listdir(directory_assets) if os.path.isfile(os.path.join(directory_assets, f))]
        canonical_verticies = self.hull.points[self.hull.vertices]
        canonical_polygon = Polygon(canonical_verticies)
        for f in assets:
            test_img = cv2.imread(os.path.join(directory_assets, f)).astype(float)
            test_img_points = get_perspective_colors(test_img)
            test_hull = ConvexHull(test_img_points)

            observed_vertices = test_hull.points[test_hull.vertices]
            feasible_ks = []
            for i, vo in enumerate(observed_vertices):
                A = np.array([[vo[0], 0], [0, vo[1]]])
                if np.linalg.det(A) == 0:
                    continue
                while len(feasible_ks) < 10*(1+i):
                    feasible = True
                    b = sample_within_polygon(canonical_polygon)
                    k_pair = np.linalg.solve(A, b)

                    j = 0
                    while feasible and j < len(observed_vertices):
                        transform = np.array([[k_pair[0], 0], [0, k_pair[1]]])
                        transformed_point = np.matmul(transform, observed_vertices[j])
                        transformed_point = Point(transformed_point)
                        j += 1
                        if not canonical_polygon.buffer(1e-10).contains(transformed_point):
                            feasible = False
                    if feasible:
                        feasible_ks.append(k_pair)

            best_ks = max(feasible_ks, key=sum)
            best_transform = np.array([best_ks[0], best_ks[1], 1])
            corrected = np.einsum("ijk, k->ijk", test_img, best_transform)
            cv2.imwrite("vault/{}_gamut_result.jpg".format(f[:-4]), corrected)

            if self.verbose:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                transformed_observed_verticies = np.einsum("ij, j -> ij", observed_vertices, best_ks)
                plt.fill(self.hull.points[self.hull.vertices, 0], self.hull.points[self.hull.vertices,1],
                         'k', alpha = 0.2, label="canonical hull")
                plt.fill(test_hull.points[test_hull.vertices, 0], test_hull.points[test_hull.vertices,1],
                         'g', alpha = 0.2, label="observed hull")
                plt.fill(transformed_observed_verticies[:, 0], transformed_observed_verticies[:, 1],
                         'g', alpha = 0.4, label="transformed observed hull")
                plt.legend()
                plt.savefig(fname="vault/gamut_transform_{}.jpg".format(f[:-4]))

        return self.hull


# stochastic grad descent, optionally minibatch
def SGD(self, X, Y, learningrate=.01, eps=1e-2, max_iter=1000, bsize=500, btype='mini', proptype='sum', momentum=0.99,
        gamma=0.9, epsilon=1e-8, debug=True):
    # print(self.W[0])
    N, D = X.shape
    C = self.C
    if btype == '': bsize = N
    itr = 0
    dW = np.inf * np.ones_like(self.W)
    dV = np.inf * np.ones_like(self.V)
    while np.linalg.norm(dW) > eps and itr < max_iter:
        itr += 1
        # learningrate = itr**-0.51
        cost = self.cost(X, Y, self.W, self.V)
        if debug and (not itr % int(max_iter / 10)):
            print('\n[At itr {}]\nNorm dW: {}\nCost: {}\n'.format(itr, np.linalg.norm(dW), cost))
        minibatch = np.random.choice(N, bsize)
        dW, dV = self.gradients(X[minibatch], Y[minibatch], self.W, self.V)
        self.W = self.W - learningrate * dW
        self.V = self.V - learningrate * dV
        if proptype == 'rms':
            self.SW = (1 - gamma) * np.power(dW, 2) + gamma * self.SW
            self.SV = (1 - gamma) * np.power(dV, 2) + gamma * self.SV
            self.W = self.W - learningrate * dW / np.sqrt(self.SW + epsilon)
            self.V = self.V - learningrate * dV / np.sqrt(self.SV + epsilon)
        if proptype == 'sum':
            self.stepW = (1 - momentum) * dW + momentum * self.stepW
            self.stepV = (1 - momentum) * dV + momentum * self.stepV
            self.W = self.W - learningrate * self.stepW
            self.V = self.V - learningrate * self.stepV

    return cost, self.W, self.V, dW, dV
