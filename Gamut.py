import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d


class Gamut(object):
    def __init__(self, verbose=True):
        self.hull = None
        self.verbose = verbose

    def train(self, directory_assets="assets/gamut_canonical"):
        assets = [f for f in os.listdir(directory_assets) if os.path.isfile(os.path.join(directory_assets, f))]
        collective_points = np.zeros((1, 2))  # points in perspective colour space
        for f in assets:
            try:
                train_img = cv2.imread(os.path.join(directory_assets, f)).astype(float)
                perspective_image = np.einsum("ijk, ij->ijk", train_img[:, :, :-1], np.reciprocal(train_img[:, :, -1]))
                points = np.reshape(perspective_image, (-1, 2))
                points = points[np.isfinite(points).all(axis=1)]
                collective_points = np.concatenate((collective_points, points))
            except:
                pass
        self.hull = ConvexHull(points)  # ndarray of floats, shape (npoints, ndim)
        if self.verbose:
            convex_hull_plot_2d(self.hull)
            plt.savefig(fname="perspective_gamut_{}.jpg".format(f[:-4]))
        return self.hull

    def get_best_map(self, gamut_canonical):
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
