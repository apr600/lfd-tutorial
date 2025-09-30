import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import scipy
from sklearn import mixture
### Utility Functions

## Line Utils
def calc_line_params(x,y,vx,vy):
    m = vy/vx
    b = y-m*x
    return m, b

## Circle Utils
def calc_circle_pos(r, xc, yc, theta):
    x = r*np.cos(theta) + xc
    y = r*np.sin(theta) + yc
    return x,y

def calc_theta(r, xc, yc, x,y):
    theta = np.atan2((y-yc),(x-xc))
    return theta

def calc_dist_center(r, xc, yc, x,y):
    return np.linalg.norm(np.array([x,y]) - np.array([xc, yc]))

def calc_nt_vecs(r, xc, yc, x,y):
    ## TODO: Compute sin/cos via geometry
    theta = calc_theta(r, xc, yc, x,y)
    tx = -np.sin(theta)
    ty = np.cos(theta)

    nx = np.cos(theta)
    ny = np.sin(theta)

    return np.array([[nx, ny],[tx, ty]])


### GMM Utils

## PLOTTING GMM CODE: from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html#sphx-glr-auto-examples-mixture-plot-gmm-sin-py

def plot_results(X, Y, means, covariances, index, title):
    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
    splot = plt.subplot()#5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.axis('equal')
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def plot_samples(X, Y, n_components, index, title):
    color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
    plt.subplot(5, 1, 4 + index)
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)

    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

def calc_gamma_mixing(gmm, x):
    gamma = np.ones(gmm.n_components)
    prob_components = gmm.predict_proba(np.array([x])).flatten()
    prods = gmm.weights_*prob_components
    total_sum = np.sum(prods)
    for k in range(gmm.n_components):
        gamma[k] *= prods[k]
        print(total_sum, prods[k])
        divisor = total_sum - prods[k]  # Subtract the product of the component
        gamma[k] /= divisor
    return gamma
