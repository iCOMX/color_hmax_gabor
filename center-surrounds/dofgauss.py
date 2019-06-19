import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def gaussian2D(x, y, sigma):
    A = 1.0 / (2 * np.pi * np.square(sigma))
    B = np.exp(-A * (np.square(x) + np.square(y)))
    return 255.0 * (A * B)

def mexicanHat(x, y, sigma1, sigma2):
    return gaussian2D(x, y, sigma1) - gaussian2D(x, y, sigma2)

def receptiveFieldMatrix(func, channel_idx=None):
    h = 30
    if channel_idx is None:
        g = np.zeros((h,h))
        mask = np.zeros((h,h))
    else:
        g = np.zeros((h,h,3))
        mask = np.zeros((h,h))
    for xi in range(0,h):
        for yi in range(0,h):
            x = xi-h/2
            y = yi-h/2
            if channel_idx is None:
                g[xi, yi] = func(x,y)
            else:
                g[xi, yi, channel_idx[0]] = func(x,y)
                g[xi, yi, channel_idx[1]] = 0
                g[xi, yi, channel_idx[2]] = 0

                mask[xi,yi] = func(x,y)
    return g, mask

def plotFilterDiff(func1, func2, cord1=[0, 1, 2], cord2=[1, 0, 2]):
    rfmat1, m1 = receptiveFieldMatrix(func1, cord1)
    rfmat2, m2 = receptiveFieldMatrix(func2, cord2)
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(3,2,1)
    ax1.imshow(rfmat1)
    ax1.set_title("Gaussian 1 - Center")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2 = fig.add_subplot(3,2,2)
    ax2.imshow(rfmat2)
    ax2.set_title("Gaussian 2 - Surround")
    ax2.set_xticks([]); ax2.set_yticks([])
    annulus = m2 - m1
    center = m1
    rfmat = np.zeros_like(rfmat1)
    rfmat = rfmat + rfmat1 * np.stack([center] * 3, axis=-1) + rfmat2 * np.stack([annulus] * 3, axis=-1)
    ax3 = fig.add_subplot(3,2,3)
    ax3.imshow(rfmat)
    ax3.set_title("DoG - Center Surround RF")
    ax3.set_xticks([]); ax3.set_yticks([])
    ax4 = fig.add_subplot(3,2,4)
    ax4.imshow(center, cmap=cm.Greys_r)
    ax4.set_title("Mask of Gaussian 1 (Grey CMAP)")
    ax4.set_xticks([]); ax4.set_yticks([])
    ax5 = fig.add_subplot(3,2,5)
    ax5.imshow(annulus, cmap=cm.Greys_r)
    ax5.set_title("Mask of Gaussian 2 - Gaussian 1 (Grey CMAP)")
    ax5.set_xticks([]); ax5.set_yticks([])
    plt.show()

def plotFilter(func):
    rfmat = receptiveFieldMatrix(func)
    #plt.imshow(rfmat, cmap=cm.Greys_r)
    plt.imshow(rfmat)
    plt.show()

if __name__ == '__main__':
    f1 = (lambda x,y: gaussian2D(x,y,1.5))
    f2 = (lambda x,y: gaussian2D(x,y,4))
    f3 = (lambda x,y: mexicanHat(x,y,3,4))
    cord2 = [1, 0, 2]
    cord1 = [0, 1, 2]
    plotFilterDiff(f1, f2, cord1, cord2)

    #plotFilter(f1)
    #plotFilter(f3)
