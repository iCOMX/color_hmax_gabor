import numpy as np
import matplotlib.pyplot as plt

def gen_gaussians(receptive_field_size, mux, muy, sig, gabor_sign, scales):

    # Two opponents (first dimension), with different RF scales (last dimension)
    filter_matrix_shape = (num_opponents, 2, receptive_field_size, receptive_field_size, 3, len(scales))
    filters = np.zeros(filter_matrix_shape, dtype=np.float32)

    # No rotation of points by theta and no phases
    opps = [[[1, 0, 0], [0, 1, 0]],         # R - G
            [[0, 0, 1], [1, 1, 0]],         # B - Y
            [[1, 0, 0], [0, 1, 1]],         # R - C
            [[0, 0, 0], [1, 1, 1]]]         # Bl - Wh
    for i in range(num_opponents):
        for j in range(2):
            opp = opps[i][j]
            for idx, f in enumerate(scales):
                sig = f * sig
                for xi in range(receptive_field_size):
                    for yi in range(receptive_field_size):
                        x = xi - receptive_field_size / 2
                        y = yi - receptive_field_size / 2

                        e = (1. / (2 * np.pi * np.square(sig))) * np.exp(-(np.square(x - mux) + np.square(y - muy)) / (2. * np.square(sig)))
                        # No modulating the Gaussian function by the sinusoidal plane wave (used for Gabor filters)
                        filters[i, j, xi, yi, :, idx] = e
                        filters[i, j, xi, yi, 0, idx] *= opp[0]
                        filters[i, j, xi, yi, 1, idx] *= opp[1]
                        filters[i, j, xi, yi, 2, idx] *= opp[2]

        a = filters[:, :, idx]
        a = a - np.mean(a)
        a = a / np.std(a)
        filters[:, :, idx] = a

    if 'pos' in gabor_sign:
        filters[np.where(filters < 0)] = 0
    elif 'neg' in gabor_sign:
        filters[np.where(filters > 0)] = 0

    return filters

def center_surround_filter(receptive_field_size, num_channel, scales):

    gfilters = gen_gaussians(receptive_field_size, 0, 0, 3, 'normal', scales)

    # Separate Gabor to positive and negative
    filt1 = gen_gaussians(receptive_field_size, 0, 0, 3, 'positive', scales)
    filt2 = gen_gaussians(receptive_field_size, 0, 0, 3, 'negative', scales)

    sign_filters_shape = (receptive_field_size, receptive_field_size, 2, len(scales))
    filters = np.empty(sign_filters_shape, dtype=np.float32)
    for idx in range(len(scales)):
        filters[:, :, 0, idx] = np.abs(filt1[:, :, idx])
        filters[:, :, 1, idx] = np.abs(filt2[:, :, idx])

    # Initialize the predefined weight matrices
    sqrt2 = np.sqrt(2)
    sqrt6 = np.sqrt(6)
    if num_channel == 8:
        sqrt3 = np.sqrt(3)
        weights = np.asarray([
            [1/sqrt2, -1/sqrt2, 0],
            [2/sqrt6, -1/sqrt6, -1/sqrt6],
            [1/sqrt6, 1/sqrt6, -2/sqrt6],
            [1/sqrt3, 1/sqrt3, 1/sqrt3]
        ])
    elif num_channel == 6:
        weights = np.asarray([
            [1/sqrt2, -1/sqrt2, 0],
            [2/sqrt6, -1/sqrt6, -1/sqrt6],
            [1/sqrt6, 1/sqrt6, -2/sqrt6]
        ])
    tweights = weights.transpose((1, 0))
    weights = np.hstack([tweights, -tweights])

    # Spatio-chromatic opponent filters
    opp_filters_shape = (receptive_field_size, receptive_field_size, 3, num_channel, len(scales))
    cfilters = np.empty(opp_filters_shape, dtype=np.float32)

    for idx in range(len(scales)):
        for i in range(3):
            for j in range(num_channel):
                if weights[i, j] >= 0:
                    sign_idx = 0
                else:
                    sign_idx = 1

                cfilters[:, :, i, j, idx] = weights[i, j] * filters[:, :, sign_idx, idx]

    fig = plt.figure()
    #for i in range(len(scales)):
    #    ax = fig.add_subplot(1, 2, i+1)
    #    ax.imshow(filt2[:, :, i])
    for i in range(num_channel):
        ax = fig.add_subplot(3,3,i+1)
        ax.imshow(cfilters[:,:,:,i,1])
    plt.show()

    return gfilters, cfilters
