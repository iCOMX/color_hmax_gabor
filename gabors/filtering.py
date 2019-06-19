import numpy as np
import yaml as pyml
from scipy.signal import convolve2d as conv2d
from single_opponency import gabor_filter

from PIL import Image
import os, glob, sys
import matplotlib.pyplot as plt

def diviseNormSingleOpponency(so_cell, scale, contrast, num_channel):
    E = np.sum(np.square(so_cell), axis=2) / num_channel

    for j in range(num_channel):
        so_cell[:, :, j, :, :] = np.sqrt(
            scale * np.square(so_cell[:, :, j, :, :]) / (contrast * contrast + E)
        )

    return so_cell

def computeDoubleOpponency(s, num_channels, filters):
    ds = np.empty(s.shape, dtype=np.float32)
    num_phases, rfsize, rfsize, num_rotations = filters.shape

    for p in range(num_phases):
        for j in range(num_channels):
            for i in range(num_rotations):
                ds[:, :, j, i, p] = conv2d(s[:, :, j, i, p],
                                           filters[p, :, :, i],
                                           boundary='fill',
                                           mode='same')
    ds[np.where(ds < 0)] = 0

    tmpdc = np.empty((s.shape[0], s.shape[1], num_channels, num_rotations))
    for i in range(num_phases):
        tmpdc = tmpdc + ds[:, :, :, :, i] / num_phases

    dc = np.empty((s.shape[0], s.shape[1], num_channels/2, num_rotations))
    for j in range(num_channels / 2):
        dc[:, :, j, :] = np.sqrt(np.square(tmpdc[:, :, j, :]) + np.square(tmpdc[:, :, j+num_channels/2, :]))

    return ds, dc

def computeSingleOpponency(img, filters):
    h, w, c = img.shape
    num_phases, rfsize, rfsize, _, num_channels, num_rotations = filters.shape
    output_size = (h, w, num_channels, num_rotations, num_phases)
    s = np.empty(output_size, dtype=np.float32)

    for p in range(num_phases):
        for j in range(num_channels):
            for i in range(num_rotations):
                for k in range(c):
                    tmp = conv2d(img[:, :, k],
                                np.squeeze(filters[p, :, :, k, j, i]),
                                boundary='fill',
                                mode='same')
                    s[:, :, j, i, p] = s[:, :, j, i, p] + tmp

    s[np.where(s < 0)] = 0

    return s

if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        args = pyml.load(f)

    phases = np.asarray(args['phases'], np.float32)
    sigma_contrast = np.float32(args['sigma_contrast'])
    rotations = np.asarray(args['orients'], np.float32)
    RFsize = np.int32(args['receptive_field_size'])
    num_channel = np.int32(args['num_channel'])
    scale_factor = np.float32(args['scaling'])
    img_path = args['img_path']
    div_range = np.asarray(args['div_range'], np.float32)

    if 'so' in config_file:
        channels = args['channel_names']
    elif 'do' in config_file:
        channels_1 = args['channel_names_1']
        channels_2 = args['channel_names_2']

    div = np.arange(div_range[0], div_range[1], -0.05)

    if 'so' in config_file:
        gfilters, cfilters = gabor_filter(RFsize,
                                        rotations,
                                        div[2],
                                        num_channel,
                                        phases)
        np.save('so_filters.npy', cfilters)
    elif 'do' in config_file:
        gfilters, cfilters = gabor_filter(RFsize,
                                        rotations,
                                        div[2],
                                        num_channel,
                                        phases)
        np.save('do_filters.npy', cfilters)

    inp_image = np.array(Image.open(img_path))
    if np.max(inp_image) > 1:
        inp_image = inp_image * 1. / 255.
    inp_image = inp_image * 2. - 1
    out_image = computeSingleOpponency(inp_image, cfilters)

    out_image = diviseNormSingleOpponency(out_image, scale_factor,
                                sigma_contrast, num_channel)

    if 'do' in config_file:
        ds, dc = computeDoubleOpponency(out_image, num_channel, gfilters)
        out_image = ds
        channels = channels_1

    # Plot the filter responses on the input image
    fig = plt.figure(figsize=(12,16))
    nrows = 2
    ncols = num_channel / nrows

    for img_num in range(num_channel):
        ax = fig.add_subplot(nrows, ncols, img_num+1)
        if 'do' in config_file:
            mean_rot_image = np.max(out_image[:, :, img_num, :, 0], axis=2)
        else:
            mean_rot_image = np.mean(out_image[:, :, img_num, :, 0], axis=2)
        ax.imshow(mean_rot_image, extent=[0, 1, 0, 1])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(channels[img_num])

    if 'do' in config_file:
        channels = channels_2
        fig1 = plt.figure(figsize=(12, 16))
        nrows = 1
        ncols = num_channel / 2
        for img_num in range(num_channel/2):
            ax = fig1.add_subplot(nrows, ncols, img_num+1)
            max_rot_image = np.max(dc[:, :, img_num, :], axis=2)
            ax.imshow(max_rot_image, extent=[0, 1, 0, 1])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(channels[img_num])

        fig2 = plt.figure(figsize=(12, 16))
        ax = fig2.add_subplot(1, 1, 1)
        plot_img = np.max(np.max(dc, axis=3), axis=2)
        ax.imshow(plot_img, extent=[0, 1, 0, 1])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Double Opponency Edge Response')

    plt.show()
