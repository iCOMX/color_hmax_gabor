import numpy as np

def gen_gabor(receptive_field_size, rot, spat_aspect,
            lamda, sgma, gabor_sign, phases):
    
    num_rotations = rot.shape[0]
    num_phases = phases.shape[0]
    filter_matrix_shape = (num_phases, receptive_field_size, receptive_field_size, num_rotations)
    filters = np.empty(filter_matrix_shape, dtype=np.float32)
    points = np.arange(-receptive_field_size/2, receptive_field_size/2, 1)

    for p in range(num_phases):
        alpha = phases[p] * np.pi / 180
        for f in range(num_rotations):
            theta = rot[f] * np.pi / 180

            for i in range(receptive_field_size):
                for j in range(receptive_field_size):
                    x = points[j] * np.cos(theta) - points[i] * np.sin(theta)
                    y = points[j] * np.sin(theta) + points[i] * np.cos(theta)

                    if np.sqrt((x*x) + (y*y)) <= (receptive_field_size / 2):
                        e = np.exp(-(x * x + spat_aspect * spat_aspect * y * y) / (2 * sgma * sgma))
                        e = e * np.cos((2 * np.pi * x) / lamda + alpha)
                    else:
                        e = 0
                    filters[p, i, j, f] = e
        
        a = filters[p, :, :, f]
        a = a - np.mean(a)
        a = a / np.std(a)
        filters[p, :, :, f] = a
    
    if 'pos' in gabor_sign:
        filters[np.where(filters < 0)] = 0
    elif 'neg' in gabor_sign:
        filters[np.where(filters > 0)] = 0

    return filters 

def gabor_filter(receptive_field_size, orients, div,
                num_channel, phases):
    
    lamda = receptive_field_size * 2 / div
    sgma = lamda * 0.8
    spat_aspect = 0.3
    
    gfilters = gen_gabor(receptive_field_size, orients, spat_aspect, lamda, sgma, 'normal', phases)

    # Separate Gabor to positive and negative
    filt1 = gen_gabor(receptive_field_size, orients, spat_aspect, lamda, sgma, 'positive', phases)
    filt2 = gen_gabor(receptive_field_size, orients, spat_aspect, lamda, sgma, 'negative', phases)

    num_phases = phases.shape[0]
    num_rotations = orients.shape[0]
    sign_filters_shape = (num_phases, receptive_field_size, receptive_field_size, 2, num_rotations)
    filters = np.empty(sign_filters_shape, dtype=np.float32)
    for p in range(num_phases):
        filters[p, :, :, 0, :] = np.abs(filt1[p, :, :, :])
        filters[p, :, :, 1, :] = np.abs(filt2[p, :, :, :])

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
    opp_filters_shape = (num_phases, receptive_field_size, receptive_field_size,
                    3, num_channel, num_rotations)
    cfilters = np.empty(opp_filters_shape, dtype=np.float32)

    for p in range(num_phases):
        for i in range(3):
            for j in range(num_channel):
                if weights[i, j] >= 0:
                    sign_idx = 0
                else:
                    sign_idx = 1
                
                cfilters[p, :, :, i, j, :] = weights[i, j] * filters[p, :, :, sign_idx, :]
    
    all_matrix_shape = (num_phases, receptive_field_size, receptive_field_size, 3,
                    num_channel * num_rotations)
    filter_all = np.empty(all_matrix_shape, dtype=np.float32)

    for p in range(num_phases):
        filter_all[p] = np.reshape(cfilters[p], all_matrix_shape[1:])
        for i in range(3):
            for j in range(num_channel * num_rotations):
                filt_norm = np.linalg.norm(filter_all[p, :, :, i, j], 2)
                if filt_norm > 0:
                    filter_all[p, :, :, i, j] = filter_all[p, :, :, i, j] / filt_norm

        cfilters[p] = np.reshape(filter_all[p], opp_filters_shape[1:])

    return gfilters, cfilters