import numpy as np
import random


def add_zeros(x, max_zeros=150):

    x = np.atleast_1d(x)
    for x_ in x:
        num_zeros_init = np.sum(x_ == 0)
        if num_zeros_init > max_zeros:
            continue
        else:
            num_zeros = random.randint(0, max_zeros - num_zeros_init)
            indices = random.sample(range(len(x_)), num_zeros)
            if len(indices ) != 0:
                x_[indices] = 0
    return x


def telluric_mask(telluric_line_file, wav):
    # Load .txt file containing information about telluric lines
    telluric_lines = np.loadtxt(telluric_line_file, skiprows=1)

    # Extract relevant information
    telluric_regions = np.column_stack((telluric_lines[:, 0], telluric_lines[:, 1]))
    # residual_intensity = telluric_lines[:,2]

    # Generate telluric mask
    telluric_mask = np.ones(len(wav))
    for region in telluric_regions:
        lower_wl, upper_wl = region
        mask = (wav > lower_wl) & (wav < upper_wl)
        telluric_mask[mask] = 0

    return telluric_mask


def mask_tellurics(telluric_line_file, X, wav):
    mask = telluric_mask(telluric_line_file, wav)

    if np.ndim(X) == 1:
        X *= mask
    elif np.ndim(X) == 2:
        for x in X:
            x *= mask

    return X


def apply_global_error_mask(x, global_mask):

    for item in x:
        item *= global_mask
    return x

