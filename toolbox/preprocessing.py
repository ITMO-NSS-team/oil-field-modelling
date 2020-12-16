import os

import matplotlib.pyplot as plt
import numpy as np
import pylops
import segyio
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize


def create_seismic_cube(segyfile_path: str):

    f = segyio.open(segyfile_path, ignore_geometry=True)
    samples = f.samples
    iline = f.attributes(segyio.TraceField.INLINE_3D)[:]
    xline = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

    # all traces
    traces = segyio.collect(f.trace)[:]
    traces_count, traces_len = traces.shape

    # seismic grid
    ilines_count = np.unique(iline)
    xlines_count = np.unique(xline)

    # edges of grid
    min_iline, max_iline = min(ilines_count), max(ilines_count)
    min_xline, max_xline = min(xlines_count), max(xlines_count)

    # create a grid step
    grid_step = samples[1] - samples[0]
    dilines = min(np.unique(np.diff(ilines_count)))
    dxlines = min(np.unique(np.diff(xlines_count)))

    # create a grid
    ilines = np.arange(min_iline, max_iline + dilines, dilines)
    xlines = np.arange(min_xline, max_xline + dxlines, dxlines)
    num_ilines, num_xlines = ilines.size, xlines.size

    ilines_grid, xlines_grid = np.meshgrid(np.arange(num_ilines),
                                           np.arange(num_xlines),
                                           indexing='ij')

    # create traces indices
    traces_indices = np.full((num_ilines, num_xlines), np.nan)
    idx_il = (iline - min_iline) // dilines
    idx_xl = (xline - min_xline) // dxlines
    traces_indices[idx_il, idx_xl] = np.arange(traces_count)
    exist_traces = np.logical_not(np.isnan(traces_indices))
    print('# traces doesnt exist: {}'.format(np.sum(~exist_traces)))

    # create a seismic cube
    cube = np.zeros((num_ilines, num_xlines, traces_len))
    cube[ilines_grid.ravel()[exist_traces.ravel()],
         xlines_grid.ravel()[exist_traces.ravel()]] = traces

    borders = (xline[0], xline[-1], iline[-1], iline[0])

    return cube, borders, grid_step


def create_wavelet(cube: np.ndarray,
                   grid_step: int):
    # lenght of wavelet in samples
    wavelet_len = 41
    #  lenght of fft
    fft_len = 2 ** 11

    # time axis for wavelet
    wavelet_time_axis = np.arange(wavelet_len) * (grid_step / 1000)
    wavelet_time_axis = np.concatenate((np.flipud(-wavelet_time_axis[1:]), wavelet_time_axis), axis=0)

    # estimate wavelet spectrum
    wavelet_fft = np.mean(np.abs(np.fft.fft(cube[..., 500:], fft_len, axis=-1)), axis=(0, 1))
    est_fft = np.fft.fftfreq(fft_len, d=grid_step / 1000)

    # create wavelet in time
    wavelet_estimated = np.real(np.fft.ifft(wavelet_fft)[:wavelet_len])
    wavelet_estimated = np.concatenate((np.flipud(wavelet_estimated[1:]), wavelet_estimated), axis=0)
    wavelet_estimated = wavelet_estimated / wavelet_estimated.max()
    center_point = np.argmax(np.abs(wavelet_estimated))

    return wavelet_estimated


def create_inversed_cube(cube: np.ndarray,
                         wavelet_estimated: np.ndarray):

    # swap time axis to first dimension
    time_start, time_end = 500, 950
    cube_small = cube[..., time_start:time_end]
    cube_small = np.swapaxes(cube_small, -1, 0)

    inversed, residual = pylops.avo.poststack.PoststackInversion(cube_small,
                                                                 wavelet_estimated,
                                                                 m0=np.zeros_like(cube_small),
                                                                 explicit=True,
                                                                 epsI=1e-3,
                                                                 simultaneous=False)
    inversed_reg, residual_reg = \
        pylops.avo.poststack.PoststackInversion(cube_small,
                                                wavelet_estimated,
                                                m0=inversed,
                                                epsI=1e-4,
                                                epsR=5e1,
                                                **dict(iter_lim=20, show=2))

    # swap time axis back to last dimension
    cube_small = np.swapaxes(cube_small, 0, -1)
    inversed = np.swapaxes(inversed, 0, -1)
    inversed_reg = np.swapaxes(inversed_reg, 0, -1)
    residual = np.swapaxes(residual, 0, -1)
    residual_reg = np.swapaxes(residual_reg, 0, -1)

    inversion_dict = {'original cube': cube_small,
                      'inversed cube': inversed,
                      'inversed reg cube': inversed_reg,
                      'residual cube': residual,
                      'residual reg cube': residual_reg}

    return inversion_dict


def save_seismic_slices(inversion_dict: dict,
                        borders: tuple):

    cube_small = inversion_dict['original cube']

    for i in range(0, 449, 1):
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.imshow(cube_small[..., i], cmap='seismic', vmin=-4, vmax=4,
                   extent=borders)
        plt.savefig(r'./image_jpg/' + str(i) + '.png', dpi=300)

    return


def get_image(image_params: tuple):
    ids = next(os.walk(r"./inputs/images"))[2]  # list of names all images in the given path
    print("No. of images = ", len(ids))

    X = np.zeros((len(ids), image_params[0], image_params[1], 1), dtype=np.float32)
    y = np.zeros((len(ids), image_params[0], image_params[1], 1), dtype=np.float32)

    for n, id_ in enumerate(ids):
        # Load images
        img = load_img(r"./inputs/images/" + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (image_params[0], image_params[1], 1), mode='constant', preserve_range=True)
        # Load masks
        mask = img_to_array(load_img(r"./mask/" + id_, grayscale=True))
        mask = resize(mask, (image_params[0], image_params[1], 1), mode='constant', preserve_range=True)
        # Save images
        X[n] = x_img / 255.0
        y[n] = mask / 255.0

    return


def augmentation(X: np.ndarray,
                 y: np.ndarray):
    X = np.append(X, [np.flipud(x) for x in X], axis=0)
    y = np.append(y, [np.flipud(x) for x in y], axis=0)

    # 90 deg rotation
    X = np.append(X, [np.flip(x, axis=(0, 1)) for x in X], axis=0)
    y = np.append(y, [np.flip(x, axis=(0, 1)) for x in y], axis=0)
    return X, y
