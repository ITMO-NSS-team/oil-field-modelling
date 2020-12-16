import os

import numpy as np
import pandas as pd

from toolbox.custom_model import SeismicModel
from toolbox.preprocessing import create_inversed_cube, create_seismic_cube, create_wavelet, save_seismic_slices


def run_reservoir_detection_problem(path_to_segy: str,
                                    train_path: str,
                                    validation_path: str,
                                    test_path: str,
                                    model_path: str):
    origin_cube, borders, grid_step = create_seismic_cube(segyfile_path=path_to_segy)
    wavelet = create_wavelet(cube=origin_cube,
                             grid_step=grid_step)
    inversed_dict = create_inversed_cube(cube=origin_cube,
                                         wavelet_estimated=wavelet)
    save_seismic_slices(inversion_dict=inversed_dict,
                        borders=borders)

    model = SeismicModel(train_path=train_path,
                         validation_path=validation_path,
                         test_path=test_path,
                         model_path=model_path)

    results = model.fit()
    model.plot_results(results, loss_flag=True)

    predictions, filenames = model.predict()

    classes = np.round(predictions)
    df_with_predictions = pd.DataFrame({"file": filenames, "pr": predictions[:, 0], "class": classes[:, 0]})

    return df_with_predictions


if __name__ == '__main__':
    path_to_segy = r'./Stacks/ST0202ZDC12-PZ-PSDM-KIRCH-FULL-D.MIG_FIN.POST_STACK.3D.JS-017534.segy'
    train_dir = r'./Test/Train'
    validation_dir = r'./Test/Validation'
    test_dir = r'./Test/Test'
    model_path = 'r./seismic_2.h5'

    if not os.path.exists(path_to_segy):
        raise ValueError(
            'Download full input data (2.6GB) from https://data.mendeley.com/datasets/g2vxy237h5/1 first. '
            'It is too large to be placed in GitHub.')

    run_reservoir_detection_problem(path_to_segy=path_to_segy,
                                    train_path=train_dir,
                                    validation_path=validation_dir,
                                    test_path=test_dir,
                                    model_path=model_path)
