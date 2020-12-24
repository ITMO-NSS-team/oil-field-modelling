import os

from seismic_analysis.semantic_segmentation_problem import run_semantic_segmantation_problem
from toolbox.preprocessing import get_image


def _clean():
    try:
        os.remove('./slice.npy')
        os.remove('./3D_VTK.vtr')
    except OSError:
        pass


def test_reservoir_segmentation():
    _clean()

    image_params = (640, 400, 1)
    np_data_path = r'./slice.npy'
    vtk_data_path = r'./3D_VTK'
    x, y = get_image(image_params, f'{__file__}/../test_data/seismic/')
    pre_fitted_model_path = 'model_oil.h5'

    run_semantic_segmantation_problem(x, y,
                                      image_params,
                                      np_data_path, vtk_data_path,
                                      pre_fitted_model_path)

    assert os.path.exists(np_data_path)
    assert os.path.exists(f'{vtk_data_path}.vtr')

    _clean()
