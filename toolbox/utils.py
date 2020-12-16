import numpy as np
from pyevtk.hl import gridToVTK


def save_to_vtk(path_to_np: str,
                path_to_vtk: str = r'./image_pred'):
    """
    save the 3d input_data to a .vtk file.

    Parameters
    ------------
    path_to_np : str
        path from where download np model
    path_to_vtk : str
        where to save the vtk model, do not include vtk extension, it does automatically
    """
    data = np.load(path_to_np)

    x = np.arange(data.shape[0] + 1)
    y = np.arange(data.shape[1] + 1)
    z = np.arange(data.shape[2] + 1)
    return gridToVTK(path_to_vtk, x, y, z, cellData={'input_data': data.copy()})
