import numpy as np
from sklearn.model_selection import train_test_split

from toolbox.custom_model import OilModel
from toolbox.preprocessing import augmentation, get_image
from toolbox.utils import save_to_vtk


def run_semantic_segmantation_problem(x: np.ndarray,
                                      y: np.ndarray,
                                      path_to_np: str,
                                      path_to_vtk: str):
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, y_train = augmentation(x_train, y_train)
    x_valid, y_valid = augmentation(x_valid, y_valid)

    model = OilModel()
    results = model.fit(train_data=(x_train, y_train),
                        val_data=(x_valid, y_valid),
                        batch_size=32,
                        epochs=50)

    model.plot_learning_curve(results, 'loss')
    model.plot_learning_curve(results, 'accuracy')

    # Predict on whole dataset
    preds_train = model.predict(X)
    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)

    np.save(path_to_np, preds_train_t)
    save_to_vtk(path_to_vtk)


if __name__ == '__main__':
    image_params = (640, 400, 5)
    np_data_path = r'./arrays/image_pred.npy'
    vtk_data_path = r'./arrays/image_pred'
    X, y = get_image(image_params)
    run_semantic_segmantation_problem(X, y, np_data_path, vtk_data_path)
