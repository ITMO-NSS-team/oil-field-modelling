# The toolbox for the oil fields modeling

This repository contains the toolbox for the oil field modeling and analysis (oil forecasting, seismic slices classification, etc). The Volve field is used as a case study.
The source data for the dataset can be found here -  “Volve oil forecasting dataset”, Mendeley Data, V1, doi: 10.17632/g2vxy237h5.1 
(URL-https://data.mendeley.com/datasets/g2vxy237h5/1)

## Oil production forecasting

The 'production_forecasting' folder contains the oil_production_forecasting_problem.py script, which allows forecasting oil production with a different model (CRMIP, ML-based, hybrid, composite, etc).

To execute and optimize the forecasting models, the [FEDOT Framework](https://github.com/nccr-itmo/FEDOT) is used. To obtain the prediction for the reservoir model, the [pyCRM](https://github.com/frank1010111/pyCRM) can be used.

The basic example of the production forecasting run is the following:

```python
from production_forecasting.oil_production_forecasting import run_oil_forecasting
run_oil_forecasting('historical_data_folder', 'crm_folder',
                    is_visualise=True, well_id='well_to_forecast')
```

The results are available in the working folder (both CSV files, images, and JSON with a description of model structure). 


## Seismic analysis
The 'seismic_analysis' folder contains Jupiter notebooks reservoir_detection_with_cnn.ipynb and oil_reservoir_semantic_segmentation.ipynb. The notebook shows how to convert a .segy file to NumPy array format, perform an inversion operation on the resulting data,  and prepare training and validation data sets of seismic time slices images. 
The CNN baseline model for binary image classification is also implemented.

To find the optimal CNN architecture, the [FEDOT-NAS tool](https://github.com/ITMO-NSS-team/nas-fedot) is used.

The basic example of the reservoir detection run is the following:

```python
from seismic_analysis.reservoir_detection_problem import run_reservoir_detection_problem

path_to_segy = r'./Inputs/PRESTACK/ST0202ZDC12-PZ-PSDM-KIRCH-FULL-D.MIG_FIN.POST_STACK.3D.JS-017534.segy'
train_dir = r'./Inputs/LABELED_IMAGES/Train'
validation_dir = r'./Inputs/LABELED_IMAGES/Validation'
test_dir = r'./Inputs/LABELED_IMAGES/Test'
model_path = 'seismic_2.h5'
run_reservoir_detection_problem(path_to_segy=path_to_segy,
                                train_path=train_dir,
                                validation_path=validation_dir,
                                test_path=test_dir,
                                model_path=model_path)
```
Procedure for starting the script is as follows:
1. A file in the format is uploaded to the 'PRESTACK' folder.segy.
2. The script saves the seismic sections in the format of the images in the folder './Output/RAW_IMAGES'.
3. Selected images of seismic sections are loaded into the 'LABELED_IMAGES' folder in each of the corresponding train, test, validation folders.
4. If a trained model already exists, it is loaded into the folder with the executable script and the model name is entered in 'model_path' variable.
5. The output of the script is a data frame with 3 columns. First column - name of the image file of the seismic slice, second column indicates the probability that a given slice to 1 of 2 classes, third column shows the class of this seismic slice.

The basic example of the semantic_segmentation reservoir run is the following:

```python
from seismic_analysis.semantic_segmentation_problem import run_semantic_segmantation_problem
from toolbox.preprocessing import get_image

image_params = (640, 400, 1)
np_data_path = r'./Outputs/SEISMIC_CUBE/slice.npy'
vtk_data_path = r'./Outputs/SEISMIC_CUBE/3D_VTK'
X, y = get_image(image_params)
run_semantic_segmantation_problem(X, y, np_data_path, vtk_data_path)
```
Procedure for starting the script is as follows:
1. The variable 'image_params' indicates the size of the image of the seismic slice that the model will work with.
2. Next, make sure that the following folders './Inputs/LABELED_IMAGES/Train/images/' and './Inputs/LABELED_IMAGES/Mask_train/' contain images of seismic slices and their "masks".
3. Next, the script splits the dataset into training and test sets and selects and saves the best model using the UNET architecture.
4. The variable 'np_data_path' indicates the path where the three-dimensional Numpy array containing the predicted shape of the oil reservoir will be stored. Variable 'vtk_data_path' specifies the path where the resulting file will be saved in the format .vtr.
5. The output of the script is a file in .vtr format that can be used for 3-d visualization of the resulting oil reservoir


