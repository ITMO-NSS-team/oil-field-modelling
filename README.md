# The toolbox for the oil fields modeling

This repository contains the toolbox for the oil field modeling and analysis (oil forecasting, seismic slices classification, etc). The Volve field is used as a case study.
The source data for the dataset can be found here -  “Volve oil forecasting dataset”, Mendeley Data, V1, doi: 10.17632/g2vxy237h5.1 
(URL-https://data.mendeley.com/datasets/g2vxy237h5/1)

## Oil production forecasting

The 'production_forecasting' folder contains the oil_production_forecasting_problem.py script, which allows forecasting oil production with a different model (CRMIP, ML-based, hybrid, composite, etc).

To execute and optimize the forecasting models, the [FEDOT Framework](https://github.com/nccr-itmo/FEDOT) is used. To obtain the prediction for the reservoir model, the [pyCRM](https://github.com/frank1010111/pyCRM) can be used.

The basic example of the production forecasting run is the following:

```python
from production_forecasting.oil_production_forecasting_problem import run_oil_forecasting
run_oil_forecasting('historical_data_folder', 'crm_folder', 
                    is_visualise=True, well_id='well_to_forecast')
```

The results are available in the working folder (both CSV files, images, and JSON with a description of model structure). 


## Seismic analysis
The 'seismic_analysis' folder contains Jupiter notebook seismic_inversion.ipynb. The notebook shows how to convert a .segy file to NumPy array format, perform an inversion operation on the resulting data,  and prepare training and validation data sets of seismic time slices images. 
The CNN baseline model for binary image classification is also implemented.

To find the optimal CNN architecture, the [FEDOT-NAS tool](https://github.com/ITMO-NSS-team/nas-fedot) is used.

