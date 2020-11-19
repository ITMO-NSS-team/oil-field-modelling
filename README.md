# The toolbox for the oil fields modelling

This repository contains the toolbox for the oil fields modelling and analysis (oil forecasting, seismic slices classification, etc). The Volve field is used as a case study.
The source data for the dataset can be found here -  “volvo oil forecasting dataset”, Mendeley Data, V1, doi: 10.17632/g2vxy237h5.1 
(URL-https://data.mendeley.com/datasets/g2vxy237h5/1)

## Oil production forecasting

The 'production_forecasting' folder contains the oil_production_forecasting_problem.py script, that allows forecasting oil production with a different model (CRMIP, ML-based, hybrid, composite, etc).

To execute and optimise the forecasting models, the [FEDOT Framework](https://github.com/nccr-itmo/FEDOT) is used.

## Seismic analysis
The 'seismic_analysis' folder contains Jupiter notebook seismic_inversion.ipynb. The notebook shows how to convert a .segy file to NumPy array format, perform an inversion operation on the resulting data,  and prepare training and validation data sets of seismic time slices images. 
The CNN baseline model for binary image classification is also implemented.

To find the optimal CNN architecture, the [FEDOT-NAS tool](https://github.com/ITMO-NSS-team/nas-fedot) is used.

