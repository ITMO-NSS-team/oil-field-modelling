# The toolbox for the oil fields modelling

This repository contains the toolbox for the oil fields modelling and analysis (oil forecasting, seismic slices classification, etc). The Volve field is used a case study.

## Seismic analysis
The 'seismic_analysis' folder contains jupiter notebook seismic_inversion.ipynb. Notebook shows how to convert a segy file to numpy.array format, perform an inversion operation on the resulting data,  and prepare training and validation data sets of seimic time slices images. 
The CNN baseline model for binary image classification is also implemented.

## Oil production forecasting

The 'production_forecasting' folder contains the oil_production_forecasting_problem.py script, that allow to forecast oil production with different model (CRMIP, ML-based, hybrid, composite, etc).
