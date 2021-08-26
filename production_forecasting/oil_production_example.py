import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
from examples.time_series.ts_forecasting_tuning import prepare_input_data
from fedot.api.api_utils import _create_multidata_pipeline, array_to_input_data
from fedot.api.main import _extract_features_from_data_part, _get_source_type
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TsForecastingParams
from scipy.stats import t as student

from production_forecasting.oil_production_forecasting import project_root

warnings.filterwarnings('ignore')
np.random.seed(2021)


def t_conf_interval(std, percentile, n):
    quantile = student.ppf(percentile, n)
    return std * quantile / np.sqrt(n)


def prepare_dataset(df, len_forecast, len_forecast_for_split, target_well_id):
    var_names = df.columns
    input_data_fit = {}
    input_data_predict = {}
    target_train = np.asarray(df[f'prod_X{target_well_id}'][:-len_forecast_for_split])
    for var_name in var_names:
        if var_name == 'DATEPRD':
            dates = list(df[var_name])
            continue
        time_series = np.asarray(df[var_name])

        # Let's divide our data on train and test samples
        train_data = time_series[:(len(time_series) - len_forecast_for_split)]
        test_data = time_series[:len_forecast_for_split]

        # Source time series
        train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                              train_data_features=train_data,
                                                              train_data_target=target_train,
                                                              test_data_features=test_data)
        train_input.task.task_params.forecast_length = len_forecast
        predict_input.task.task_params.forecast_length = len_forecast

        task.task_params.forecast_length = len_forecast

        if 'crm' in var_name:
            var_name = f'exog_{var_name}'
        input_data_fit[var_name] = train_input
        input_data_predict[var_name] = predict_input

    return dates, target_train, input_data_fit, input_data_predict, predict_input, train_input, time_series


def run_oil_forecasting(path_to_file, path_to_file_crm, len_forecast, len_forecast_full,
                        with_visualisation, well_id) -> None:
    df = pd.read_csv(path_to_file)
    df_crm = pd.read_csv(path_to_file_crm)

    len_forecast_for_split = len_forecast_full
    dates, target_train, input_data_fit, input_data_predict, test_data, train_data, time_series = \
        prepare_dataset(df, len_forecast, len_forecast_for_split, well_id)

    task_parameters = TsForecastingParams(forecast_length=len_forecast)
    train_data.task.task_params = task_parameters
    ###############
    data_part_transformation_func = partial(array_to_input_data,
                                            target_array=train_data.target, task=train_data.task)

    # create labels for data sources
    sources = dict(
        (f'{_get_source_type(data_part_key)}/{data_part_key}', data_part_transformation_func(
            features_array=_extract_features_from_data_part(data_part)))
        for (data_part_key, data_part) in train_data.features.items())
    data = MultiModalData(sources)
    pp = Pipeline(_create_multidata_pipeline(train_data.task, train_data, has_categorical_features=False))
    pp.fit(data)


if __name__ == '__main__':
    for well in ['7078']:  # ['5351', '5599', '7078', '7289']:  # , '7405f']:
        full_path_train_crm = f'input_data/oil_crm_prod_X{well}.csv'
        full_path_train_crm = os.path.join(str(project_root()), full_path_train_crm)

        file_path_train = f'input_data/oil_prod_X{well}.csv'
        full_path_train = os.path.join(str(project_root()), file_path_train)

        run_oil_forecasting(path_to_file=full_path_train,
                            path_to_file_crm=full_path_train_crm,
                            len_forecast=100,
                            len_forecast_full=400,
                            with_visualisation=True,
                            well_id=well)
