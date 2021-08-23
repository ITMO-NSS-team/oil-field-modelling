import os
import warnings

import numpy as np
import pandas as pd
from examples.time_series.ts_forecasting_tuning import prepare_input_data
from fedot.api.main import Fedot
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams
from matplotlib import pyplot as plt
from scipy.stats import t as student
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

        if any(np.isnan(time_series)):
            continue

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

    return dates, target_train, input_data_fit, input_data_predict, test_data, train_data, time_series


def run_oil_forecasting(path_to_file, path_to_file_crm, len_forecast, len_forecast_full,
                        with_visualisation, well_id) -> None:
    df = pd.read_csv(path_to_file)
    df_crm = pd.read_csv(path_to_file_crm)

    len_forecast_for_split = len_forecast_full
    dates, target_train, input_data_fit, input_data_predict, test_data, \
    train_data, time_series = prepare_dataset(df, len_forecast, len_forecast_for_split, well_id)

    dates, target_train_crm, input_data_fit_crm, input_data_predict_crm, test_data_crm, \
    train_data, time_series = prepare_dataset(df_crm, len_forecast, len_forecast_for_split, well_id)

    task_parameters = TsForecastingParams(forecast_length=len_forecast)

    if not os.path.exists(f'pipeline_{well_id}/pipeline_{well_id}.json'):
        model = Fedot(problem='ts_forecasting', task_params=task_parameters, timeout=1,
                      verbose_level=4)

        # run AutoML model design in the same way
        pipeline = model.fit(features=input_data_fit, target=target_train)
        pipeline.save(f'pipeline_{well_id}', datetime_in_path=False)
    else:
        pipeline = Pipeline()
        pipeline.load(f'pipeline_{well_id}/pipeline_{well_id}.json')

    if not os.path.exists(f'pipeline_crm_{well_id}/pipeline_crm_{well_id}.json'):
        model = Fedot(problem='ts_forecasting', task_params=task_parameters, verbose_level=4)

        # run AutoML model design in the same way
        pipeline_crm = model.fit(features=input_data_fit_crm, target=target_train_crm)
        pipeline_crm.save(f'pipeline_crm_{well_id}', datetime_in_path=False)
    else:
        pipeline_crm = Pipeline()
        pipeline_crm.load(f'pipeline_crm_{well_id}/pipeline_crm_{well_id}.json')

    sources = dict((f'data_source_ts/{data_part_key}', data_part)
                   for (data_part_key, data_part) in input_data_predict.items())
    input_data_predict_mm = MultiModalData(sources)

    sources_crm = dict((f'data_source_ts/{data_part_key}', data_part)
                       for (data_part_key, data_part) in input_data_predict_crm.items())
    input_data_predict_mm_crm = MultiModalData(sources_crm)

    forecast = out_of_sample_ts_forecast(pipeline, input_data_predict_mm, horizon=len_forecast_full)
    forecast_crm = out_of_sample_ts_forecast(pipeline_crm, input_data_predict_mm_crm, horizon=len_forecast_full)

    predicted = np.ravel(np.array(forecast))
    predicted_crm = np.ravel(np.array(forecast_crm))
    predicted_only_crm = np.asarray(df_crm[f'crm_{well_id}'][-len_forecast_full:])

    test_data = np.ravel(test_data)

    print('CRM')
    predicted_only_crm[np.isnan(predicted_only_crm)] = 0
    mse_before = mean_squared_error(test_data, predicted_only_crm, squared=False)
    mae_before = mean_absolute_error(test_data, predicted_only_crm)
    print(f'RMSE - {mse_before:.4f}')
    print(f'MAE - {mae_before:.4f}\n')

    print('ML')
    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    print(f'RMSE - {mse_before:.4f}')
    print(f'MAE - {mae_before:.4f}\n')

    print('AutoML+CRM')
    mse_before = mean_squared_error(test_data, predicted_crm, squared=False)
    mae_before = mean_absolute_error(test_data, predicted_crm)
    print(f'RMSE - {mse_before:.4f}')
    print(f'MAE - {mae_before:.4f}\n')

    if with_visualisation:
        # x = range(0, len(time_series))
        x_for = range(len(train_data), len(time_series))
        plt.plot(x_for, time_series[-len_forecast_full:], label='Actual time series', linewidth=0.5)
        plt.plot(x_for, predicted, label='ML', linewidth=0.5)
        plt.plot(x_for, predicted_crm, label='ML+CRM', linewidth=0.5)
        plt.plot(x_for, predicted_only_crm, label='CRM', linewidth=0.5)

        # ci = t_conf_interval(np.std(predicted), 0.975, len(predicted)) * 1.96
        # plt.fill_between(x_for, (predicted - ci), (predicted + ci),
        #                 color='orange', alpha=.5)

        ci_crm = t_conf_interval(np.std(predicted_crm), 0.975, len(predicted_crm)) * 1.96
        plt.fill_between(x_for, (predicted_crm - ci_crm), (predicted_crm + ci_crm),
                         color='orange', alpha=.5)

        ci_crmonly = t_conf_interval(np.std(predicted_only_crm), 0.975, len(predicted_only_crm)) * 1.96
        plt.fill_between(x_for, (predicted_only_crm - ci_crmonly), (predicted_only_crm + ci_crmonly),
                         color='green', alpha=.5)

        plt.xlabel('Days from 2013.06.01')
        plt.ylabel('Oil volume, m3')
        plt.legend()
        # plt.grid()
        plt.title(well_id)
        plt.tight_layout()
        plt.show()


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
