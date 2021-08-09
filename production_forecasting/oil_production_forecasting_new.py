import os
import timeit
import warnings

import numpy as np
import pandas as pd
from examples.ts_forecasting_tuning import prepare_input_data
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from production_forecasting.oil_production_forecasting import project_root

warnings.filterwarnings('ignore')
np.random.seed(2021)


def make_forecast(pipeline, train: InputData, predict: InputData,
                  train_exog: InputData, predict_exog: InputData):
    """
    Function for predicting values in a time series

    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()

    second_node_name = 'exog_ts_data_source'

    if train_exog is None:
        second_node_name = 'lagged/2'
        train_exog = train
        predict_exog = predict

    train_dataset = MultiModalData({
        'lagged/1': train,
        second_node_name: train_exog,
    })

    predict_dataset = MultiModalData({
        'lagged/1': predict,
        second_node_name: predict_exog,
    })

    pipeline.fit_from_scratch(train_dataset)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train pipeline\n')

    # Predict
    predicted_values = pipeline.predict(predict_dataset)
    predicted_values = predicted_values.predict

    return predicted_values


def run_oil_forecasting(path_to_file, len_forecast,
                        with_visualisation, well_id) -> None:
    df = pd.read_csv(path_to_file)
    var_names = df.columns

    input_data_fit = {}
    input_data_predict = {}
    len_forecast_for_split = 4 * len_forecast
    target_train = np.asarray(df[f'prod_X{well_id}'][:-len_forecast_for_split])
    dates = []
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

        input_data_fit[var_name] = train_input
        input_data_predict[var_name] = predict_input

    parent_nodes = []
    for name in input_data_fit.keys():
        parent_nodes.append(SecondaryNode('ridge',
                                          nodes_from=[SecondaryNode('lagged',
                                                                    nodes_from=[
                                                                        PrimaryNode(f'data_source_ts/{name}')])]))
    predefined = Pipeline(SecondaryNode('ridge', nodes_from=parent_nodes))

    task_parameters = TsForecastingParams(forecast_length=len_forecast)

    model = Fedot(problem='ts_forecasting', task_params=task_parameters, timeout=1.0)

    # run AutoML model design in the same way
    pipeline = model.fit(features=input_data_fit, target=target_train)  # , predefined_model=predefined)

    sources = dict((f'data_source_ts/{data_part_key}', data_part)
                   for (data_part_key, data_part) in input_data_predict.items())
    input_data_predict_mm = MultiModalData(sources)

    forecast = in_sample_ts_forecast(pipeline, input_data_predict_mm, horizon=400)

    predicted = np.ravel(np.array(forecast))
    test_data = np.ravel(test_data)

    print(f'Predicted values: {predicted[:5]}')
    print(f'Actual values: {test_data[:5]}')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    print(f'RMSE - {mse_before:.4f}')
    print(f'MAE - {mae_before:.4f}\n')

    if with_visualisation:
        x = range(0, len(time_series))
        x_for = range(len(train_data), len(time_series))
        plt.plot(x, time_series, label='Actual time series')
        plt.plot(x_for, predicted, label='Forecast')
        # plt.xticks(x, dates, rotation=45)

        # n = 25
        # ax = plt.gca()
        # [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]

        plt.xlabel('Days from 2013.06.01')
        plt.ylabel('Oil volume, m3')
        plt.legend()
        # plt.grid()
        plt.title(well_id)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    for well in ['5351', '5599', '7078', '7289', '7405f']:
        full_path_train_crm = f'input_data/oil_crm_prod_X{well}.csv'
        full_path_train_crm = os.path.join(str(project_root()), full_path_train_crm)

        file_path_train = f'input_data/oil_prod_X{well}.csv'
        full_path_train = os.path.join(str(project_root()), file_path_train)

        run_oil_forecasting(path_to_file=full_path_train,
                            len_forecast=100,
                            with_visualisation=True,
                            well_id=well)
