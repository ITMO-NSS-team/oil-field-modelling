import os
import sys
from copy import copy

path = os.path.abspath('./FEDOT')
if path not in sys.path:
    sys.path.append(path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtw import dtw
from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, OutputData
from core.models.preprocessing import EmptyStrategy
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from core.utils import project_root


# static example of the composite chain (without optimisation)
def get_comp_chain():
    node_first = PrimaryNode('lstm')
    node_second = PrimaryNode('rfr')

    node_final = SecondaryNode('linear',
                               nodes_from=[node_first, node_second],
                               manual_preprocessing_func=EmptyStrategy)
    chain = Chain(node_final)
    return chain


def get_crm_prediction_with_intervals(well_name):
    print(os.getcwd())

    file_path_crm = f'./production_forecasting/data/crmip.csv'

    data_frame = pd.read_csv(file_path_crm, sep=',')
    crm = data_frame[f'mean_{well_name}'][(300 - 1):700]
    crm[np.isnan(crm)] = 0
    return crm


def calculate_validation_metric(pred: OutputData, pred_crm, pred_crm_opt, valid: InputData,
                                name: str, is_visualise=False):
    forecast_length = pred.task.task_params.forecast_length

    skip_start_id = 100

    # skip initial part of time series
    predicted = pred.predict[~np.isnan(pred.predict)]
    predicted_crm = pred_crm.predict[~np.isnan(pred.predict)]
    predicted_crm_opt = pred_crm_opt.predict[~np.isnan(pred.predict)]

    real = valid.target[skip_start_id:][~np.isnan(pred.predict)]

    crm = get_crm_prediction_with_intervals(name)

    # the quality assessment for the simulation results
    rmse_ml = mse(y_true=real, y_pred=predicted, squared=False)
    rmse_ml_crm = mse(y_true=real, y_pred=predicted_crm, squared=False)
    rmse_crm_opt = mse(y_true=real, y_pred=predicted_crm_opt, squared=False)
    rmse_crm = mse(y_true=real, y_pred=crm, squared=False)

    dist = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    dtw_ml, _, _, _ = dtw(real, predicted, dist)
    dtw_ml_crm, _, _, _ = dtw(real, predicted_crm, dist)
    dtw_crm_opt, _, _, _ = dtw(real, predicted_crm_opt, dist)
    dtw_crm, _, _, _ = dtw(real, list(crm), dist)

    # plot results
    if is_visualise:
        compare_plot(predicted, predicted_crm, predicted_crm_opt, real,
                     forecast_length=forecast_length,
                     model_name=name, err=rmse_crm)

    return rmse_crm, rmse_ml, rmse_ml_crm, rmse_crm_opt, \
           dtw_crm, dtw_ml, dtw_ml_crm, dtw_crm_opt


def get_crm_intervals(model_name):
    file_path_crm = f'./production_forecasting/data/crmip.csv'

    data_frame = pd.read_csv(file_path_crm, sep=',')
    validation_range_start, validation_range_end = 300, 700
    mean = data_frame[f'mean_{model_name}'][validation_range_start:validation_range_end]
    min_int = data_frame[f'min_{model_name}'][validation_range_start:validation_range_end]
    max_int = data_frame[f'max_{model_name}'][validation_range_start:validation_range_end]

    times = [_ for _ in range(len(mean))]

    return min_int, mean, max_int, times


def compare_plot(predicted, predicted_crm, predicted_crm_opt, real, forecast_length, model_name, err):
    min_int, mean, max_int, times = get_crm_intervals(model_name)

    plt.clf()
    _, ax = plt.subplots()
    plt.plot(times, mean, label='CRM')
    plt.fill_between(times, min_int, max_int, alpha=0.2)

    plt.plot(real, linewidth=1, label="Observed", alpha=0.8)
    plt.plot(predicted, linewidth=1, label="ML", alpha=0.8)
    plt.plot(predicted_crm, linewidth=1, label="ML+CRM", alpha=0.8)
    plt.plot(predicted_crm_opt, linewidth=1, label="Evo ML+CRM", alpha=0.8)

    ax.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Oil volume')
    plt.title(f'Oil production for {forecast_length} hours in {model_name}, RMSE={round(err)} m3')
    plt.savefig(f'{model_name}.png')
    # plt.show()


def merge_datasets(prediction_full: OutputData, prediction: OutputData, pred_step):
    prediction_full_new = copy(prediction_full)
    if not prediction_full:
        prediction_full_new = prediction
    else:
        prediction_full_new.idx = np.append(prediction_full.idx, prediction.idx[~np.isnan(prediction.predict)])
        prediction_full_new.predict = np.append(prediction_full.predict,
                                                prediction.predict[~np.isnan(prediction.predict)], axis=0)

    if pred_step > 0:
        prediction_full_new.predict = prediction_full_new.predict[:len(prediction_full_new.predict) - 1]
        prediction_full_new.idx = prediction_full_new.idx[:len(prediction_full_new.idx) - 1]

    return prediction_full_new


def run_oil_forecasting_problem(train_file_path,
                                train_file_path_crm,
                                forecast_length, max_window_size,
                                is_visualise=False,
                                well_id='Unknown'):
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length,
                                             max_window_size=max_window_size,
                                             return_all_steps=False,
                                             make_future_prediction=False))

    full_path_train = os.path.join(str(project_root()), train_file_path)
    dataset_to_train = InputData.from_csv(
        full_path_train, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    # a dataset for a final validation of the composed model
    full_path_test = os.path.join(str(project_root()), train_file_path)
    dataset_to_validate = InputData.from_csv(
        full_path_test, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    full_path_train_crm = os.path.join(str(project_root()), train_file_path_crm)
    dataset_to_train_crm = InputData.from_csv(
        full_path_train_crm, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    dataset_to_validate_crm = copy(dataset_to_train_crm)

    prediction_full = None
    prediction_full_crm = None
    prediction_full_crm_opt = None

    forecast_window_shift_num = 4

    depth = 100

    for forecasting_step in range(forecast_window_shift_num):
        start = 0 + depth * forecasting_step
        end = depth * 2 + depth * (forecasting_step + 1)

        dataset_to_train_local = dataset_to_train.subset(start, end)
        dataset_to_train_local_crm = dataset_to_train_crm.subset(start, end)

        start = 0 + depth * forecasting_step
        end = depth * 2 + depth * (forecasting_step + 1)

        dataset_to_validate_local = dataset_to_validate.subset(start + depth, end + depth)
        dataset_to_validate_local_crm = dataset_to_validate_crm.subset(start + depth, end + depth)

        chain_simple = Chain(PrimaryNode('lstm'))
        chain_simple_crm = Chain(PrimaryNode('lstm'))
        chain_crm_opt = get_comp_chain()

        chain_simple.fit_from_scratch(input_data=dataset_to_train_local, verbose=False)
        chain_simple_crm.fit_from_scratch(input_data=dataset_to_train_local_crm, verbose=False)
        chain_crm_opt.fit_from_scratch(input_data=dataset_to_train_local_crm, verbose=False)

        prediction = chain_simple.predict(dataset_to_validate_local)
        prediction_crm = chain_simple_crm.predict(dataset_to_validate_local_crm)
        prediction_crm_opt = chain_crm_opt.predict(dataset_to_validate_local_crm)

        prediction_full = merge_datasets(prediction_full, prediction, forecasting_step)
        prediction_full_crm = merge_datasets(prediction_full_crm, prediction_crm, forecasting_step)
        prediction_full_crm_opt = merge_datasets(prediction_full_crm_opt, prediction_crm_opt, forecasting_step)

    rmse_on_valid_simple = calculate_validation_metric(
        prediction_full, prediction_full_crm, prediction_full_crm_opt, dataset_to_validate,
        well_id,
        is_visualise)

    print(well_id)
    print(f'RMSE CRM: {round(rmse_on_valid_simple[0])}')
    print(f'RMSE ML: {round(rmse_on_valid_simple[1])}')
    print(f'RMSE ML with CRM: {round(rmse_on_valid_simple[2])}')
    print(f'Evo RMSE ML with CRM: {round(rmse_on_valid_simple[3])}')

    print(f'DTW CRM: {round(rmse_on_valid_simple[4])}')
    print(f'DTW ML: {round(rmse_on_valid_simple[5])}')
    print(f'DTW ML with CRM: {round(rmse_on_valid_simple[6])}')
    print(f'DTW RMSE ML with CRM: {round(rmse_on_valid_simple[7])}')

    return rmse_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from Volve dataset of oil field

    for well in ['5351', '5599', '7078', '7289', '7405f']:
        full_path_train_crm = f'../production_forecasting/data/oil_crm_prod_X{well}.csv'
        full_path_train_crm = os.path.join(str(project_root()), full_path_train_crm)

        file_path_train = f'../production_forecasting/data/oil_prod_X{well}.csv'
        full_path_train = os.path.join(str(project_root()), file_path_train)

        run_oil_forecasting_problem(full_path_train,
                                    full_path_train_crm,
                                    forecast_length=100,
                                    max_window_size=100,
                                    is_visualise=True,
                                    well_id=well)
