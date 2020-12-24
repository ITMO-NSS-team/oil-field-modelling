import datetime
import os
from copy import copy
from pathlib import Path

import pandas as pd
from fedot.core.composer.node import PrimaryNode
from fedot.core.composer.ts_chain import TsForecastingChain
from fedot.core.models.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from toolbox.forecasting_utils import calculate_validation_metric, get_comp_chain, merge_datasets

forecast_window_shift_num = 4

depth = 100


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent


def run_oil_forecasting(train_file_path,
                        train_file_path_crm,
                        forecast_length=25, max_window_size=50,
                        is_visualise=False,
                        well_id='Unknown',
                        max_time=datetime.timedelta(minutes=10)):
    """
    :param train_file_path: path to the historical well input_data
    :param train_file_path_crm: path to the CRM forecasts
    :param forecast_length: the length of the forecast for one iteration
    :param max_window_size: the size of the forecast window
    :param is_visualise: generate images of results
    :param well_id: id of the well to simulate
    :return: quality metrics
    """
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length,
                                             max_window_size=max_window_size,
                                             return_all_steps=False,
                                             make_future_prediction=False))

    full_path_train = train_file_path
    dataset_to_train = InputData.from_csv(
        full_path_train, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    # a dataset for a final validation of the composed model
    full_path_test = train_file_path
    dataset_to_validate = InputData.from_csv(
        full_path_test, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    full_path_train_crm = train_file_path_crm
    dataset_to_train_crm = InputData.from_csv(
        full_path_train_crm, task=task_to_solve, data_type=DataTypesEnum.ts,
        delimiter=',')

    dataset_to_validate_crm = copy(dataset_to_train_crm)

    prediction_full = None
    prediction_full_crm = None
    prediction_full_crm_opt = None

    for forecasting_step in range(forecast_window_shift_num):
        start = 0
        end = depth * (forecasting_step + 1)

        dataset_to_train_local = dataset_to_train.subset(start, end)
        dataset_to_train_local_crm = dataset_to_train_crm.subset(start, end)

        start = depth * (forecasting_step + 1)
        end = depth * (forecasting_step + 2)

        dataset_to_validate_local = dataset_to_validate.subset(start + depth, end + depth)
        dataset_to_validate_local_crm = dataset_to_validate_crm.subset(start + depth, end + depth)

        chain_simple = TsForecastingChain(PrimaryNode('rfr'))
        chain_simple_crm = TsForecastingChain(PrimaryNode('rfr'))
        chain_crm_opt = get_comp_chain(f'{well_id}_{forecasting_step}', dataset_to_train_local_crm,
                                       max_time)

        chain_simple.fit_from_scratch(input_data=dataset_to_train_local, verbose=False)
        chain_simple_crm.fit_from_scratch(input_data=dataset_to_train_local_crm, verbose=False)
        chain_crm_opt.fit_from_scratch(input_data=dataset_to_train_local_crm, verbose=False)

        prediction = chain_simple.forecast(dataset_to_train_local, dataset_to_validate_local)
        prediction_crm = chain_simple_crm.forecast(dataset_to_train_local_crm, dataset_to_validate_local_crm)
        prediction_crm_opt = chain_crm_opt.forecast(dataset_to_train_local_crm, dataset_to_validate_local_crm)

        prediction_full = merge_datasets(prediction_full, prediction, forecasting_step)
        prediction_full_crm = merge_datasets(prediction_full_crm, prediction_crm, forecasting_step)
        prediction_full_crm_opt = merge_datasets(prediction_full_crm_opt, prediction_crm_opt, forecasting_step)

    real_data = dataset_to_validate.subset(start=len(dataset_to_validate.idx) - len(prediction_full_crm.idx),
                                           end=len(dataset_to_validate.idx))

    rmse_on_valid_simple = calculate_validation_metric(
        prediction_full, prediction_full_crm, prediction_full_crm_opt,
        real_data, well_id, is_visualise)

    frame = pd.DataFrame({'simple_pred': prediction.predict,
                          'prediction_crm': prediction_crm.predict,
                          'prediction_crm_opt': prediction_crm_opt.predict})

    frame.to_csv(f'predictions_{well_id}.csv')

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
        full_path_train_crm = f'input_data/oil_crm_prod_X{well}.csv'
        full_path_train_crm = os.path.join(str(project_root()), full_path_train_crm)

        file_path_train = f'input_data/oil_prod_X{well}.csv'
        full_path_train = os.path.join(str(project_root()), file_path_train)

        run_oil_forecasting(full_path_train,
                            full_path_train_crm,
                            forecast_length=25,
                            max_window_size=50,
                            is_visualise=True,
                            well_id=well)
