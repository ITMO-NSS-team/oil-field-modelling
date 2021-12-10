from os.path import join as join

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams

from production_forecasting.crm.crm import CRMP as crm
from toolbox.preprocessing import project_root


def prepare_data():
    filepath = join(project_root(), 'production_forecasting', 'input_data', 'MONSON')
    parse_date = True  # This dataset has dates instead of elapsed time. Hence convert to timedelta

    qi = pd.read_excel(join(filepath, 'injection.xlsx'), engine='openpyxl')
    qp = pd.read_excel(join(filepath, 'production.xlsx'), engine='openpyxl')
    percent_train = 0.5

    time_colname = 'Time [days]'
    if parse_date:
        qi[time_colname] = (qi.Date - qi.Date[0]) / pd.to_timedelta(1, unit='D')

    num_items = 40

    InjList = [x for x in qi.keys() if x.startswith('I')]
    PrdList = [x for x in qp.keys() if x.startswith('P')]
    t_arr = qi[time_colname].values[:num_items]

    N_inj = len(InjList)
    N_prd = len(PrdList)
    qi_arr = qi[InjList].values[:num_items, :]
    q_obs = qp[PrdList].values[:num_items, :]

    # Separation into training and test set

    n_train = int(percent_train * len(t_arr))

    q_obs_train = q_obs[:n_train, :]
    q_obs_test = q_obs[n_train:, :]

    forecast_length = 2

    ds_train = {}
    ds_test = {}

    for i in range(q_obs.shape[1]):
        ds_train[f'data_source_ts/prod_{i}'] = InputData(idx=np.arange(0, n_train),
                                                         features=q_obs[:n_train, i][..., np.newaxis],
                                                         target=q_obs_train,
                                                         data_type=DataTypesEnum.ts,
                                                         task=Task(TaskTypesEnum.ts_forecasting,
                                                                   task_params=TsForecastingParams(
                                                                       forecast_length=forecast_length)))

        ds_test[f'data_source_ts/prod_{i}'] = InputData(idx=np.arange(n_train, len(t_arr)),
                                                        features=q_obs[:n_train, i][..., np.newaxis],
                                                        target=q_obs_test,
                                                        data_type=DataTypesEnum.ts,
                                                        task=Task(TaskTypesEnum.ts_forecasting,
                                                                  task_params=TsForecastingParams(
                                                                      forecast_length=forecast_length)))
    for i in range(qi_arr.shape[1]):
        ds_train[f'data_source_ts/inj_{i}'] = InputData(idx=np.arange(0, n_train),
                                                        features=qi_arr[:n_train, i][..., np.newaxis],
                                                        target=q_obs_train,
                                                        data_type=DataTypesEnum.ts,
                                                        task=Task(TaskTypesEnum.ts_forecasting,
                                                                  task_params=TsForecastingParams(
                                                                      forecast_length=forecast_length)))

        ds_test[f'data_source_ts/inj_{i}'] = InputData(idx=np.arange(n_train, len(t_arr)),
                                                       features=qi_arr[n_train:, i][..., np.newaxis],
                                                       target=q_obs_test,
                                                       data_type=DataTypesEnum.ts,
                                                       task=Task(TaskTypesEnum.ts_forecasting,
                                                                 task_params=TsForecastingParams(
                                                                     forecast_length=forecast_length)))
    input_data_train = MultiModalData(ds_train)
    input_data_test = MultiModalData(ds_test)

    return input_data_train, input_data_test


def crm_fit(features: np.array, target: np.array, params: dict):
    t_arr = np.asarray(range(len(features)))
    input_series_train = [t_arr, features]

    q_obs_train = features

    n_inj = 5
    n_prd = 5

    tau = np.ones(n_prd)
    gain_mat = np.ones([n_inj, n_prd])
    gain_mat = gain_mat / (np.sum(gain_mat, 1).reshape([-1, 1]))
    qp0 = np.array([[0, 0, 0, 0, 0]])
    J = np.array([[1, 1, 1, 1, 1]]) / 10
    inputs_list = [tau, gain_mat, qp0]
    crm_model = crm(inputs_list, include_press=False)

    # Fitting
    # initial guess

    init_guess = inputs_list
    crm_model.fit_model(input_series=input_series_train,
                        q_obs=q_obs_train,
                        init_guess=init_guess)

    return crm_model


def crm_predict(fitted_model: any, features: np.array, params: dict):
    t_arr = np.asarray(range(len(features)))
    input_series_test = [t_arr, features]

    qp_pred_test = fitted_model.prod_pred(input_series=input_series_test)
    qp_pred_test = qp_pred_test[:, 0]
    return qp_pred_test, 'ts'


def get_simple_pipeline(multi_data):
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """

    inj_list = []
    prod_list = []

    for i, data_id in enumerate(multi_data.keys()):
        if 'inj_' in data_id:
            inj_list.append(PrimaryNode(data_id))
        if 'prod_' in data_id:
            lagged_node = SecondaryNode('lagged', nodes_from=[PrimaryNode(data_id)])
            lagged_node.custom_params = {'window_size': 2}

            prod_list.append(lagged_node)

    # For custom model params as initial approximation and model as function is necessary
    custom_node = SecondaryNode('custom', nodes_from=inj_list)
    custom_node.custom_params = {'model_predict': crm_predict,
                                 'model_fit': crm_fit}

    exog_pred_node = SecondaryNode('exog_ts', nodes_from=[custom_node])

    final_ens = [exog_pred_node] + prod_list

    node_final = SecondaryNode('ridge', nodes_from=final_ens)
    pipeline = Pipeline(node_final)
    pipeline.show()

    return pipeline


def run_exp():
    train_data, test_data = prepare_data()
    pipeline = get_simple_pipeline(train_data)

    pipeline.fit_from_scratch(train_data)
    pipeline.print_structure()

    predicted = pipeline.predict(test_data)
    print(predicted.predict)


if __name__ == '__main__':
    run_exp()
