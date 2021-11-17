from os.path import join as join

import numpy as np
import pandas as pd
from fedot.core.data.data import InputData
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
    percent_train = 0.7

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
    n_test = len(t_arr) - int(percent_train * len(t_arr))

    q_obs_train = q_obs[:n_train, 0]
    q_obs_test = q_obs[n_train:, 0]

    input_data_train = InputData(idx=np.arange(0, n_train),
                                 features=qi_arr[:n_train, 0],
                                 target=q_obs_train,
                                 data_type=DataTypesEnum.ts,
                                 task=Task(TaskTypesEnum.ts_forecasting,
                                           task_params=TsForecastingParams(forecast_length=1)))

    input_data_test = InputData(idx=np.arange(n_train, len(t_arr)),
                                features=qi_arr[n_train:, 0],
                                target=q_obs_test,
                                data_type=DataTypesEnum.ts,
                                task=Task(TaskTypesEnum.ts_forecasting,
                                          task_params=TsForecastingParams(forecast_length=1)))

    return input_data_train, input_data_test


def crm_func(train_data, test_data, params):
    input_series_train = train_data.features
    input_series_test = test_data.features

    q_obs_train = train_data.target

    # InjList = [x for x in qi.keys() if x.startswith('I')]
    # PrdList = [x for x in qp.keys() if x.startswith('P')]

    N_inj = 5
    N_prd = 5

    tau = np.ones(N_prd)
    gain_mat = np.ones([N_inj, N_prd])
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

    qp_pred_test = crm_model.prod_pred(input_series=input_series_test)

    return qp_pred_test, 'ts'


def get_simple_pipeline():
    """
        Pipeline looking like this
        lagged -> custom -> ridge
    """
    lagged_node = PrimaryNode('lagged')
    lagged_node.custom_params = {'window_size': 5}

    # For custom model params as initial approximation and model as function is necessary
    custom_node = SecondaryNode('custom', nodes_from=[lagged_node])
    custom_node.custom_params = {'model': crm_func}

    node_final = SecondaryNode('ridge', nodes_from=[custom_node])
    pipeline = Pipeline(node_final)

    return pipeline


def run_exp():
    train_data, test_data = prepare_data()
    pipeline = get_simple_pipeline()

    pipeline.fit_from_scratch(train_data)
    pipeline.print_structure()

    predicted = pipeline.predict(test_data)
    print(predicted)


if __name__ == '__main__':
    run_exp()
