import datetime
import os
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtw import dtw
from fedot.core.composer.gp_composer.fixed_structure_composer \
    import FixedStructureComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements
from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.quality_metrics_repository import MetricsRepository, \
    RegressionMetricsEnum
from sklearn.metrics import mean_squared_error as mse


def project_root():
    """Returns project root folder."""
    return f'{Path(__file__).parent.parent}/production_forecasting/'


def get_comp_pipeline(exp_id: str, data: InputData,
                      max_time=datetime.timedelta(minutes=10)):
    """ Generate the model using evolutionary composer
    :param exp_id: id of the experiment
    :param data: dataset of the building of composite model
    :param max_time: maximal optimisation time
    :return: the model with optimal structure
    """
    return Pipeline(PrimaryNode('rfr'))
    if os.path.exists(f'{exp_id}.json'):
        pipeline = Pipeline()
        pipeline.load(f'{exp_id}.json')
        return pipeline

    node_first = PrimaryNode('dtreg')
    node_second = PrimaryNode('rfr')

    node_final = SecondaryNode('linear',
                               nodes_from=[node_first, node_second])
    pipeline = Pipeline(node_final)

    available_model_types = ['rfr', 'linear', 'ridge', 'lasso', 'dtreg', 'knnreg']

    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types, max_arity=2,
        max_depth=1, pop_size=10, num_of_generations=10,
        crossover_prob=0, mutation_prob=0.8,
        timeout=max_time)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    builder = FixedStructureComposerBuilder(task=data.task).with_requirements(composer_requirements).with_metrics(
        metric_function).with_initial_pipeline(pipeline)
    composer = builder.build()

    pipeline = composer.compose_pipeline(data=data,
                                         is_visualise=False)

    ts_pipeline = Pipeline(pipeline.root_node)

    print([str(_) for _ in ts_pipeline.nodes])

    ts_pipeline.save(f'{exp_id}.json')
    return ts_pipeline


def get_crm_prediction_with_intervals(well_name):
    file_path_crm = f'./input_data/crmip.csv'
    file_path_crm = os.path.join(str(project_root()), file_path_crm)

    data_frame = pd.read_csv(file_path_crm, sep=',')
    crm = data_frame[f'mean_{well_name}'][(300 - 1):700]
    crm[np.isnan(crm)] = 0
    return crm


def calculate_validation_metric(pred: OutputData, pred_crm, pred_crm_opt, valid: InputData,
                                name: str, is_visualise=False):
    forecast_length = pred.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[~np.isnan(pred.predict)]
    predicted_crm = pred_crm.predict[~np.isnan(pred.predict)]
    predicted_crm_opt = pred_crm_opt.predict[~np.isnan(pred.predict)]

    real = valid.target[~np.isnan(pred.predict)]

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

    return (rmse_crm, rmse_ml, rmse_ml_crm, rmse_crm_opt,
            dtw_crm, dtw_ml, dtw_ml_crm, dtw_crm_opt)


def get_crm_intervals(model_name):
    file_path_crm = f'./input_data/crmip.csv'
    file_path_crm = os.path.join(str(project_root()), file_path_crm)

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
