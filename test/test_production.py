import datetime
import os
from pathlib import Path

from production_forecasting.oil_production_forecasting import run_oil_forecasting


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent


def _clean(well: str):
    try:
        for i in range(4):
            os.remove(f'{well}_{i}.json')
        os.remove(f'predictions_{well}.csv')
    except OSError:
        pass


def test_simple_production_forecast():
    well = '5351'

    _clean(well)

    full_path_train_crm = f'test_data/production/oil_crm_prod_X{well}.csv'
    full_path_train_crm = os.path.join(str(project_root()), full_path_train_crm)

    file_path_train = f'test_data/production/oil_prod_X{well}.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    obtained_error = run_oil_forecasting(path_to_file=full_path_train,
                                         path_to_file_crm=full_path_train_crm,
                                         len_forecast=5,
                                         len_forecast_full=5,
                                         ax=None,
                                         well_id=well,
                                         timeout=0.01)
