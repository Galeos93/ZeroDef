import pathlib
import pytest

from zero_deforestation import data


@pytest.fixture()
def data_path():
    path = pathlib.Path(data.__file__).parent
    return path


@pytest.fixture()
def zero_deforestation_train_df_path(data_path):
    df_path = data_path / "train.csv"
    return df_path
