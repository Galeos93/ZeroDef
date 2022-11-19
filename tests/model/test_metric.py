from os import stat
import pytest

import torch

from zero_deforestation.model import metric


class TestF1Score:
    @staticmethod
    @pytest.mark.parametrize(
        "groundtruth,prediction,expected_score",
        ([]),
        ([]),
    )
    def test_given_grountruth_predictions_f1_score_is_expected(
        groundtruth,
        prediction,
        expected_score,
    ):
        pass
