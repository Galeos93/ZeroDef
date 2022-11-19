from os import stat
import pytest

import torch

from zero_deforestation.model import metric


class TestAccuracy:
    @staticmethod
    def test_given_input_output_is_correct():
        target = torch.tensor([1, 1, 1, 1])
        output = torch.tensor([-10, -10, 10, 10])
        output_metric = metric.accuracy(output, target)
        assert output_metric == 0.5


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
