from zero_deforestation.dataset import ZeroDeforestationDataset
from zero_deforestation.data_loader.augmentations import ImgAugTransform


class TestZeroDeforestationDataset:
    @staticmethod
    def test_given_test_dataset_iteration_is_correct(zero_deforestation_train_df_path):
        n_iterations = 100
        image_size = (332, 332)

        dataset = ZeroDeforestationDataset(
            str(zero_deforestation_train_df_path),
            image_size=image_size,
            return_label=False,
        )

        for i in range(n_iterations):
            image = dataset[i]["image"]
            assert image.shape == image_size + (3,)

    @staticmethod
    def test_given_train_dataset_iteration_is_correct(zero_deforestation_train_df_path):
        n_iterations = 100
        image_size = (332, 332)

        dataset = ZeroDeforestationDataset(
            str(zero_deforestation_train_df_path),
            image_size=image_size,
            return_label=True,
        )

        for i in range(n_iterations):
            image = dataset[i]["image"]
            label = dataset[i]["target"]
            assert image.shape == image_size + (3,)
            assert 0 <= label <= 2
