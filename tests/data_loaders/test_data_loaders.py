from zero_deforestation.data_loader.data_loaders import ZeroDeforestationDataLoader
from zero_deforestation.data_loader.augmentations import ImgAugTransform
from zero_deforestation.dataset import ZeroDeforestationDataset


class TestZeroDeforestationDataLoader:
    @staticmethod
    def test_given_data_loader_output_is_correct(zero_deforestation_train_df_path):
        n_iterations = 10
        image_size = (332, 332)

        transformation = ImgAugTransform(train=True)

        dataset = ZeroDeforestationDataset(
            str(zero_deforestation_train_df_path),
            image_size=image_size,
            return_label=True,
            transform=transformation,
        )

        data_loader = ZeroDeforestationDataLoader(
            dataset=dataset,
            sampler=None,
            batch_size=32,
        )

        for batch_idx, data in enumerate(data_loader):
            assert data["image"].shape == (32, 3, 332, 332)
            assert data["target"].shape == (32,)
            if batch_idx == n_iterations:
                break
