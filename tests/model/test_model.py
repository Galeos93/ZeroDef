from zero_deforestation.data_loader import data_loaders, augmentations
from zero_deforestation.model import model


class TestCNNModel:
    @staticmethod
    def test_given_input_from_dataloader_output_is_correct(
        zero_deforestation_train_df_path,
    ):
        image_size = (332, 332)
        batch_size = 8
        n_classes = 3
        n_iterations = 10

        transformation = augmentations.ImgAugTransform(train=True)

        dataset = data_loaders.ZeroDeforestationDataset(
            str(zero_deforestation_train_df_path),
            image_size=image_size,
            return_label=True,
            transform=transformation,
        )

        data_loader = data_loaders.ZeroDeforestationDataLoader(
            dataset=dataset,
            sampler=None,
            batch_size=batch_size,
        )

        for batch_idx, data in enumerate(data_loader):
            model_instance = model.CNNModel(n_classes)
            output = model_instance(data["image"])
            assert output.shape == (batch_size, n_classes)

            if batch_idx == n_iterations:
                break
