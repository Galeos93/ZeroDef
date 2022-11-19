"""Module with torch DataLoaders."""

from torch.utils.data.dataloader import DataLoader


class ZeroDeforestationDataLoader(DataLoader):
    """Data Loader for Zero Deforestation challenge data."""

    def __init__(
        self,
        dataset,
        sampler,
        nworkers=4,
        batch_size=64,
        shuffle=True,
    ):
        self.n_samples = len(dataset)
        self.batch_size = batch_size
        super().__init__(
            dataset=dataset,
            num_workers=nworkers,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
        )
