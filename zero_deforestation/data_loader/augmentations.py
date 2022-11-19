"""Module with data augmentators."""

import torchvision.transforms as T


class ImgAugTransform:
    """Preprocessing that standardizes an image and converts it to a torch.Tensor."""

    def __init__(self, train):
        self.train = train

        self.transform = self.build_transforms()

    @staticmethod
    def build_transforms():

        transformations = [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        return T.Compose(transformations)

    def __call__(self, images):
        return self.transform(images)
