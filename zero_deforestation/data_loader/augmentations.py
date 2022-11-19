import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self, train):
        self.train = train

        self.transform = self.build_transforms()

    def build_transforms(self):

        transformations = [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        return T.Compose(transformations)

    def __call__(self, images):
        return self.transform(images)
