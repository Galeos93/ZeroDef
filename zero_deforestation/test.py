"""Module to obtain test inferences for the Zero Deforestation challenge."""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import zero_deforestation.dataset as module_data
import zero_deforestation.data_loader.data_loaders as module_dataloader
import zero_deforestation.data_loader.augmentations as module_augmentations
import zero_deforestation.model.loss as module_loss
import zero_deforestation.model.metric as module_metric
import zero_deforestation.model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger("test")

    # set data augmentation
    aug = config.init_obj("aug", module_augmentations, train=False)

    # setup dataset instances
    dataset = config.init_obj("dataset", module_data, transform=aug, return_label=False)

    # setup data_loader instances
    data_loader = config.init_obj(
        "data_loader", module_dataloader, dataset=dataset, sampler=None
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    model_outputs = []

    with torch.no_grad():
        for _, (data) in enumerate(tqdm(data_loader)):
            images = data["image"]
            images = images.to(device)
            output = model(images)
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)

            model_outputs.append(output.cpu().detach().numpy())

    concat_output = np.concatenate(model_outputs)
    output = {"target": {str(idx): int(x) for idx, x in enumerate(concat_output)}}
    with open("predictions.json", "w") as f_hdl:
        json.dump(output, f_hdl, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config_obj = ConfigParser.from_args(args)
    main(config_obj)
