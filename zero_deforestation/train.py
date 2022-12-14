"""Main script to train a model."""

import argparse
import collections
import torch
import numpy as np
import zero_deforestation.dataset as module_data
import zero_deforestation.data_loader.data_loaders as module_dataloader
import zero_deforestation.data_loader.augmentations as module_augmentations
import zero_deforestation.model.loss as module_loss
import zero_deforestation.model.metric as module_metric
import zero_deforestation.model.model as module_arch
from zero_deforestation.parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # set data augmentation
    train_aug = config.init_obj("train_aug", module_augmentations, train=True)
    val_aug = config.init_obj("train_aug", module_augmentations, train=False)

    # setup dataset instances
    train_dataset = config.init_obj(
        "train_dataset", module_data, transform=train_aug, return_label=True
    )
    val_dataset = config.init_obj(
        "val_dataset", module_data, transform=val_aug, return_label=True
    )

    # setup data_loader instances
    data_loader = config.init_obj(
        "train_data_loader", module_dataloader, dataset=train_dataset, sampler=None
    )
    valid_data_loader = config.init_obj(
        "val_data_loader", module_dataloader, dataset=val_dataset, sampler=None
    )

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing
    # lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


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

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config_obj = ConfigParser.from_args(args, options)
    main(config_obj)
