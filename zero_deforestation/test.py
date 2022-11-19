"""Module to obtain test inferences for the Zero Deforestation challenge."""

import argparse
import torch
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
    aug = config.init_obj("train_aug", module_augmentations, train=False)

    # setup dataset instances
    dataset = config.init_obj(
        "train_dataset", module_data, transform=aug, return_label=True
    )

    # setup data_loader instances
    data_loader = config.init_obj(
        "train_data_loader", module_dataloader, dataset=dataset, sampler=None
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

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

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    model_outputs = []

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            model_outputs.append(output)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for index_2, metric in enumerate(metric_fns):
                total_metrics[index_2] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    logger.info(log)


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
