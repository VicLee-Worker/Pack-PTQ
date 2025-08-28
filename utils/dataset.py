import os
import torch
import torchvision
from loguru import logger
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
from PIL import Image


def build_dataset(args):
    """
    Builds the dataset and data loaders for training and validation.

    Args:
        args: Command-line arguments containing dataset and model configuration.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    model_type = args.model_name.split("_")[0]
    if "deit" in model_type:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif 'vit' in model_type:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif 'swin' in model_type:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

    # Data directories
    traindir = os.path.join(args.dataset_path, 'train')
    valdir = os.path.join(args.dataset_path, 'val')

    # Validation dataset and loader
    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Training dataset and loader
    train_dataset = datasets.ImageFolder(traindir, train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader

def build_transform(input_size=224, interpolation="bicubic",
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                   crop_pct=0.875):
    """
    Builds the transformation pipeline for the dataset.

    Args:
        input_size (int): The size of the input image.
        interpolation (str): The interpolation method for resizing.
        mean (tuple): The mean values for normalization.
        std (tuple): The standard deviation values for normalization.
        crop_pct (float): The crop percentage.

    Returns:
        transforms.Compose: The composed transformation pipeline.
    """
    def _pil_interp(method):
        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(transforms.Resize(size, interpolation=ip))
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_ImageNet_data(data_path: str = '', input_size: int = 224,
                       batch_size: int = 64, workers: int = 4) -> tuple:
    """
    Builds the ImageNet dataset and data loaders for training and validation.

    Args:
        data_path (str): The path to the ImageNet dataset.
        input_size (int): The size of the input image.
        batch_size (int): The batch size for the data loaders.
        workers (int): The number of worker processes for data loading.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True
    )

    return train_loader, val_loader

def get_train_samples(train_loader, num_samples):
    """
    Retrieves a specified number of training samples from the data loader.

    Args:
        train_loader (DataLoader): The training data loader.
        num_samples (int): The number of samples to retrieve.

    Returns:
        tuple: A tuple containing the training data and targets.
    """
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]


def prepare_calibration_data(args):
    """
    Prepare the calibration dataset and log the process.

    Args:
        args: Command-line arguments.

    Returns:
        tuple: Train loader, test loader, calibration data, and calibration targets.
    """
    logger.info("Preparing calibration dataset...")
    train_loader, test_loader = build_ImageNet_data(
        batch_size=args.batch_size,
        workers=args.num_workers,
        data_path=args.dataset_path
    )
    cali_data, cali_target = get_train_samples(train_loader, num_samples=args.calibration_samples)
    logger.info("Calibration data preparation completed")
    return train_loader, test_loader, cali_data, cali_target