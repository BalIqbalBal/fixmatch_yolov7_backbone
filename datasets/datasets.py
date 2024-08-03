import random
from typing import List, Tuple, Callable

from torchvision import transforms
from torch.utils.data import Dataset

from datasets.config import *
from datasets.custom_datasets import *

import numpy as np
import random
from typing import List, Tuple


DATASET_GETTERS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "imagenet": datasets.ImageNet,
    "svhn": datasets.SVHN,
    "stl10": STL10,
    "caltech101": Caltech101,
    "caltech256": Caltech256,
    "ham10000": HAM10000,
    "yolodataset": YOLODATASET,
}


def get_datasets(
        root_dir: str,
        dataset: str,
        num_labeled: int,
        num_validation: int = 1,
        labeled_train_transform: Callable = None,
        unlabeled_train_transform: Callable = None,
        test_transform: Callable = None,
        download: bool = True,
        dataset_indices: Optional[List] = None
):
    """
    Method that returns all dataset objects required for semi-supervised learning: labeled train set, unlabeled train
    set, validation set and test set. The returned dataset objects can be used as input for data loaders used during
    model training.

    Parameters
    ----------
    root_dir: str
        Path to root data directory to load datasets from or download them to, if not downloaded yet, e.g. `./data`.
    dataset: str
        Name of dataset, e.g. `cifar10`, `imagenet`, etc.
    num_labeled: int
        Number of samples selected for the labeled training set for semi-supervised learning. These samples are
        selected from all training samples.
    num_validation: int
        Number of samples selected for the validation set. These samples are selected from all available
        training samples.
    labeled_train_transform: Callable
        Transform / augmentation strategy applied to the labeled training set.
    unlabeled_train_transform: Callable
        Transform / augmentation strategy applied to the unlabeled training set.
    test_transform: Callable
        Transform / augmentation strategy applied to the validation and test set.
    download: bool
        Boolean indicating whether the dataset should be downloaded or not. If yes, the get_base_sets method will
        download the dataset to the root dir if possible. Automatic downloading is supported for CIFAR-10, CIFAR-100,
        STL-10 and ImageNet.
    dataset_indices: Optional[Dict]
        Dictionary containing indices for the labeled and unlabeled training sets, validation set and test set for
        initialization. This argument should be used if training is resumed, i.e. initializing the dataset splits to
        the same indices as in the previous training run, and dataset indices are loaded. An alternative use case,
        would be to select initial indices in a principled way, e.g. selecting diverse initial samples based on
        representations provided by self-supervised learning.
    Returns
    -------
    dataset_tuple: Tuple[Dict, List, List]
        Returns tuple containing dataset objects of all relevant datasets. The first tuple element is a dictionary
        containing the labeled training dataset at key `labeled` and the unlabeled training dataset at key unlabeled.
        The second and third elements are the validation dataset and the test dataset.
    """
    base_set, test_set = get_base_sets(
        dataset, root_dir, download=download, test_transform=test_transform
    )

    base_indices = list(range(len(base_set)))
    if dataset_indices is None:
        if dataset != 'stl10':
            num_training = 1 - num_validation
            train_indices, validation_indices = get_uniform_split(base_set.targets, base_indices, split_pct=num_training)
            labeled_indices, unlabeled_indices = get_uniform_split(base_set.targets, train_indices, split_pct=num_labeled)
        else:
            labeled_indices, unlabeled_indices, validation_indices = sample_stl10_ssl_indices(
                base_set.targets,
                base_set.labeled_indices,
                base_set.unlabeled_indices,
                num_validation,
                num_labeled
            )
    else:
        labeled_indices, unlabeled_indices, validation_indices = (
            dataset_indices["train_labeled"],
            dataset_indices["train_unlabeled"],
            dataset_indices["validation"],
        )

    labeled_train_set = CustomSubset(
        base_set, labeled_indices, transform=labeled_train_transform
    )
    unlabeled_train_set = CustomSubset(
        base_set, unlabeled_indices, transform=unlabeled_train_transform
    )
    validation_set = CustomSubset(
        base_set, validation_indices, transform=test_transform
    )

    return (
        {"labeled": labeled_train_set, "unlabeled": unlabeled_train_set},
        validation_set,
        test_set,
    )


def get_base_sets(dataset, root_dir, download=True, test_transform=None):
    base_set = DATASET_GETTERS[dataset](root_dir, train=True, download=download)
    test_set = DATASET_GETTERS[dataset](
        root_dir, train=False, download=download, transform=test_transform
    )
    return base_set, test_set

def get_uniform_split(targets: List, indices: List, split_pct: float = None, split_num: int = None) -> Tuple[List, List]:
    """
    Method that splits provided train_indices uniformly according to targets / class labels.

    Parameters
    ----------
    targets: List
        List of targets / class labels corresponding to provided indices of dataset.
    indices: List
        List of dataset indices on which split should be performed.
    split_num: int
        Number of total samples selected for first split.
    split_pct: float
        Percentage of all indices which are selected for the first split.

    Returns
    -------
    split_indices: Tuple[List, List]
        Returns two lists, which contain the indices split according to the parameters split_num or split_pct.
    """

    if split_pct is not None and split_num is not None:
        raise ValueError('Expected either split_pct or split_num, not both.')
    
    if split_pct is None and split_num is None:
        raise ValueError('Expected either split_pct or split_num to be not None.')

    unique_targets = np.unique(targets)
    num_classes = len(unique_targets)

    if split_pct is not None:
        split_num = int(split_pct * len(indices))

    target_array = np.array(targets)
    indices_array = np.array(indices)

    split0_indices, split1_indices = [], []
    for class_label in unique_targets:
        class_indices = indices_array[target_array[indices_array] == class_label]
        np.random.shuffle(class_indices)
        samples_for_class = min(len(class_indices), split_num // num_classes)
        split0_indices.extend(class_indices[:samples_for_class])
        split1_indices.extend(class_indices[samples_for_class:])

    # Adjust split0_indices to match split_num exactly
    if len(split0_indices) < split_num:
        additional_needed = split_num - len(split0_indices)
        additional_indices = np.random.choice(split1_indices, size=additional_needed, replace=False)
        split0_indices.extend(additional_indices)
        split1_indices = list(set(split1_indices) - set(additional_indices))
    
    return split0_indices, split1_indices