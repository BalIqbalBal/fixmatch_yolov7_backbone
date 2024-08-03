from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from typing import Tuple, Any
import math


def get_sampler(
    dataset: Dataset,
    num_samples: int,
    reweighted: bool = False,
    median_freq: bool = False,
    replacement: bool = True,
):
    """
    Method that generates torch samplers.

    Parameters
    ----------
    dataset: Dataset
        Torch base dataset object from which samples are selected.
    replacement: bool
        Boolean flag indicating whether samples should be drawn with replacement or not.
    num_samples: int
        Number of samples that are drawn. Should only be specified when sampling with replacement, i.e. replacement=True
    reweighted: bool
        See get_reweighted_sampler
    median_freq: bool
        See get_reweighted_sampler

    Returns
    ----------
    sampler: Sampler (either RandomSampler or WeightedRandomSampler)
            Returns torch sampler instance, which can be used as input to a torch DataLoader. If reweighted=True,
            a WeightedRandomSampler instance is returned, while if reweighted=False a RandomSampler instance is
            returned.
    """
    if reweighted:
        return get_reweighted_sampler(dataset.targets, num_samples, replacement, median_freq)
    else:
        return get_uniform_sampler(dataset, replacement, num_samples)


def get_uniform_sampler(dataset: Dataset, replacement: bool = False, num_samples: int = None):
    """
    Method that generates samplers that randomly select samples from the dataset with equal probability.

    Parameters
    ----------
    dataset: Dataset
        Torch base dataset object from which samples are selected.
    replacement: bool
        Boolean flag indicating whether samples should be drawn with replacement or not.
    num_samples: int
        Number of samples that are drawn. Should only be specified when sampling with replacement, i.e. replacement=True

    Returns
    ----------
    random_sampler: RandomSampler
        Returns random sampler instance, which can be used as input to a torch DataLoader.
    """
    if replacement:
        return RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
    else:
        return RandomSampler(dataset, replacement=replacement)


def get_reweighted_sampler(targets: List, num_samples: int, replacement: bool = True, median_freq: bool = False):
    """
    Method that generates a weighted random sampler that selects samples with a probability inversely proportional
    to their frequency (or median frequency) in the dataset. This sampling strategy can be useful when working with
    highly imbalanced datasets.

    Parameters
    ----------
    targets: List
        List of sample targets / classes based on which sampling weights are computed.
    num_samples: int
        Number of samples that are drawn. Should only be specified when sampling with replacement, i.e. replacement=True
    replacement: bool
        Boolean flag indicating whether samples should be drawn with replacement or not.
    median_freq: bool
        Boolean flag indicating whether sample weights are computed to be inversely proportional to frequency or median
        frequency of sample class.
    Returns
    ----------
    weighted_samples: WeightedRandomSampler
        Returns the weighted sampler instance, which can be used as input to a torch DataLoader.
    """
    labels, counts = np.unique(targets, return_counts=True)
    if not median_freq:
        class_weights = 1 / (counts / np.sum(counts))
    else:
        class_weights = 1 / (counts / np.median(counts))
    sample_weights = np.zeros(len(targets))
    for class_label in labels:
        sample_weights[np.array(targets) == class_label] = class_weights[
            class_label
        ]
    num_samples = num_samples if replacement else len(targets)
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=replacement)


def create_loaders(
    args,
    train_labeled: Dataset,
    train_unlabeled: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    mu: int = 7,
    total_iters: int = None,
) -> Tuple[Tuple[DataLoader, DataLoader], DataLoader, DataLoader]:
    """
    Create data loaders for semi-supervised learning.

    Parameters
    ----------
    args: Any
        Object containing command line arguments.
    train_labeled: Dataset
        Labeled training dataset.
    train_unlabeled: Dataset
        Unlabeled training dataset.
    validation_dataset: Dataset
        Validation dataset.
    test_dataset: Dataset
        Test dataset.
    batch_size: int
        Batch size for labeled data.
    num_workers: int
        Number of subprocesses for data loading.
    mu: int
        Multiplier for unlabeled batch size.
    total_iters: int, optional
        Total number of iterations per epoch. If None, it's calculated automatically.

    Returns
    -------
    Tuple[Tuple[DataLoader, DataLoader], DataLoader, DataLoader]
        Tuple containing (labeled_loader, unlabeled_loader), validation_loader, test_loader.
    """
    if total_iters is None:
        # Calculate total_iters to cover the entire dataset at least once
        total_iters = math.ceil(max(len(train_labeled), len(train_unlabeled) / mu) / batch_size)

    num_labeled_samples = total_iters * batch_size

    labeled_sampler = get_sampler(
        train_labeled,
        num_samples=num_labeled_samples,
        reweighted=args.weighted_sampling,
    )
    
    unlabeled_sampler = get_sampler(
        train_unlabeled,
        num_samples=num_labeled_samples * mu,
        reweighted=False,
    )

    train_loader_labeled = DataLoader(
        train_labeled,
        batch_size=batch_size,
        sampler=labeled_sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=args.pin_memory
    )

    train_loader_unlabeled = DataLoader(
        train_unlabeled,
        batch_size=batch_size * mu,
        sampler=unlabeled_sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=args.pin_memory
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )

    return (train_loader_labeled, train_loader_unlabeled), validation_loader, test_loader