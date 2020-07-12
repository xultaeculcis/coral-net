import torch
from torch.utils.data import Sampler


#  --- Sampler  ----
class ImbalancedDatasetSampler(Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    Original implementation: https://github.com/ufoym/imbalanced-dataset-sampler

    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        super().__init__(dataset)
        self.dataset = dataset
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(idx)]
                   for idx in self.indices]
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    def _get_label(self, idx):
        return self.dataset.label_for(idx)

    def __iter__(self):
        return (self.indices[i.item()] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
