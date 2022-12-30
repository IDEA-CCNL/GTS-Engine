import random
from torch.utils.data import DataLoader, Sampler, Dataset, distributed


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            offset = k*self.batch_size
            batch_indices = indices[offset:offset+self.batch_size]

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.__getclass__(idx)
                pair_indices.append(random.choice(self.dataset._classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        return (len(self.dataset)+self.batch_size-1) // self.batch_size