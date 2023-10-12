import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ProtoNetDataSet(Dataset):
    def __init__(self, data_q, data_s, n_shots):
        """
        data : tensor
            A tensor containing all data points. Shape [c, q, .., ..]
        n_shots : int
            Number of support examples per class in each episode.
        n_queries : int
            Number of query examples per class in each episode.
        n_classes : int
            Number of classes in each episode.
        """
        self.data_q = data_q
        self.data_s = data_s
        self.n_shots = n_shots
        self.n_queries = data_q.size(1)
        self.n_class = data_q.size(0)
        
        assert self.n_class == data_s.size(0)
        
    @property
    def data_s(self):
        # TODO take only n_shots for support set
        return self.data_s
    
    def __len__(self):
        return self.n_queries
    
    def __getitem__(self, idx):
        """
        Get one element from each class
        """
        return self.data_q[torch.arange(self.n_class), idx].unsqueeze(1)

        
class ProtoNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_s = dataset.data_s
    
    def __iter__(self):
        for batch in super().__iter__():
            sample = {}
            sample['xs'] = self.data_s
            sample['xq'] = batch
            yield sample
