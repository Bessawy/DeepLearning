import torch


def euclidean_dist(x: torch.Tensor, y: torch.Tensor):
    ''' Compute euclidean distance between two tensors
        
        params:
            x: n x d
            y: m x d
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
