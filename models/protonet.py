import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .blocks import conv_block, Flatten
from utils.dist import euclidean_dist

class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder

    @classmethod
    def defualt_encoder(cls, x_dim=1, hid_dim=64, z_dim=64):
        encoder = nn.Sequential(
            conv_block(x_dim[0], hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )
        return cls(encoder)
    
    def loss(self, sample):
        ''' prtototypical loss

            params:
                sample: dict
                    xs: support set (n_class, n_query, ...)
                    xq: query set (n_class, n_query, ...)
        '''

        xs = Variable(sample['xs']) 
        xq = Variable(sample['xq'])

        n_class = xs.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)

        assert xq.size(0) == n_class

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
    
    def predict(self, sample):
        ''' prtototypical prediction

            params:
                sample: dict
                    xs: support set (n_class, n_query, ...)
                    xq: query set (n_query, ...)
        '''

        xs = Variable(sample['xs']) 
        xq = Variable(sample['xq'])
    
        n_class = xs.size(0)
        n_support = xs.size(1)
        n_query = xq.size(0)

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                    xq.view(n_query, *xq.size()[2:])], 0)
        
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]  

        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        _, y_hat = log_p_y.max(2)

        return y_hat