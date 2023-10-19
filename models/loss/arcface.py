import torch.nn as nn
import torch
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, feature_in, feature_out=24, margin=0.3, scale = 64):
        '''
            params:
                feature_in: embed sized
                feature_out: number of classes
        '''
        super().__init__()
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.scale = scale
        self.margin = margin 
        
        self.weights = nn.Parameter(torch.FloatTensor(feature_out,feature_in))
        nn.init.xavier_normal_(self.weights)

    def forward(self, features, targets):
        cos_theta = F.linear(features, F.normalize(self.weights), bias=None)
        cos_theta = cos_theta.clip(-1+1e-7, 1-1e-7)
        
        arc_cos = torch.acos(cos_theta)
        M = F.one_hot(targets, num_classes = self.feature_out) * self.margin
        arc_cos += M
        
        cos_theta_2 = torch.cos(arc_cos)
        logits = cos_theta_2 * self.scale
        return logits