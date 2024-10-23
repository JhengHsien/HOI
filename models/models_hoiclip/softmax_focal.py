import torch
import torch.nn as nn
import torch.nn.functional as F



class SoftmaxFocalLoss(nn.Module):
    def __init__(self,gamma=2.0,weight=None,reduction="mean"):
        super().__init__()
        self.weight=weight
        self.gamma=gamma
        assert reduction in ["sum","mean","none"]
        self.reduction=reduction

    def forward(self,pred,target, weights=None, gamma=2.0):
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        self.gamma = gamma
        ce = F.cross_entropy(pred, target,reduction="none").view(-1)
        pt=torch.exp(-ce)

        if self.weight!=None:
            target=target.view(-1)
            weights=self.weight[target]
        else:
            weights=torch.ones_like(target)
        weights = weights.sum()

        focal=weights*((1-pt)**self.gamma)
        if self.reduction=="mean":
            return (focal*ce).sum()/weights.sum()
        
        elif self.reduction=="sum":
            return (focal*ce).sum()
        
        else:
            return focal*ce