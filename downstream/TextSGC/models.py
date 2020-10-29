import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class SGC(nn.Module):
    def __init__(self, nfeat, nclass, bias=False):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass, bias=bias)
        torch.nn.init.xavier_normal_(self.W.weight)

    def forward(self, x):
        out = self.W(x)
        return out

def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=True):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model