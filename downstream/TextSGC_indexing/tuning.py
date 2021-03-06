import time
import argparse
import numpy as np
import pickle as pkl
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from args import get_text_args
from utils import *
from train import train_linear
import torch
from torch.utils.tensorboard import SummaryWriter
#import torch.nn.functional as F
from models import get_model
from math import log

#torch.cuda.set_device(1)

parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
parser.add_argument('--dataset', type=str, default='20ng',
                    choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','covid_19_production','pubmed'],
                    help='dataset name')
parser.add_argument('--tokeniser', action='store',type=str, default='treebank',
                    help='tokeniser to use')
# args = parser.parse_known_args() 
# print(args)                   
args = get_text_args()
args.device = 'cuda' if args.cuda else 'cpu'
set_seed(args.seed, args.cuda)

tokeniser = args.tokeniser
best_weight_decays = []

for i in range(5):
    writer = SummaryWriter()

    sp_adj, index_dict, label_dict = load_corpus_crossval(args.dataset,i, args.tokeniser)
    adj = sparse_to_torch_sparse(sp_adj, device=args.device)

    adj_dense = sparse_to_torch_dense(sp_adj, device='cuda')
    feat_dict, precompute_time = sgc_precompute(adj, adj_dense, args.degree-1, index_dict)
    if args.dataset == "mr": nclass = 1
    else: nclass = max(label_dict["train"])+1

    def linear_objective(space):
        model = get_model(args.model, nfeat=feat_dict["train"].size(1),
                        nclass=nclass,
                        nhid=0, dropout=0, cuda=args.cuda)
        val_acc, _, _ = train_linear(model, feat_dict, space['weight_decay'], args.dataset=="mr",i)
        #print( 'weight decay ' + str(space['weight_decay']) + '\n' + \
            #'overall accuracy: ' + str(val_acc))
        # writer.add_scalar("Weight decay/tuning", space['weight_decay'])
        writer.add_scalar("Accuracy/tuning", val_acc)
        return {'loss': -val_acc, 'status': STATUS_OK}

    # Hyperparameter optimization
    space = {'weight_decay' : hp.loguniform('weight_decay', log(1e-6), log(1e-0))}

    best = fmin(linear_objective, space=space, algo=tpe.suggest, max_evals=100)
    print(best)
    writer.flush()
    writer.close()

    # add best weight decay to an array
    np.append(best_weight_decays, best['weight_decay'])

    with open('tuned_result/{}.{}.{}.SGC_ref.tuning.txt'.format(args.dataset,args.tokeniser,i), 'w') as f:
        f.write(str(best['weight_decay']))

# then show mean?
print(best_weight_decays)


