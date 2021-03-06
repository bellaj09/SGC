import time
import argparse
import numpy as np
import pickle
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import tabulate
from functools import partial
from utils import *
from models import SGC

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20ng', help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=120,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='training batch size.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')
parser.add_argument('--preprocessed', action='store_true',
                    help='use preprocessed data')
parser.add_argument('--tokeniser', action='store',type=str, default='treebank',
                    choices=['manual', 'scispacy','ref','nltk','treebank'],
                    help='tokeniser to use')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'
if args.tuned:
    with open("tuned_result/{}.SGC.tuning.txt".format(args.dataset), "r") as f:
        args.weight_decay = float(f.read())

torch.backends.cudnn.benchmark = True
set_seed(args.seed, args.cuda)

#sp_adj, index_dict, label_dict = load_corpus(args.dataset)


def train_linear(model, feat_dict, weight_decay, binary=False,i=0):
    sp_adj, index_dict, label_dict = load_corpus_crossval(args.dataset,i,args.tokeniser)
    for k, v in label_dict.items():
        if args.dataset == "mr":
            label_dict[k] = torch.Tensor(v).to(args.device)
        else:
            label_dict[k] = torch.LongTensor(v).to(args.device)

    ######### For weighted cross entropy #######
    
    total_train_labels = len(label_dict["train"])
    class_weights = []
    labels = label_dict["train"].cpu().numpy()
    for c in label_dict["train"].unique().tolist(): 
        num = np.count_nonzero(labels == c)
        class_weights.append(float(num))
    max_weight = np.max(class_weights)
    class_weights =  max_weight * np.reciprocal(class_weights)
    weights = torch.Tensor(class_weights).cuda()
    ################################################

    features = torch.arange(sp_adj.shape[0]).to(args.device)

    adj = sparse_to_torch_sparse(sp_adj, device=args.device)
    if not binary:
        act = partial(F.log_softmax, dim=1)
        criterion = F.nll_loss
    else:
        act = torch.sigmoid
        criterion = F.binary_cross_entropy
    optimizer = optim.LBFGS(model.parameters())
    #optimizer = optim.Adamax(model.parameters())
    best_val_loss = float('inf')
    best_val_acc = 0
    plateau = 0
    start = time.perf_counter()

    ####### REFERENCE
    for epoch in range(args.epochs):
        def closure():
            optimizer.zero_grad()
            output = model(feat_dict["train"].cuda()).squeeze()
            l2_reg = 0.5*weight_decay*(model.W.weight**2).sum()
            loss = criterion(act(output), label_dict["train"].cuda())+l2_reg # Reference
            # if i == 4: 
            #         loss = criterion(act(output), label_dict["train"].cuda())+l2_reg
            # else: 
            #     loss = criterion(act(output), label_dict["train"].cuda(), weight=weights)+l2_reg
            loss.backward()
            return loss

        optimizer.step(closure)

    ## BATCHING

    # for epoch in range(args.epochs):
    #     ### DataLoader - split feat_dict into batches. 
    #     train_data = data_utils.TensorDataset(feat_dict["train"].cuda(), label_dict["train"].cuda())
    #     train_loader = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    #     for n, (batch_feat, batch_label) in enumerate(train_loader): # make predictions in batches
    #         def closure():
    #             optimizer.zero_grad()
    #             output = model(batch_feat.cuda()).squeeze()
    #             l2_reg = 0.5*weight_decay*(model.W.weight**2).sum()
    #             loss = criterion(act(output), batch_label.cuda())+l2_reg
    #             writer.add_scalar("Loss/train", loss, epoch)
    #             loss.backward()
    #             return loss
    #         optimizer.step(closure)

    train_time = time.perf_counter()-start
    val_res = eval_linear(model, feat_dict["val"].cuda(),
                          label_dict["val"].cuda(), binary,i)
    return val_res['accuracy'], model, train_time

def eval_linear(model, features, label, binary=False, i=0):
    ######### For weighted cross entropy #######
    total_train_labels = len(label)
    class_weights = []
    labels = label.cpu().numpy()
    for c in label.unique().tolist(): 
        num = np.count_nonzero(labels == c)
        class_weights.append(float(num))
    max_weight = np.max(class_weights)
    class_weights =  max_weight * np.reciprocal(class_weights)
    weights = torch.Tensor(class_weights).cuda()
    ################################################

    model.eval()
    if not binary:
        act = partial(F.log_softmax, dim=1)
        criterion = F.nll_loss
    else:
        act = torch.sigmoid
        criterion = F.binary_cross_entropy

    with torch.no_grad():
        output = model(features).squeeze()
        loss = criterion(act(output), label)
        # if i == 4: 
        #     loss = criterion(act(output), label)
        # else: 
        #     loss = criterion(act(output), label, weight=weights)
        if not binary: predict_class = output.max(1)[1]
        else: predict_class = act(output).gt(0.5).float()
        correct = torch.eq(predict_class, label).long().sum().item()
        acc = correct/predict_class.size(0)

    return {
        'loss': loss.item(),
        'accuracy': acc
    }
if __name__ == '__main__':
    if args.dataset == "mr": nclass = 1
    else: nclass = label_dict["train"].max().item()+1
    if not args.preprocessed:
        adj_dense = sparse_to_torch_dense(sp_adj, device='cpu')
        feat_dict, precompute_time = sgc_precompute(adj, adj_dense, args.degree-1, index_dict)
    else:
        # load the relased degree 2 features
        with open(os.path.join("preprocessed",
            "{}.pkl".format(args.dataset)), "rb") as prep:
            feat_dict =  pkl.load(prep)
        precompute_time = 0

    model = SGC(nfeat=feat_dict["train"].size(1),
                nclass=nclass)
    if args.cuda: model.cuda()
    val_acc, best_model, train_time = train_linear(model, feat_dict, args.weight_decay, args.dataset=="mr")
    test_res = eval_linear(best_model, feat_dict["test"].cuda(),
                           label_dict["test"].cuda(), args.dataset=="mr")
    train_res = eval_linear(best_model, feat_dict["train"].cuda(),
                            label_dict["train"].cuda(), args.dataset=="mr")
    print("Total Time: {:2f}s, Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}".format(precompute_time+train_time, train_res["accuracy"], val_acc, test_res["accuracy"]))
