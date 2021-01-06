import time
import argparse
import numpy as np
import pandas as pd
import pickle
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # visualisation tool
import tabulate
from functools import partial
from utils import *
from models import SGC
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

torch.cuda.set_device(1) # When GPU 0 is out of memory

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20ng', help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=3,
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
parser.add_argument('--tokeniser', type=str, action='store',default='treebank',
                    choices=['manual', 'scispacy','ref','nltk','treebank'],
                    help='tokeniser to use')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

torch.backends.cudnn.benchmark = True
set_seed(args.seed, args.cuda)
torch.cuda.set_device(1)

test_acc = np.zeros(5)
time_arr = np.zeros(5)

for i in range(5): 

    if args.tuned:
        with open("tuned_result/{}.{}.{}.SGC_ref.tuning.txt".format(args.dataset,args.tokeniser,i), "r") as f:
            args.weight_decay = float(f.read())

    sp_adj, index_dict, label_dict = load_corpus_crossval(args.dataset,i,args.tokeniser) # loads the BCD graph (D has the BioBERT embeddings)
    for k, v in label_dict.items():
        if args.dataset == "mr":
            label_dict[k] = torch.Tensor(v).to(args.device)
        else:
            label_dict[k] = torch.LongTensor(v).to(args.device)
    features = torch.arange(sp_adj.shape[0]).to(args.device)

    adj = sparse_to_torch_sparse(sp_adj, device=args.device)
    
    def train_linear(model, feat_dict, weight_decay, binary=False):
        total_train_labels = len(label_dict["train"])
        class_weights = []
        labels = label_dict["train"].cpu().numpy()
        for c in label_dict["train"].unique().tolist(): 
            num = np.count_nonzero(labels == c)
            class_weights.append(float(num))
        max_weight = np.max(class_weights)
        class_weights =  max_weight * np.reciprocal(class_weights)
        weights = torch.Tensor(class_weights).cuda()
        writer = SummaryWriter()
        if not binary:
            act = partial(F.log_softmax, dim=1)
            criterion = F.nll_loss
        else:
            act = torch.sigmoid
            criterion = F.binary_cross_entropy
        optimizer = optim.LBFGS(model.parameters())
        best_val_loss = float('inf')
        best_val_acc = 0
        plateau = 0
        start = time.perf_counter()
        
        for epoch in range(args.epochs):
            # DataLoader - split feat_dict into batches. 
            # for each batch: 
            # train_data = data_utils.TensorDataset(feat_dict["train"].cuda(), label_dict["train"].cuda())
            # train_loader = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            # for n, (batch_feat, batch_label) in enumerate(train_loader): # make predictions in batches
            def closure():
                optimizer.zero_grad()
                output = model(feat_dict["train"].cuda()).squeeze()
                l2_reg = 0.5*weight_decay*(model.W.weight**2).sum()
                if i == 4: 
                    loss = criterion(act(output), label_dict["train"].cuda())+l2_reg
                else: 
                    loss = criterion(act(output), label_dict["train"].cuda(), weight=weights)+l2_reg # sigmoid activation function with the weighted cross entropy
                writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                return loss
            optimizer.step(closure)
        train_time = time.perf_counter()-start
        val_res, val_matrix = eval_linear(model, feat_dict["val"].cuda(),
                            label_dict["val"].cuda(), binary)     
        writer.flush()
        writer.close()
        return val_res['accuracy'], model, train_time

    def eval_linear(model, features, label, binary=False):
        model.eval()

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

        if not binary:
            act = partial(F.log_softmax, dim=1)
            criterion = F.nll_loss
        else:
            act = torch.sigmoid
            criterion = F.binary_cross_entropy

        with torch.no_grad():
            output = model(features).squeeze()
            if i == 4: 
                loss = criterion(act(output), label)
            else: 
                loss = criterion(act(output), label, weight=weights)
            if not binary: predict_class = output.max(1)[1]
            else: predict_class = act(output).gt(0.5).float()
            correct = torch.eq(predict_class, label).long().sum().item()
            acc = correct/predict_class.size(0)
            print_matrix = torch.cat([predict_class, label],0)

        # TRYING TO GET AUROC SCORES

        # y_scores = output.max(1)[0]
        # min_max_scaler = preprocessing.MinMaxScaler()
        # y_scores = min_max_scaler.fit_transform(y_scores)
        # #print(output_scaled.shape) (n_observations, 23)
        # auroc = y_scores
        # print('auroc scores', auroc)
        # y_true = label.cpu()
        # macro_auroc_score = roc_auc_score(y_true, y_scores, multi_class='ovo', average='macro')
        # w_auroc_score = roc_auc_score(y_true, y_scores, multi_class='ovo', average='weighted')
        # print('macro auroc', macro_auroc_score)
        # print('weighted auroc', w_auroc_score)

        # OPTIMISED PRECISION    

        return {
            'loss': loss.item(),
            'accuracy': acc
        }, print_matrix

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
        test_res, test_matrix = eval_linear(best_model, feat_dict["test"].cuda(),
                            label_dict["test"].cuda(), args.dataset=="mr")
        train_res, train_matrix = eval_linear(best_model, feat_dict["train"].cuda(),
                                label_dict["train"].cuda(), args.dataset=="mr")
        print("Total Time: {:2f}s, Train acc: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}".format(precompute_time+train_time, train_res["accuracy"], val_acc, test_res["accuracy"]))
        test_acc[i] = test_res["accuracy"]
        time_arr[i] = precompute_time+train_time
        test_res_file = open("results/{}.{}.SGC_ref.results.txt".format(args.dataset,i), 'w')
        printing = test_matrix.cpu().numpy()
        np.savetxt("results/{}.{}.SGC_ref.results.txt".format(args.dataset,i),printing)
        test_res_file.close()

        # test_auroc_file = open("results/{}.{}.SGC_ref.auroc.txt".format(args.dataset,i), 'w')
        # df_auroc = pd.DataFrame(test_auroc)
        # df_auroc.to_csv("results/{}.{}.SGC_ref.auroc.txt".format(args.dataset,i),index=False,header=False)
        # # np.savetxt("results/{}.{}.SGC_ref.auroc.txt".format(args.dataset,i),test_auroc)
        # test_auroc_file.close()
        
        # For PubMed - deleting big objects and clearing GPU space 
        del sp_adj
        del adj
        del adj_dense
        del feat_dict 
        del precompute_time
        del index_dict
        del label_dict
        del features
        torch.cuda.empty_cache()

if __name__ == '__main__':
    print("Mean test accuracy: {:4f}, std dev: {:4f}".format(np.mean(test_acc)*100,np.std(test_acc)*100))
    macrof1 = np.empty(5)
    weightedf1 = np.empty(5)
    op_scores = np.empty(5)
    avg_acc = np.empty(5)

    for i in range(5):
        full = pd.read_csv('results/{}.{}.SGC_ref.results.txt'.format(args.dataset,i),header=None)
        df = full.iloc[range(round(len(full)/2)),:].reset_index(drop=True)
        df = df.rename(columns = {0:"predictions"})
        df['labels'] = full.iloc[round(len(full)/2):,:].reset_index(drop=True)
        pred = df['predictions']
        true = df['labels']
        macrof1[i] = f1_score(true,pred,average='macro')
        weightedf1[i] = f1_score(true,pred,average='weighted')
        cnf_matrix = confusion_matrix(true, pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        TNR = TN/(TN+FP) 
        TPR = TP/(TP+FN)
        ACC = (TP+TN)/(TP+FP+FN+TN)

        mean_spec = np.mean(TNR)
        mean_recall = np.mean(TPR)
        mean_acc = np.mean(ACC)

        op_scores[i] = mean_acc - (abs(mean_spec - mean_recall)/(mean_spec + mean_recall))

    #print('Macro F1 scores:', macrof1)
    print('Mean Macro F1:', np.mean(macrof1)*100, ' with std: ', np.std(macrof1)*100)
    #print('Weighted F1 scores:', weightedf1)
    print('Mean Weighted F1:', np.mean(weightedf1)*100, ' with std: ', np.std(weightedf1)*100)
    #print('Avg Accuracy Scores:', avg_acc)
    #print('Mean Avg Accuracy:', np.mean(avg_acc)*100, ' with std: ', np.std(avg_acc)*100)
    #print('OP Scores:', op_scores)
    print('Mean OP:', np.mean(op_scores)*100, ' with std: ', np.std(op_scores)*100)

    print("Mean train time: {:2f}s, std dev: {:4f}s".format(np.mean(time_arr), np.std(time_arr)))
    
