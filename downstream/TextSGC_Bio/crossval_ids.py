from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, loadWord2Vec
import sys
import argparse
import random
from collections import Counter
import pandas as pd
import pickle as pkl
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description='Get train/test ids for each fold, save into ind.dataset.phase.x/y')
parser.add_argument('--dataset', type=str, default='20ng',
                    choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','covid_19_production'],
                    help='dataset name')
args = parser.parse_args()

dataset = args.dataset

df = pd.read_csv('data/' + dataset + '.txt', sep='\t', header=None)
X = df[0]
Y = df[2]
skf = StratifiedKFold(n_splits=5)
for i, arrays in enumerate(skf.split(X, Y)): 
    train_index = arrays[0]
    test_index = arrays[1]
    df.iloc[[train_index],1] = 'train'
    df.iloc[[test_index],1] = 'test'
    filename = '{}{}.txt'.format(dataset,i)
    print(filename)
    df.to_csv(filename,sep='\t', header=False,index=False)

for i in range(5): 
    train_val_ids = []
    test_ids = []
    doc_name_list = []

    with open('data/' + dataset + str(i) + '.txt', 'r') as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            doc_name_list.append(line.strip())
            _, data_name, data_label = line.strip().split("\t")
            if data_name.find('test') != -1:
                test_ids.append(id)
            elif data_name.find('train') != -1:
                train_val_ids.append(id)
            label_names.add(data_label)
        label_names = list(label_names)
        label_names_to_index = {name:i for i, name in enumerate(label_names)}
        for id, line in enumerate(lines):
            _, data_name, data_label_name = line.strip().split("\t")
            if data_name.find('test') != -1:
                test_labels.append(label_names_to_index[data_label_name])
            elif data_name.find('train') != -1:
                train_val_labels.append(label_names_to_index[data_label_name])

    with open('data/corpus/' + dataset + '_labels.txt', 'w') as f:
        f.write('\n'.join(label_names))

    print("Loaded labels and indices", i)


    # split training and validation. validation is a random 10% of the training set
    idx = list(range(len(train_val_labels)))
    random.shuffle(idx)
    train_val_ids = [train_val_ids[i] for i in idx]
    train_val_labels = [train_val_labels[i] for i in idx]

    idx = list(range(len(test_labels)))
    random.shuffle(idx)
    test_ids = [test_ids[i] for i in idx]
    test_labels = [test_labels[i] for i in idx]

    train_val_size = len(train_val_ids)
    val_size = int(0.1 * train_val_size)
    train_size = train_val_size - val_size
    train_ids, val_ids = train_val_ids[:train_size], train_val_ids[train_size:]
    train_labels, val_labels = train_val_labels[:train_size], train_val_labels[train_size:]

    # dump objects
    f = open("data/ind.{}.{}.{}.x".format(dataset, i, "train"), 'wb')
    pkl.dump(train_ids, f)
    f.close()

    f = open("data/ind.{}.{}.{}.y".format(dataset, i, "train"), 'wb')
    pkl.dump(train_labels, f)
    f.close()

    f = open("data/ind.{}.{}.{}.x".format(dataset, i, "val"), 'wb')
    pkl.dump(val_ids, f)
    f.close()

    f = open("data/ind.{}.{}.{}.y".format(dataset, i, "val"), 'wb')
    pkl.dump(val_labels, f)
    f.close()

    f = open("data/ind.{}.{}.{}.x".format(dataset, i, "test"), 'wb')
    pkl.dump(test_ids, f)
    f.close()

    f = open("data/ind.{}.{}.{}.y".format(dataset, i, "test"), 'wb')
    pkl.dump(test_labels, f)
    f.close()