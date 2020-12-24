import argparse
from nltk.tokenize import sent_tokenize
import re

parser = argparse.ArgumentParser(description='Prepare corpora for training BioBERT')
parser.add_argument('--dataset', type=str, default='20ng',
                    choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'yelp', 'ag_news', 'covid_19_production','pubmed'],
                    help='dataset name')

args = parser.parse_args()

dataset = args.dataset

doc_content_list = []
f = open('../data/corpus/' + dataset + '.txt', 'rb')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))

with open('data/corpus/'+ dataset + '_trainbiobert.txt', 'w') as f:
    for doc in doc_content_list:
        sent = sent_tokenize(doc)
        for i,s in enumerate(sent):
            s = s.strip().lower()
            s = re.sub(r'[^a-zA-Z0-9  -]',r'',s)
            sent[i] = s
        #sent.append('\n') # empty line at end of doc.
        f.write('\n'.join(sent)) # write each sentence on a new line
        f.write(" \n") # empty line at end of doc.