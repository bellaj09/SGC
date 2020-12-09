from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
#from nltk.stem import WordNetLemmatizer 
from utils import clean_str, loadWord2Vec, clean_str_manual, clean_str_scispacy 
import argparse
import random
from collections import Counter
import time
from tqdm import tqdm
import spacy
import re

# nltk.download()
stop_words = set(stopwords.words('english'))
print(stop_words)

parser = argparse.ArgumentParser(description='Build Document Graph')
parser.add_argument('--dataset', type=str, default='20ng',
                    choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','covid_19_production','pubmed'],
                    help='dataset name')
parser.add_argument('--tokeniser', type=str, default='ref',
                    choices=['manual', 'scispacy','ref'],
                    help='tokeniser to use')
args = parser.parse_args()

dataset = args.dataset

train_val_ids = []
test_ids = []

with open('../data/' + dataset + '.txt', 'r') as f:
    lines = f.readlines()
    for id, line in enumerate(lines):
        _, data_name, data_label = line.strip().split("\t")
        if data_name.find('test') != -1:
            test_ids.append(id)
        elif data_name.find('train') != -1:
            train_val_ids.append(id)

idx = list(range(len(train_val_ids)))
random.shuffle(idx)
train_val_ids = [train_val_ids[i] for i in idx]

idx = list(range(len(test_ids)))
random.shuffle(idx)
test_ids = [test_ids[i] for i in idx]

train_val_size = len(train_val_ids)
val_size = int(0.1 * train_val_size)
train_size = train_val_size - val_size
train_ids, val_ids = train_val_ids[:train_size], train_val_ids[train_size:]

doc_content_list = []
f = open('../data/corpus/' + dataset + '.txt', 'rb')
for line in f.readlines():
    doc_content_list.append(line.strip().decode('latin1'))
f.close()

with open('data/ind.train.ids', "w") as f:
    f.write('\n'.join([str(i) for i in train_ids]))
with open('data/ind.val.ids', "w") as f:
    f.write('\n'.join([str(i) for i in val_ids]))
with open('data/ind.test.ids', "w") as f:
    f.write('\n'.join([str(i) for i in test_ids]))

# doc_content_list becomes a list of every test/train abstract in the txt, latin1 decoded and trailing/leading whitespaces removed

def get_clean_words(docs):
    clean_words = []
    #lemmatizer = WordNetLemmatizer() 
    progress_bar = tqdm(docs)
    progress_bar.set_postfix_str("tokenising documents")

    if args.tokeniser == "scispacy": # Load model once if using scispacy
        nlp = spacy.load("en_core_sci_lg")

    for doc in progress_bar:
        
        if args.tokeniser == "manual":
            temp = clean_str_manual(doc).split()
            temp = list(filter(lambda x : x not in stop_words, temp))
            
        elif args.tokeniser == "scispacy":
            doc = doc.strip().lower() # lowercase
            doc = re.sub(r'[?|$|.|!|,]',r'',doc) 
            doc = re.sub(r"\s{2,}", " ", doc) # remove duplicate whitespaces
            doc_temp = nlp(doc)
            temp = [token.text for token in doc]
            temp = list(filter(lambda x : x not in stop_words, temp))

        elif args.tokeniser == "ref":
            temp = clean_str(doc).split()
            temp = list(filter(lambda x : x not in stop_words, temp))

        # Lemmatisation of all words in temp. 
        # for i in range(len(temp)):
        #     current_word = temp[i]
        #     temp[i] = lemmatizer.lemmatize(current_word)

        clean_words.append(temp)
    return clean_words
start = time.time()
clean_words = get_clean_words(doc_content_list) 
end = time.time()
print("Tokenisation time: {}s".format(end - start))


# clean_words is an array of all the abstracts, each has its words listed, split by whitespace

word_freq = Counter() # initialising as a Counter object
# total = 0
for i in train_ids+test_ids+val_ids:
    doc_words = clean_words[i]
    word_freq.update(doc_words) 
    
# goes through each train/test/val abstract and updates the global vocab tally with the words contained.

vocab, count = zip(*word_freq.most_common()) # counting frequency of all the words. UNzips into two iterables - word in vocab | count of that word
if dataset == "mr":
    cutoff = -1
else:
    cutoff = count.index(5) # sets cutoff to be the words that occur at least 5 times!!!!!!!

vocab = set(vocab[:cutoff]) 

clean_docs = []
for words in clean_words: # Loops through every single abstract's cleaned words
    closed_words = [w for w in words if w in vocab ] # an array of the words in each abstract, if they are in the vocab of words that appear at least 5 times.
    doc_str = ' '.join(closed_words)
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs) # each abstract, cleaned, stopwords removed, tokenised by whitespace. 

f = open('data/corpus/' + dataset + '.clean.txt', 'w') 
f.write(clean_corpus_str)
f.close()

dataset = args.dataset
min_len = 10000
aver_len = 0
max_len = 0

f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    temp = line.split()
    aver_len = aver_len + len(temp)
    if len(temp) < min_len:
        min_len = len(temp)
    if len(temp) > max_len:
        max_len = len(temp)
f.close()
aver_len = 1.0 * aver_len / len(lines)
print('min_len : ' + str(min_len))
print('max_len : ' + str(max_len))
print('average_len : ' + str(aver_len))
