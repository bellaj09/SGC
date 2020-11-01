from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer 
from utils import clean_str, loadWord2Vec
import sys
import argparse
import random
from collections import Counter
import pandas as import pd
# For BioLemmatizer
import subprocess
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

# nltk.download()
stop_words = set(stopwords.words('english'))
print(stop_words)

parser = argparse.ArgumentParser(description='Build Document Graph')
parser.add_argument('--dataset', type=str, default='20ng',
                    choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','covid_19_production'],
                    help='dataset name')
args = parser.parse_args()

dataset = args.dataset

train_val_ids = []
test_ids = []

with open('data/' + dataset + '.txt', 'r') as f:
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
f = open('data/corpus/' + dataset + '.txt', 'rb')
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
    lemmatizer = WordNetLemmatizer() 
    for doc in docs:
        if args.dataset != "mr":
            temp = clean_str(doc).split()
            temp = list(filter(lambda x : x not in stop_words, temp)) # stopword removal and unique tokens only
        else:
            temp = clean_str(doc).split() # A list of all words extracted by splitting whitespace 
        tagged_df = pd.DataFrame(pos_tag(temp))
        tagged_df.to_csv('tagged_string.txt',sep = '\t',header = False, index = False)
        subprocess.run(["java -Xmx1G -jar biolemmatizer-core-1.2-jar-with-dependencies.jar -l -i 'tagged_string.txt' -o 'biolemmatizer_output.txt'"], shell=True)
        df = pd.read_csv('biolemmatizer_output.txt', header=None, sep='\t')
        temp = df[2]
        temp = ' '.join(temp)        
        clean_words.append(temp)
    return clean_words
clean_words = get_clean_words(doc_content_list) 


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
