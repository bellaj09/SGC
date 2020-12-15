from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer 
from utils import clean_str, loadWord2Vec, clean_str_manual 
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
import argparse
import random
from collections import Counter
import time
from tqdm import tqdm
import spacy
import re
import numpy as np
import pandas as pd
import subprocess
nltk.download('averaged_perceptron_tagger')

parser = argparse.ArgumentParser(description='Build Document Graph')
parser.add_argument('--dataset', type=str, default='20ng',
                    choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','covid_19_production','pubmed'],
                    help='dataset name')
parser.add_argument('--tokeniser', type=str, default='treebank', # Chosen tokeniser: Treebank + Manual Rules
                    choices=['manual', 'scispacy','ref','nltk','treebank'],
                    help='tokeniser to use')
parser.add_argument('--stopwords', type=str, default='stanford',
                    choices=['nltk','stanford', 'pubmed','top50','top100','none','pubmednltk'],
                    help='stopwords list')
parser.add_argument('--lemmatiser', type=str, default='none',
                    choices=['wordnet','bio','none'],
                    help='lemmatisation algorithm')                                                      
args = parser.parse_args()

dataset = args.dataset
tokeniser = args.tokeniser
lemmatiser = args.lemmatiser

train_val_ids = []
test_ids = []

# Define stopwords
# nltk.download() 
if args.stopwords == "nltk":
    stop_words = set(stopwords.words('english'))
elif args.stopwords == 'stanford':
    stop_words = {'disease', 'diseases', 'disorder', 'symptom', 'symptoms', 'drug', 'drugs', 'problems', 'problem','prob', 'probs', 'med', 'meds',
    'pill', 'pills', 'medicine', 'medicines', 'medication', 'medications', 'treatment', 'treatments', 'caps', 'capsules', 'capsule',
    'tablet', 'tablets', 'tabs', 'doctor', 'dr', 'dr.', 'doc', 'physician', 'physicians', 'test', 'tests', 'testing', 'specialist', 'specialists',
   'side-effect', 'side-effects', 'pharmaceutical', 'pharmaceuticals', 'pharma', 'diagnosis', 'diagnose', 'diagnosed', 'exam',
    'challenge', 'device', 'condition', 'conditions', 'suffer', 'suffering' ,'suffered', 'feel', 'feeling', 'prescription', 'prescribe',
    'prescribed', 'over-the-counter', 'otc', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before',
'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he',
"he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into',
'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
'other', 'ought', 'our', 'ours' , 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't",
'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they',
"they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd",
"we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's",
'whom', 'why', "why's", "with", "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
'yourselves', "n't", "'re", "'ve", "'d", "'s", "'ll", "'m"}
elif args.stopwords == 'pubmed':
    stop_words = {'a', 'about', 'again', 'all', 'almost', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'between', 'both', 'but', 'bycan', 'could', 'did', 'do', 'does', 'done', 'due', 'during', 'each', 'either', 'enough', 'especially', 'etc', 'for', 'found', 'from', 'further', 'had', 'has', 'have', 'having', 'here', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'kg', 'km', 'made', 'mainly', 'make', 'may', 'mg', 'might', 'ml', 'mm', 'most', 'mostly', 'must', 'nearly', 'neither', 'no', 'nor', 'obtained', 'of', 'often', 'on', 'our', 'overall', 'perhaps', 'pmid', 'quite', 'rather', 'really', 'regarding', 'seem', 'seen', 'several', 'should', 'show', 'showed', 'shown', 'shows', 'significantly', 'since', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'therefore', 'these', 'they', 'this', 'those', 'through', 'thus', 'to', 'upon', 'various', 'very', 'was', 'we', 'were', 'what', 'when', 'which', 'while', 'with', 'within', 'without', 'would'}
elif args.stopwords == 'none':
    stop_words = {}
elif args.stopwords == 'pubmednltk':
    nltk_stops = set(stopwords.words('english'))
    pubmed_stops = {'a', 'about', 'again', 'all', 'almost', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'between', 'both', 'but', 'bycan', 'could', 'did', 'do', 'does', 'done', 'due', 'during', 'each', 'either', 'enough', 'especially', 'etc', 'for', 'found', 'from', 'further', 'had', 'has', 'have', 'having', 'here', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'kg', 'km', 'made', 'mainly', 'make', 'may', 'mg', 'might', 'ml', 'mm', 'most', 'mostly', 'must', 'nearly', 'neither', 'no', 'nor', 'obtained', 'of', 'often', 'on', 'our', 'overall', 'perhaps', 'pmid', 'quite', 'rather', 'really', 'regarding', 'seem', 'seen', 'several', 'should', 'show', 'showed', 'shown', 'shows', 'significantly', 'since', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'therefore', 'these', 'they', 'this', 'those', 'through', 'thus', 'to', 'upon', 'various', 'very', 'was', 'we', 'were', 'what', 'when', 'which', 'while', 'with', 'within', 'without', 'would'}
    stop_words = nltk_stops | pubmed_stops


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

if args.stopwords == 'top50':
    word_freq = Counter()
    # total = 0
    for i in train_ids+test_ids+val_ids:
        all_words = doc_content_list[i].lower().split()
        word_freq.update(all_words)     
    vocab, count = zip(*word_freq.most_common()) 
    stop_words = set(vocab[:49]) # take the top 50 words

if args.stopwords == 'top100':
    word_freq = Counter()
    # total = 0
    for i in train_ids+test_ids+val_ids:
        all_words = doc_content_list[i].lower().split()
        word_freq.update(all_words)     
    vocab, count = zip(*word_freq.most_common()) 
    stop_words = set(vocab[:99]) # take the top 50 words

print('Stop Words: ',stop_words)

def get_clean_words(docs):
    clean_words = []
    
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
            #doc = re.sub(r'[?|$|.|!|,]',r'',doc) 
            doc = re.sub(r'[^a-zA-Z0-9 \/-]'," ",doc) # replace with space non-alphanumeric characters except for dashes and slashes
            doc = re.sub(r"\s{2,}", " ", doc) # remove duplicate whitespaces
            doc_temp = nlp(doc)
            temp = [token.text for token in doc_temp]
            temp = list(filter(lambda x : x not in stop_words, temp))
            
        elif args.tokeniser == "ref":
            temp = clean_str(doc).split()
            temp = list(filter(lambda x : x not in stop_words, temp))

        elif args.tokeniser == 'nltk': # try not to americanize yet
            # americanize options
            doc = doc.strip().lower()
            temp = nltk.word_tokenize(doc)
            temp = list(filter(lambda x : x not in stop_words, temp))

        elif args.tokeniser == 'treebank':
            # CHOSEN: Treebank + Manual Rules
            doc = doc.strip().lower()

            doc = re.sub(r'[^a-zA-Z0-9  -]',r'',doc) # all special characters can just disappear, except for hyphen
            temp = TreebankWordTokenizer().tokenize(doc)
            temp = list(filter(lambda x : x not in stop_words, temp))
        
        if args.lemmatiser == 'wordnet':
            lemmatizer = WordNetLemmatizer() 
            def get_wordnet_pos(treebank_tag):  # Convert Treebank POS tags to WordNet
                if treebank_tag.startswith('J'):
                    return wn.ADJ
                elif treebank_tag.startswith('V'):
                    return wn.VERB
                elif treebank_tag.startswith('N'):
                    return wn.NOUN
                elif treebank_tag.startswith('R'):
                    return wn.ADV
                else:
                    return wn.NOUN
            temp_pos = nltk.pos_tag(temp)
            temp = []
            for i in range(len(temp_pos)):
                current_word = temp_pos[i][0]
                current_tag = get_wordnet_pos(temp_pos[i][1])
                temp.append(lemmatizer.lemmatize(current_word, current_tag))
       
        elif args.lemmatiser == 'bio':
            tagged_df = pd.DataFrame(nltk.pos_tag(temp))
            tagged_df.to_csv('tagged_string.txt',sep = '\t',header = False, index = False)
            subprocess.run(["java -Xmx1G -jar biolemmatizer-core-1.2-jar-with-dependencies.jar -l -i 'tagged_string.txt' -o 'biolemmatizer_output.txt'"], shell=True)
            df = pd.read_csv('biolemmatizer_output.txt', header=None, sep='\t')
            temp = df[2].to_list()
        
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
    cutoff = count.index(5) # sets cutoff to be the words that occur at least 5 times!

vocab = set(vocab[:cutoff]) 

clean_docs = []
for words in clean_words: # Loops through every single abstract's cleaned words
    closed_words = [w for w in words if w in vocab ] # an array of the words in each abstract, if they are in the vocab of words that appear at least 5 times.
    doc_str = ' '.join(closed_words)
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs) # each abstract, cleaned, stopwords removed, tokenised by whitespace. 

f = open('data/corpus/' + dataset + '.' + tokeniser + '.' + lemmatiser + '.clean.txt', 'w') 
f.write(clean_corpus_str)
f.close()

dataset = args.dataset
min_len = 10000
aver_len = 0
max_len = 0

f = open('data/corpus/' + dataset + '.' + tokeniser + '.' + lemmatiser + '.clean.txt', 'r')
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
