import pandas as pd
doc_content_list = []
## OHSUMED
with open('data/corpus/ohsumed.treebank.bio.clean.txt','r') as f:
    lines = f.readlines()
    for i,text in enumerate(lines):
        doc_content_list.append(text.strip().split())
with open('data/corpus/covid_19_production.treebank.bio.clean.txt','r') as f:
    lines = f.readlines()
    for i,text in enumerate(lines):
        doc_content_list.append(text.strip().split())
with open('data/corpus/pubmed.treebank.bio.clean.txt','r') as f:
    lines = f.readlines()
    for i,text in enumerate(lines):
        doc_content_list.append(text.strip().split())
from gensim.models import Word2Vec
import multiprocessing
cores = multiprocessing.cpu_count()
print('number of cores:', cores)
model = Word2Vec(min_count=5,
                     window=10,
                     size=712,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     workers=cores-1)
from time import time
t = time()

model.build_vocab(doc_content_list, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
t = time()

model.train(doc_content_list, total_examples=model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# Total number of documents used to train Word2Vec model
print('Total number of training docs: ', model.corpus_count)

# Save the trained model
model.save('data/trained_w2v_model.bin')

import csv        

words = list(model.wv.vocab)  

with open('data/word2vec_vocab.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(words)

vectors = model[model.wv.vocab]
with open('data/word2vec_vectors.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(vectors)

print('most similar words to INFECTION')
model.wv.most_similar(positive=["infection"])