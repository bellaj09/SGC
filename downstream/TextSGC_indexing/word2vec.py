# import pandas as pd
# doc_content_list = []
# ## OHSUMED
# with open('data/corpus/ohsumed.treebank.bio.clean.txt','r') as f:
#     lines = f.readlines()
#     for i,text in enumerate(lines):
#         doc_content_list.append(text.strip().split())
# with open('data/corpus/covid_19_production.treebank.bio.clean.txt','r') as f:
#     lines = f.readlines()
#     for i,text in enumerate(lines):
#         doc_content_list.append(text.strip().split())
# with open('data/corpus/pubmed.treebank.bio.clean.txt','r') as f:
#     lines = f.readlines()
#     for i,text in enumerate(lines):
#         doc_content_list.append(text.strip().split())
# from gensim.models import Word2Vec
# import multiprocessing
# cores = multiprocessing.cpu_count()
# print('number of cores:', cores)
# model = Word2Vec(min_count=5,
#                      window=10,
#                      size=712,
#                      sample=6e-5, 
#                      alpha=0.03, 
#                      min_alpha=0.0007, 
#                      workers=cores-1)
# from time import time
# t = time()

# model.build_vocab(doc_content_list, progress_per=10000)

# print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
# t = time()

# model.train(doc_content_list, total_examples=model.corpus_count, epochs=30, report_delay=1)

# print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# # Total number of documents used to train Word2Vec model
# print('Total number of training docs: ', model.corpus_count)

# # Save the trained model
# model.save('data/trained_w2v_model.bin')

# import csv        

# words = list(model.wv.vocab)  

# with open('data/word2vec_vocab.tsv', 'w', newline='') as f_output:
#     tsv_output = csv.writer(f_output, delimiter='\n')
#     tsv_output.writerow(words)

# vectors = model[model.wv.vocab]
# with open('data/word2vec_vectors.tsv', 'w', newline='') as f_output:
#     tsv_output = csv.writer(f_output, delimiter='\n')
#     tsv_output.writerow(vectors)

# print('most similar words to INFECTION')
# print(model.wv.most_similar(positive=["infection"]))

# from gensim.models import KeyedVectors

# model_2 = Word2Vec(size=300, min_count=1)
# model_2.build_vocab(doc_content_list)
# total_examples = model_2.corpus_count
# print(total_examples)

# ptmodel = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
# print('Updating with Google News vocab')
# model_2.build_vocab([list(ptmodel.vocab.keys())], update=True)
# print('Finished updating. Intersecting vectors....')
# model_2.intersect_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)
# print('Training new model')
# model_2.train(doc_content_list, total_examples=total_examples, epochs=model_2.iter)

# model_2.save('data/finetuned_w2v_model.bin')
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

finetuned_model = Word2Vec.load('data/finetuned_w2v_model.bin')
print('most similar words to INFECTION')
print(finetuned_model.wv.most_similar(positive=["infection"]))

import csv

# get two tsv's of the corpora's vocab and their vectors 
corp_vocab = []

dataset = ['ohsumed','covid_19_production','pubmed']
tokeniser = 'treebank'
lemmatiser = 'bio'

for d in dataset:
    with open('data/corpus/' + d + '.' + tokeniser  + '.' + lemmatiser + '_vocab.txt', 'r') as f:
        lines = f.readlines()
        for l in line:
            corp_vocab.append(str(l))

corp_vocab = list(corp_vocab)

with open('data/ftword2vec_corp_vocab.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(corp_vocab)

vectors = finetuned_model[corp_vocab]

with open('data/ftword2vec_corp_vectors.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    for v in vectors:
        tsv_output.writerow(v)

