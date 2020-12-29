import h5py 
import csv
import numpy as np

embedding_path = 'data/corpus/all_corpora_biobert-large_embeddings.h5'
word_vector_map = h5py.File(embedding_path, 'r')

### Closest words to infect

# closest_words = ['infect', 'infected', 'infection', 'infectious', 'virusinfect','hivinfect','covid19infect','coinfect','infectionrelated','influenza','infectivity']
# corp_vocab = []
# vectors = []

# for word in closest_words:
#     corp_vocab.append(str(word))
#     vectors.append(np.array(word_vector_map[word]['embedding'][:]))

# with open('data/biobert_infect_vocab.tsv', 'w', newline='') as f_output:
#     tsv_output = csv.writer(f_output, delimiter='\n')
#     tsv_output.writerow(corp_vocab)

# with open('data/biobert_infect_vectors.tsv', 'w', newline='') as f_output:
#     tsv_output = csv.writer(f_output, delimiter='\t',lineterminator='\n')
#     for v in vectors:
#         tsv_output.writerow(v)

## all embeddings

corp_vocab = []
vectors = []

for word in word_vector_map:
    corp_vocab.append(str(word))
    vectors.append(np.array(word_vector_map[word]['embedding'][:]))

with open('data/biobert_all_vocab.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(corp_vocab)

with open('data/biobert_all_vectors.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t',lineterminator='\n')
    for v in vectors:
        tsv_output.writerow(v)