import h5py 
import csv

embedding_path = 'data/corpus/all_corpora_ft-biobert-large_embeddings.h5'
word_vector_map = h5py.File(embedding_path, 'r')

corp_vocab = []
vectors = []

for word in word_vector_map:
    corp_vocab.append(str(word))
    vectors = np.array(word_vector_map[word]['embedding'][:])

with open('data/ftbiobert_all_vocab.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(corp_vocab)

with open('data/ftbiobert_all_vectors.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t',lineterminator='\n')
    for v in vectors:
        tsv_output.writerow(v)