import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    HfArgumentParser,
    set_seed,
    BertTokenizer
)
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import os
import math
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F
import tensorflow as tf

# Set device to GPU if available
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Read Ohsumed dataset
ohsumed_df = pd.read_csv('ohsumed0.txt', header=None, delimiter='\t')

for i in ohsumed_df.index: 
    ohsumed_df.loc[i,0] = re.sub('data/','', ohsumed_df.loc[i,0])
    ohsumed_df.loc[i,2] = re.sub('C','', ohsumed_df.loc[i,2])
    ohsumed_df.loc[i,2] = int(ohsumed_df.loc[i,2])-1

all_texts = []
all_labels = []

for i in ohsumed_df.index:
    f = open(ohsumed_df.loc[i,0],'r')
    text = f.read()
    all_texts.append(text)
    all_labels.append(ohsumed_df.loc[i,2])

tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1', do_lower_case=True)

# Print the original sentence.
print(' Original: ', all_texts[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(all_texts[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(all_texts[0])))

max_len = 0

# For every sentence...
for sent in all_texts:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)