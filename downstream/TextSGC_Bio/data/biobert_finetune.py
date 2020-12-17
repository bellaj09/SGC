from transformers import BertTokenizerFast
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
import torch

tokenizer = BertTokenizerFast.from_pretrained('dmis-lab/biobert-large-cased-v1.1')
# Read Ohsumed dataset
ohsumed_df = pd.read_csv('ohsumed0.txt', header=None, delimiter='\t')

for i in ohsumed_df.index: 
    ohsumed_df.loc[i,0] = re.sub('data/','', ohsumed_df.loc[i,0])
    ohsumed_df.loc[i,2] = re.sub('C','', ohsumed_df.loc[i,2])
    ohsumed_df.loc[i,2] = int(ohsumed_df.loc[i,2])-1

train_texts = []
train_labels = []
test_texts =[]
test_labels = []

for i in ohsumed_df.index:
    f = open(ohsumed_df.loc[i,0],'r')
    text = f.read()
    if ohsumed_df.loc[i,1] == 'test':
        test_texts.append(text)
        test_labels.append(ohsumed_df.loc[i,2])
    else: 
        train_texts.append(text)
        train_labels.append(ohsumed_df.loc[i,2])

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size = 0.1, stratify=train_labels)
train_encodings = tokenizer(train_texts, truncation = True, padding = 'max_length', max_length=512)
val_encodings = tokenizer(val_texts, truncation = True, padding = 'max_length', max_length=512)
test_encodings = tokenizer(test_texts, truncation = True, padding = 'max_length', max_length=512)

class OhsumedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = OhsumedDataset(train_encodings, train_labels)
val_dataset = OhsumedDataset(val_encodings, val_labels)
test_dataset = OhsumedDataset(test_encodings, test_labels)

from transformers import BertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-large-cased-v1.1",num_labels = 23)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()