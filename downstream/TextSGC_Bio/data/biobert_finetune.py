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

from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-large-cased-v1.1",num_labels = 23)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
torch.save(model,'corpus/')