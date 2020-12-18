import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
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
from tqdm import tqdm
import os

# parser = argparse.ArgumentParser(description='Build Document Graph')
# parser.add_argument('--dataset', type=str, default='ohsumed',
#                     choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'yelp', 'ag_news', 'covid_19_production','pubmed'],
#                     help='dataset name')
# parser.add_argument('--tokeniser', type=str, default='treebank',
#                     choices=['manual', 'scispacy','ref','nltk','treebank'],
#                     help='tokeniser to use')    
# parser.add_argument('--lemmatiser', type=str, default='bio',
#                     choices=['wordnet','bio','none'],
#                     help='lemmatisation algorithm')                    
# args = parser.parse_args()

# Read in the tokenised, lemmatised clean text. -> can't do because we don't have the labels for them!
# doc_content_list = []
# with open('data/corpus/' + dataset + '.' + tokeniser + '.' + lemmatiser + '.clean.txt', 'r') as f:
#     lines = f.readlines()
#     doc_content_list = [l.strip() for l in lines]

max_len = 506
#dataset = args.dataset

#tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1')
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1')

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

# Set text input embedding
full_input_ids = []
full_input_masks = []
full_segment_ids = []

SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

UNK_ID = tokenizer.encode("<unk>")[0]
CLS_ID = tokenizer.encode("<cls>")[0]
SEP_ID = tokenizer.encode("<sep>")[0]
MASK_ID = tokenizer.encode("<mask>")[0]
EOD_ID = tokenizer.encode("<eod>")[0]

for i, text in enumerate(all_texts):
    # Tokenize sentence to token id list 
    tokens_a = tokenizer.encode(text)
    
    # Trim the len of text
    if(len(tokens_a)>max_len-2):
        tokens_a = tokens_a[:max_len-2]
        
    tokens = []
    segment_ids = []
    
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
        
    # Add <sep> token 
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)
    
    
    # Add <cls> token
    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)
    
    input_ids = tokens
    
    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length at fornt
    if len(input_ids) < max_len:
        delta_len = max_len - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len
    
    full_input_ids.append(input_ids)
    full_input_masks.append(input_mask)
    full_segment_ids.append(segment_ids)
    
    if 3 > i:
        print("No.:%d"%(i))
        print("sentence: %s"%(text))
        print("input_ids:%s"%(input_ids))
        print("attention_masks:%s"%(input_mask))
        print("segment_ids:%s"%(segment_ids))
        print("\n")

# Splitting data into train and test
tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs = train_test_split(full_input_ids, all_labels,full_input_masks,full_segment_ids, random_state=4, test_size=0.2)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
tr_segs = torch.tensor(tr_segs)
val_segs = torch.tensor(val_segs)

batch_num = 8

# Set token embedding, attention embedding, segment embedding
train_data = TensorDataset(tr_inputs, tr_masks,tr_segs, tr_tags)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_num,drop_last=True)

valid_data = TensorDataset(val_inputs, val_masks,val_segs, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_num)

# Folder contains confg(json) and weight(bin) files
#model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-large-cased-v1.1',num_labels=23)
config = AutoConfig.from_pretrained('dmis-lab/biobert-large-cased-v1.1')
model = AutoModel.from_pretrained('dmis-lab/biobert-large-cased-v1.1', config=config, num_labels=23)
model.to(device)

# Add multi GPU support
if n_gpu >1:
    model = torch.nn.DataParallel(model)

# Set epoch and grad max num
epochs = 3
max_grad_norm = 1.0
# Cacluate train optimiazaion num
num_train_optimization_steps = int( math.ceil(len(tr_inputs) / batch_num) / 1) * epochs

## FINE TUNING
FULL_FINETUNING = True
if FULL_FINETUNING:
    # Fine tune model all layer parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    # Only fine tune classifier parameters
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

model.train()
print("***** Running training *****")
print("  Num examples = %d"%(len(tr_inputs)))
print("  Batch size = %d"%(batch_num))
print("  Num steps = %d"%(num_train_optimization_steps))
for _ in trange(epochs,desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_segs,b_labels = batch
        
        # forward pass
        outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
        loss, logits = outputs[:2]
        if n_gpu>1:
            # When multi gpu, average it
            loss = loss.mean()
        
        # backward pass
        loss.backward()
        
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        
        # update parameters
        optimizer.step()
        optimizer.zero_grad()
        
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

# Save model
save_address = 'corpus/biobert_finetuned'
# Make dir if doesn't exist
if not os.path.exists(save_address):
        os.makedirs(save_address)
model_to_save = model.module if hasattr(model, 'module') else model
# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(save_address, "pytorch_model.bin")
output_config_file = os.path.join(save_address, "config.json")
# Save model into file
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(save_address)

## Load the trained model
model = BertForSequenceClassification.from_pretrained(save_address,num_labels=23)

# Set model to GPU
model.to(device);
if n_gpu >1:
    model = torch.nn.DataParallel(model)

## Evaluate model
model.eval();

# Set acc funtion
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

y_true = []
y_predict = []
print("***** Running evaluation *****")
print("  Num examples ={}".format(len(val_inputs)))
print("  Batch size = {}".format(batch_num))
for step, batch in enumerate(valid_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_segs,b_labels = batch
    
    with torch.no_grad():
        outputs = model(input_ids =b_input_ids,token_type_ids=b_segs, input_mask = b_input_mask,labels=b_labels)
        tmp_eval_loss, logits = outputs[:2]
    
    # Get text classification prediction results
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    tmp_eval_accuracy = accuracy(logits, label_ids)
#     print(tmp_eval_accuracy)
#     print(np.argmax(logits, axis=1))
#     print(label_ids)
    
    # Save predict and real label reuslt for analyze
    for predict in np.argmax(logits, axis=1):
        y_predict.append(predict)
        
    for real_result in label_ids.tolist():
        y_true.append(real_result)

    
    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy
   
    nb_eval_steps += 1
    
    
eval_loss = eval_loss / nb_eval_steps
eval_accuracy = eval_accuracy / len(val_inputs)
loss = tr_loss/nb_tr_steps 
result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'loss': loss}
report = classification_report(y_pred=np.array(y_predict),y_true=np.array(y_true))

# Save the report into file
output_eval_file = os.path.join(save_address, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    print("***** Eval results *****")
    for key in sorted(result.keys()):
        print("  %s = %s"%(key, str(result[key])))
        writer.write("%s = %s\n" % (key, str(result[key])))
        
    print(report)
    writer.write("\n\n")  
    writer.write(report)

#### OLD VERSION ###

# tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-large-cased-v1.1')

# # Read Ohsumed dataset
# ohsumed_df = pd.read_csv('ohsumed0.txt', header=None, delimiter='\t')

# for i in ohsumed_df.index: 
#     ohsumed_df.loc[i,0] = re.sub('data/','', ohsumed_df.loc[i,0])
#     ohsumed_df.loc[i,2] = re.sub('C','', ohsumed_df.loc[i,2])
#     ohsumed_df.loc[i,2] = int(ohsumed_df.loc[i,2])-1

# train_texts = []
# train_labels = []
# test_texts =[]
# test_labels = []

# for i in ohsumed_df.index:
#     f = open(ohsumed_df.loc[i,0],'r')
#     text = f.read()
#     if ohsumed_df.loc[i,1] == 'test':
#         test_texts.append(text)
#         test_labels.append(ohsumed_df.loc[i,2])
#     else: 
#         train_texts.append(text)
#         train_labels.append(ohsumed_df.loc[i,2])

# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size = 0.1, stratify=train_labels)
# train_encodings = tokenizer(train_texts, truncation = True, padding = 'max_length', max_length=512)
# val_encodings = tokenizer(val_texts, truncation = True, padding = 'max_length', max_length=512)
# test_encodings = tokenizer(test_texts, truncation = True, padding = 'max_length', max_length=512)

# class OhsumedDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = OhsumedDataset(train_encodings, train_labels)
# val_dataset = OhsumedDataset(val_encodings, val_labels)
# test_dataset = OhsumedDataset(test_encodings, test_labels)

# from torch.utils.data import DataLoader
# from transformers import BertForSequenceClassification, AdamW

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-large-cased-v1.1",num_labels = 23)
# model.to(device)
# model.train()

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(3):
#     print('epoch',epoch)
#     progress_bar = tqdm(train_loader)
#     progress_bar.set_postfix_str("finetuning in batches")
#     for batch in progress_bar:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()

# model.eval()
# torch.save(model,'corpus/biobert_finetuned.pth')