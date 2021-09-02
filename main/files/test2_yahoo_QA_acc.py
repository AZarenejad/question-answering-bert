import xml.etree.ElementTree as ET
import re
import os
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import tensorflow as tf
out_results=open('../../hdd/ghasemi/results.txt','w',encoding='utf-8')
out1=open('../../hdd/ghasemi/out_QA_acc','w',encoding='utf-8')
def MAP(AP):
    sum=0
    for t in range(0,len(AP)):
        #print("AP on question",(t+388),":",AP)
        sum+=AP[t]
    map_measure=sum/len(AP)
    return map_measure


def Avg_prec(pred, true_label):
    avg_p=0.0
    j=0

    dual_list=[]
    print('pred:',pred)
    print('true_label:',true_label)
    for j in range (0,len(pred)):
        #print('j:',j)
        dual_list.append([pred[j],true_label[j]])
    sorted_list = sorted(dual_list, key=lambda tup: tup[0],reverse=1)
    #print(sorted_list)
    ranking=[]
    TP=0
    prec=0
    for j in range(0,len(pred)):
        ranking.append(sorted_list[j][1])
        if ranking[j]!=0:
            TP+=1
            prec+=TP/(j+1)
    if TP!=0:
        avg_p=prec/TP
    return avg_p

############from BERT-test-QQP.py#############
import tensorflow as tf
import torch

# If there's a GPU available...

if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#device = torch.device("cpu")
############################################################
import pandas as pd
#############################################################

###############################
# Get the lists of sentences and their labels.
##############################
from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#################################
# Print the original sentence.
#################################
######################################
# Tokenize all of the sentences and map the tokens to thier word IDs.
#################################
from torch.utils.data import TensorDataset, random_split
#########################
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
###########################################
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
print('line 153')
# Tell pytorch to run this model on the GPU.
model.cuda()
######################sth was here##############
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#############################
# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
################################
from transformers import get_linear_schedule_with_warmup
############################
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
###############################
import time
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
#################################
import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
############################################

#########################################
import matplotlib.pyplot as plt
########################### performance on test set####################

import pandas as pd

# Load the dataset into a pandas dataframe.
#df = pd.read_csv("../../hdd/ghasemi/QQP/test.tsv", delimiter='\t', header=None,names=['sentence_source', 'label', 'label_notes', 'sentence'])

#df = pd.read_csv("../../hdd/ghasemi/QQP/test-small.tsv", delimiter='\t', header=None, names=['id', 'qid1', 'qid2', 'question1','question2','is_duplicate'])

#Q1s = df.question1.values
#Q2s = df.question2.values
#labels = df.is_duplicate.values

#for i in range(len(labels)):
 #   labels[i]=int(labels[i])


# Report the number of sentences.
#print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
#sentences = df.sentence.values
#labels = df.label.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

#for i in range(len(Q1s)):
    #Q1=Q1s[i]
    #Q2 = Q2s[i]
#file=open('C:\shima\PHD\search\QA\dataset\Quora\Quora-small-tab.txt','w', encoding="utf-8")
#input_path='../../hdd/ghasemi/yahoo-labeled/yahoo.data'
input_path_Q='../../hdd/ghasemi/yahoo-labeled/yahoo.data'
input_Q=open(input_path_Q,'r',encoding="utf-8")
lines_Q=input_Q.readlines()

input_path_A='../../hdd/ghasemi/yahoo-labeled/yahoo.url_answer'
input_A=open(input_path_A,'r',encoding="utf-8")
lines_A=input_A.readlines()
################# prepare test data #####################
AP=[]
labels=[]
i=0
for line in lines_Q:
    partitions_Q = line.split('\t')
    Q1=partitions_Q[0]

    partitions_A = lines_A[i].split('\t')
    A2 = partitions_A[1]
    #Q2=partitions[1]
    labels.append(int(partitions_Q[2]))
    encoded_dict = tokenizer.encode_plus(
        Q1,
        A2, # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=20,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
    i+=1
# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.
batch_size = 1

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

############################# Load a trained model and vocabulary that you have fine-tuned ##########################
model.eval()
output_dir = './model_save_QA_5M/'
model = model.from_pretrained(output_dir)
tokenizer = tokenizer.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)



########################### evaluate on test set##################
# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions, true_labels = [], []

# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]
    #print('output:',outputs[0])
    #for i in range(0,len(batch)):
        #print('output_distance:',(logits[i][0]-logits[i][1]))
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    #print('label:',label_ids)
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

print('    DONE.')
#accuracy=flat_accuracy(predictions,true_labels)
#print('acc:',accuracy)
################################
#print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))
################################ evaluation of metrics #############
for i in range(0,50,len(predictions)):
    print('predictions:',predictions[i])
    print('true_labels:', true_labels[i])

AP=[]
    #model = Net()
    #model.load_state_dict(torch.load('mytrainedmodel-triples-600.pt'))
k=0
flag=0
accuracy=0
Q_partitions=lines_Q[k].split('\t')
Q1_prev=Q_partitions[0]
Q2_prev=Q_partitions[1]
label=int(Q_partitions[2])
while(k<len(lines_Q)):
    pred_tmp=[]
    true_label_tmp = []
    while(1):
        tmp = np.argmax(predictions[k], axis=1).flatten()
        #tmp=predictions[k][0][1]-predictions[k][0][0]
        pred_tmp.append(int(tmp))
        true_label_tmp.append(true_labels[k][0])
        if (k == len(lines_Q)):
            break
        k += 1
        if k==len(lines_Q):
            break
        Q_partitions = lines_Q[k].split('\t')

        Q1_new = Q_partitions[0]
        Q2_new = Q_partitions[1]
        #label = int(Q_partitions[2])
        if(Q1_new==Q1_prev):
            Q1_prev=Q1_new
            Q2_prev=Q2_new
        else:
            Q1_prev = Q1_new
            Q2_prev = Q2_new
            break
    AP.append(Avg_prec(pred_tmp, true_label_tmp))
    accuracy+=accuracy_score(true_label_tmp,pred_tmp,normalize=False)
    for l in range(0,len(pred_tmp)):
        str1=str(predictions[l])+'\t'+str(true_label_tmp[l])+'\n'
        out1.write(str1)
print('acc:',(accuracy/len(input_ids)))
map_measure=MAP(AP)
print("map_measure:",map_measure,'\n')
out_results.write(str(map_measure))
out_results.close()
"""
AP=[]
    #model = Net()
    #model.load_state_dict(torch.load('mytrainedmodel-triples-600.pt'))
    k=0
    flag=0
    Q_partitions=lines[k].split('\t')
    Q1_prev=Q_partitions[0]
    Q2_prev=Q_partitions[1]
    label=int(Q_partitions[2])
    k=1
    while(k<len(lines)):
        pred=[]
        true_label = []
        while(1):
            score = similarity(Q1_prev, Q2_prev, dic,beta)
            # score = similarity(Q1,Q2, dic)
            pred.append(score)
            true_label.append(label)
            if (k == len(lines)):
                break
            Q_partitions = lines[k].split('\t')
            k+=1
            Q1_new = Q_partitions[0]
            Q2_new = Q_partitions[1]
            label = int(Q_partitions[2])
            if(Q1_new==Q1_prev):
                Q1_prev=Q1_new
                Q2_prev=Q2_new
            else:
                Q1_prev = Q1_new
                Q2_prev = Q2_new
                break
        AP.append(Avg_prec(pred, true_label))
    map_measure=MAP(AP)
    print("beta:",beta)
    print("map_measure:",map_measure,'\n')



from sklearn.metrics import matthews_corrcoef,accuracy_score
accuracy=0
matthews_set = []
pred_labels=[]
# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
    # The predictions for this batch are a 2-column ndarray (one column for "0"
    # and one column for "1"). Pick the label with the highest value and turn this
    # in to a list of 0s and 1s.
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    pred_labels.append(pred_labels_i)
    # Calculate and store the coef for this batch.
    matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
    matthews_set.append(matthews)
    print('true_labels[i]:',true_labels[i])
    print('pred_labels[i]:', pred_labels_i)
    print('\n')
    AP.append(Avg_prec(pred_labels_i, true_labels[i]))
    accuracy+=accuracy_score(true_labels[i], pred_labels_i,normalize=False)
print('accuracy:',(accuracy/len(input_ids)))
map_measure=MAP(AP)
print('MAP:',map_measure)
"""

