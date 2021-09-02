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
import tensorflow as tf

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
    
    #print('pred:',pred)
    #print('true_label:',true_label)
    for j in range (0,10):
        #print('j:',j)
        dual_list.append([pred[j],true_label[j]])
    #sorted_list = sorted(dual_list, key=lambda tup: tup[0],reverse=1)
    sorted_list = sorted(dual_list, key=lambda tup: tup[0], reverse=1)
    #print(sorted_list)
    ranking=[]
    TP=0
    prec=0
    for j in range(0,10):
        ranking.append(sorted_list[j][1])
        if ranking[j]!=0:
            TP+=1
            prec+=TP/(j+1)
    if TP!=0:
        avg_p=prec/TP
    print('avg_prec:',avg_p)
    print('\n')
    return avg_p

############from BERT-test-QQP.py#############
import tensorflow as tf
import torch
"""
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
"""    
device = torch.device("cpu")
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
#model.cuda()
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
seed_val = 8

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
input_path='./SemEval2017-task3-English-test-labeled.xml'
input=open(input_path,'r',encoding="utf-8")

tree = ET.parse(input)
root = tree.getroot()
org_Q=root.findall('OrgQuestion')
#org_Q=root.findall('orgquestion')
AP=[]
labels=[]
for k in range(0,int(len(org_Q)/10)):
    #for i in range(k*10,(k+1)*10):   # for each of 10 original questions
    pred=[]
    true_label=[]
    for i in range(0,10):
        org_body=org_Q[(k*10)+i].find('OrgQBody')
        org_sub=org_Q[(k*10)+i].find('OrgQSubject')
        thread=org_Q[(k*10)+i].find('Thread')
        rel_Q = thread.find('RelQuestion')
        relevancy=rel_Q.attrib['RELQ_RELEVANCE2ORGQ']
        topic=rel_Q.attrib['RELQ_CATEGORY']
        rel_body=rel_Q.find('RelQBody')
        Ans = thread.findall('RelComment')
        answer=''
        for l  in range(len(Ans)):
            A_relevancy=Ans[l].attrib['RELC_RELEVANCE2RELQ']
            if A_relevancy=='Good':
                A_tag=Ans[l].find('RelCText')
                answer=A_tag.text
                break
        print('answer:',answer)
        if l==10:
            print('there is a relQ without GOOD answer:',thread.attrib['THREAD_SEQUENCE'])
        Q1=org_body.text+' '+org_sub.text
        Q2=rel_body.text
        #print('Q1:',Q1)
        #print('Q2:',Q2)

        """"
        org_body = org_Q[(k * 10) + i].find('pe_orgqbody')
        #org_body = org_Q[(k * 10) + i].find('orgqbody')
        thread = org_Q[(k * 10) + i].find('thread')
        rel_Q = thread.find('relquestion')
        relevancy = rel_Q.attrib['relq_relevance2orgq']
        rel_body = rel_Q.find('relqbody')
        """
        #print("org text:",org_body.text)
        #print("rel text:", rel_body.text)
        #print('label:',relevancy)
        label=1
        if relevancy=='Irrelevant':
        #if relevancy == 'irrelevant':
            label=0
        labels.append(label)
    #print('Q1:', Q1)
    #print('\n')
    #print('Q2:',Q2)

    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
        Q1,
        answer,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.
batch_size = 10

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

############################# Load a trained model and vocabulary that you have fine-tuned ##########################
model.eval()
output_dir = './model_save_QA_24K/'
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
distance=[]
for x in range(10):
    distance.append(0)
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
    print('output:',outputs[0])

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    for i in range(0,10):
        distance[i]=logits[i][1]-logits[i][0]
        print('output_distance:',distance[i])

    print('label:',label_ids)
    # Store predictions and true labels
    predictions.append(logits)
   # print('predictions:',predictions)
    true_labels.append(label_ids)
    
    #print('predictions:',predictions)

print('    DONE.')
################################
#print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))
################################
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
    #pred_labels_i=predictions[i]
    #print('predictions[i]',predictions[i])
    pred_labels.append(pred_labels_i)
    # Calculate and store the coef for this batch.
    #matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
    #matthews_set.append(matthews)
    print('true_labels[i]:',true_labels[i])
    print('pred_labels[i]:', pred_labels_i)
    
    AP.append(Avg_prec(pred_labels_i, true_labels[i]))
    print('\n')
    accuracy+=accuracy_score(true_labels[i], pred_labels_i,normalize=False)
print('accuracy:',(accuracy/len(input_ids)))
map_measure=MAP(AP)
print('MAP:',map_measure)
"""
#######################################
# Create a barplot showing the MCC score for each batch of test samples.
import matplotlib.pyplot as plt

import seaborn as sns
ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

plt.title('MCC Score per Batch')
plt.ylabel('MCC Score (-1 to +1)')
plt.xlabel('Batch #')

plt.show()
#####################  Now we'll combine the results for all of the batches and calculate our final MCC score. #################
# Combine the results across all batches.
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('Total MCC: %.3f' % mcc)
############end of from BERT-test-QQP.py##########################
"""
