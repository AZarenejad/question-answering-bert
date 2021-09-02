import nltk
#nltk.download('punkt')
import os
import torch
import pandas as pd
########### Q2 plays the role of Answer here!!
input_path='../../hdd/ghasemi/yahoo-BERT-label1.txt'
df = pd.read_csv(input_path, delimiter='\t', header=None, names=['id', 'qid1', 'qid2', 'question1','question2','is_duplicate'])
Q1s = df.question1.values
Q2s = df.question2.values
labels = df.is_duplicate.values

i=0
sum_Q1=0
sum_Q2=0
sum_A2=0
label_0=0
label_1=0
for i in range(len(Q1s)):
    Q1=Q1s[i]
    Q2 = Q2s[i]
    sum_Q1 +=len(nltk.word_tokenize(Q1))
    sum_Q2 += len(nltk.word_tokenize(Q2))
    #sum_A2 += len(nltk.word_tokenize(A2))
    label=labels[i]
    if label!=0 and label!=1:
        print(label)
        label = 0
        print('bug in line  :',i+1)

    if label == 0:
        label_0 += 1
    if label == 1:
        label_1 += 1
    if i%100000==0:
        print(i)

avg_Q1=sum_Q1/len(Q1s)
avg_Q2=sum_Q2/len(Q1s)
#avg_A2=sum_A2/len(len(Q1s))
print('avg_Q1=',avg_Q1)
print('avg_Q2=',avg_Q2)
#print('avg_A2=',avg_A2)
print('label_0:',label_0,'   ',label_0/len(labels))
print('label_1:',label_1,'   ',label_1/len(labels))
