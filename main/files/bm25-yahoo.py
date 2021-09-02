from rank_bm25 import BM25Okapi

"""corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]"""
corpus=[]
tmp=0
# this file is for creating a version of yahoo QA alignment which have label 1 and 0
import os
import torch
import random
"""
input_path_Q='./yahoo-BERT-tmp.txt'
input_path_A='./yahoo-A.txt'
out=open('./yahoo-BERT-3.txt','w',encoding="utf-8")
"""
input_path_Q='../../hdd/ghasemi/yahoo-BERT-label1.txt'
input_path_A='../../hdd/ghasemi/yahoo-A.txt'
out=open('../../hdd/ghasemi/yahoo-BERT-3-v2.txt','w',encoding="utf-8")

input_Q=open(input_path_Q,'r',encoding="utf-8")
input_A=open(input_path_A,'r',encoding="utf-8")

lines_Q=input_Q.readlines()
lines_A=input_A.readlines()
i=0
num=0
for line in lines_A:
    if tmp==100:
        break
    corpus.append(line)
    tmp+=1


tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

for line_Q in lines_Q:
    
    newline = line_Q
    tmp=line_Q.split('\t')
    Q=tmp[3]
    out.write(newline)

    j=random.randint(1,len(lines_A)-1)
    A=lines_A[j]
    A = A.replace('\n', '')
    label='0'
    newline=str(i)+'\t'+str(i)+'\t'+str(i)+'\t'+Q+'\t'+A+'\t'+label+'\n'
    #newline=A+'\n'
    out.write(newline)
    #out.write('\n')
    tokenized_query = Q.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    ranked_list=bm25.get_top_n(tokenized_query, corpus, n=2)
    ranked_list[0]=ranked_list[0].replace('\n','')
    if ranked_list[0]!=tmp[4]: # if bm25 selects the original answer, we use the the second nearest answer
       # print(ranked_list[0])
        #print(tmp[4])

        nearest_A=ranked_list[0]
    else:
        nearest_A=ranked_list[1]
    nearest_A = nearest_A.replace('\n', '')
    label = '0'
    newline = str(i) + '\t' + str(i) + '\t' + str(i) + '\t' + Q + '\t' + nearest_A + '\t' + label + '\n'
    out.write(newline)
    i+=1
out.close()
input_Q.close()
input_A.close()


