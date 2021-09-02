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
input_path_Q='./yahoo-topic.txt'
input_path_topic='./only-topics.txt'
out=open('./yahoo-BERT-topic-3.txt','w',encoding="utf-8")
"""
input_path_Q='../../hdd/ghasemi/yahoo_topic/yahoo-topic.txt'
input_path_topic='../../hdd/ghasemi/yahoo_topic/only-topics.txt'
out=open('../../hdd/ghasemi/yahoo_topic/yahoo-BERT-topic-3.txt','w',encoding="utf-8")

input_Q=open(input_path_Q,'r',encoding="utf-8")
input_topic=open(input_path_topic,'r',encoding="utf-8")

lines_Q=input_Q.readlines()
lines_topic=input_topic.readlines()
i=0
num=0
for line in lines_topic:

    corpus.append(line)
    tmp+=1


tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

for line_Q in lines_Q:
    newline = line_Q
    tmp=line_Q.split('\t')
    Q=tmp[0]
    label='1'
    Q_topic=tmp[1].replace('\n','')
    newline = str(i) + '\t' + str(i) + '\t' + str(i) + '\t' + Q + '\t' + Q_topic + '\t' + '1' + '\n'
    out.write(newline)

    j=random.randint(1,len(lines_topic)-1)
    topic=lines_topic[j]
    topic = topic.replace('\n', '')
    while topic==Q_topic:
        j = random.randint(1, len(lines_topic) - 1)
        topic = lines_topic[j]
        topic = topic.replace('\n', '')
    label='0'
    newline=str(i)+'\t'+str(i)+'\t'+str(i)+'\t'+Q+'\t'+topic+'\t'+label+'\n'
    #newline=A+'\n'
    out.write(newline)
    #out.write('\n')
    """
    ########### bm25 works good but it is not good for this task
    tokenized_query = Q.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    ranked_list=bm25.get_top_n(tokenized_query, corpus, n=2)
    ranked_list[0]=ranked_list[0].replace('\n','')
    if ranked_list[0]!=Q_topic: # if bm25 selects the original answer, we use the the second nearest answer
        print(ranked_list[0])
        print(Q_topic)
        nearest_topic=ranked_list[0]
    else:
        nearest_topic=ranked_list[1]
    nearest_topic = nearest_topic.replace('\n', '')
    label = '0'
    newline = str(i) + '\t' + str(i) + '\t' + str(i) + '\t' + Q + '\t' + nearest_topic + '\t' + label + '\n'
    out.write(newline)
    #############end of bm25
    """
    ####### another random topic
    j = random.randint(1, len(lines_topic) - 1)
    topic = lines_topic[j]
    topic = topic.replace('\n', '')
    while topic==Q_topic:
        j = random.randint(1, len(lines_topic) - 1)
        topic = lines_topic[j]
        topic = topic.replace('\n', '')
    label = '0'
    newline = str(i) + '\t' + str(i) + '\t' + str(i) + '\t' + Q + '\t' + topic + '\t' + label + '\n'
    # newline=A+'\n'
    out.write(newline)
    i+=1
out.close()
input_Q.close()
input_topic.close()


