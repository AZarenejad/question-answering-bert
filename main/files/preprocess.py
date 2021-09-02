# this file is for creating a version of yahoo QA alignment which all of the words exist in glove
import os
import torch
import csv
#input_path='./yahoo-BERT.txt'
input_path='../../hdd/ghasemi/Quora/train.txt'
out=open('../../hdd/ghasemi/Quora/train.tsv','w',encoding="utf-8")
input=open(input_path,'r',encoding="utf-8")

#lines=input.readlines()
i=0
num=0

for tmp in csv.reader(input):
    #line=line.replace('\"','')
    #tmp=line.split(',')
    if len(tmp)!=6:
        print('!=6 ,line num:',i,tmp)
        i+=1

        continue
    id1 = tmp[0]
    id2 = tmp[1]
    id3=tmp[2]
    Q1=tmp[3]
    Q2=tmp[4]
    label=tmp[5].replace('\n','')
    if Q2=='' or Q1=='' or label=='':
        print('empty field, line num:',i,tmp)
        i+=1
        continue
    #print(str(i)+'\n')
    newline=id1+'\t'+id2+'\t'+id3+'\t'+Q1+'\t'+Q2+'\t'+label
    newline=newline.replace('\"','')
    out.write(newline)
    if i!=len(tmp)-1:
        out.write('\n')
    i+=1
out.close()
out=open('../../hdd/ghasemi/Quora/train.tsv','r',encoding="utf-8")
lines2=out.readlines()
print(len(lines2) )
input.close()
out.close()
