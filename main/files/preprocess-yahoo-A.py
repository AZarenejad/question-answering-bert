# this file is for creating a version of yahoo QA alignment which have label 1 and 0
import os
import torch
import random
#input_path='./yahoo-BERT-label1.txt'
input_path='../../hdd/ghasemi/yahoo-BERT-label1.txt'
out=open('../../hdd/ghasemi/yahoo-A.txt','w',encoding="utf-8")
input=open(input_path,'r',encoding="utf-8")
lines=input.readlines()
i=0
num=0

for line in lines:
    tmp = line.split('\t')
    A = tmp[4]
    newline = A + '\n'
    out.write(newline)

out.close()
input.close()

