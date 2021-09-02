# this file is for creating a version of yahoo QA alignment which all of the words exist in glove
import os
import torch
input=open('../../hdd/ghasemi/yahoo-BERT-3-1M.txt','r',encoding="utf-8")
output=open('../../hdd/ghasemi/yahoo-BERT-3-small.txt','w',encoding="utf-8")

lines=input.readlines()
i=0
num=0

for line in lines:
    i += 1
    if i<50000:
        output.write(line)

input.close()
output.close()
