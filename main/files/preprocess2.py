# this file is for creating a version of yahoo QA alignment which all of the words exist in glove
import os
import torch
input=open('../../hdd/ghasemi/yahoo-BERT-3.txt','r',encoding="utf-8")
output=open('../../hdd/ghasemi/yahoo-BERT-3-7M.txt','w',encoding="utf-8")

lines=input.readlines()
i=0
num=0

for line in lines:
    if i<7000000:
        output.write(line)
        i+=1
input.close()
output.close()