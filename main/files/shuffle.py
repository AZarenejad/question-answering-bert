
import os
import torch
import random
import random
input_path='../../hdd/ghasemi/yahoo-BERT-3-v2.txt'
out=open('../../hdd/ghasemi/yahoo-BERT-3.txt','w',encoding="utf-8")

input=open(input_path,'r',encoding="utf-8")

lines = input.readlines()
random.shuffle(lines)
out.writelines(lines)

out.close()
input.close()
