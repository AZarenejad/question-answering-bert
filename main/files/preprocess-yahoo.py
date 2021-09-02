# this file is for creating a version of yahoo QA alignment which have label 1 and 0
import os
import torch
import random
"""input_path_Q='C:\shima\PHD\search\QA\dataset\yasin_data\YahooFiles\Q-A-all.txt'
input_path_A='./yahoo-A.txt'
out=open('./yahoo-BERT.txt','w',encoding="utf-8")
"""
"""input_path_Q='./yahoo-BERT-label1.txt'
input_path_A='./yahoo-A.txt'
out=open('./yahoo-BERT-tmp.txt','w',encoding="utf-8")"""

input_path_Q='../../hdd/ghasemi/yahoo-BERT-label1.txt'
input_path_A='../../hdd/ghasemi/yahoo-A.txt'
out=open('../../hdd/ghasemi/yahoo-BERT.txt','w',encoding="utf-8")

input_Q=open(input_path_Q,'r',encoding="utf-8")
input_A=open(input_path_A,'r',encoding="utf-8")

lines_Q=input_Q.readlines()
lines_A=input_A.readlines()
i=0
num=0

for line_Q in lines_Q:
    newline = line_Q
    tmp=line_Q.split('\t')
    Q=tmp[3]
    out.write(newline)
    """ tmp = line_Q.split('Q_A_alignment')
    Q = tmp[0]
    A_best = tmp[1]
    A_best=A_best.replace('\n','')
    label = '1'
    newline = str(i) + '\t' + str(i) + '\t' + str(i) + '\t' + Q + '\t' + A_best + '\t' + label + '\n'
    out.write(newline)"""

    j=random.randint(1,len(lines_A)-1)
    A=lines_A[j]
    A = A.replace('\n', '')
    label='0'
    newline=str(i)+'\t'+str(i)+'\t'+str(i)+'\t'+Q+'\t'+A+'\t'+label+'\n'
    #newline=A+'\n'
    out.write(newline)
    #out.write('\n')
    i+=1
out.close()
input_Q.close()
input_A.close()

