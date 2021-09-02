from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import *
from nltk.stem import PorterStemmer
import math
#yahoofile=open('C:\shima\PHD\search\QA\dataset\yasin_data\YahooFiles\Q-A-all-glove.txt','r',encoding='utf8')
#path='C:\\shima\\PHD\\search\\QA\\dataset\\yasin_data\\YahooFiles\\partitions_Q\\'
path='../../hdd/ghasemi/Partitions_Q/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))

for f in files:
    print(f)

stop_words = stopwords.words('english') + list(punctuation)
newStopWords = ['need','know','would','like','think','want','help','one','please','get','anyone','really','find','good','also'\
    ,'go','thank','thanks','make','even','look','looking','ones','use','lot','much','give','see','tell','something'\
    ,'always','anything','best','day','dont','mean','right','guy','guys','never','say','people','going','play','played'\
    ,'got','time','name','tried','try','love','still','back','last','im','years','well','year','long','better','give',\
                'buy','use','used','u','place','places','keep','way','top','two','sure','work','getting','feel','told','thing','things','new','read',\
                'hear','everyone','mean','could','many','question','said','belive','friend','friends','take','someone'\
                ,'since','ideas','wondering','around','old','without','home','ever','yahoo','world','us','went','girl','etc','yet','little','great','different','start','come','able','let','else','started','times','ago','made','real','seems','away','may','every','belive','days','problem','ok','answers','found','life']
stop_words.extend(newStopWords)
def tokenize(text):

    #words = word_tokenize(text)
    punc = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
    text = text.replace("'", "")
    for c in punc:
        text = text.replace(c, " ")
    text = text.replace("  ", " ")
    words=text.split(' ')
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit() and len(w)>1]

#****************build the vocabulary in one pass
"""vocabulary = set()
ps = PorterStemmer()
lines=yahoofile.readlines()
for line in lines:
    words = tokenize(line)
    i=0
    for word in words:
        #word=ps.stem(word)#********stemming
        words[i]=word
        i+=1
    vocabulary.update(words)

vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}

VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(lines)

print(VOCABULARY_SIZE, DOCUMENTS_COUNT)
"""
#lines=yahoofile.readlines()
#DOCUMENTS_COUNT = len(lines)
#yahoofile.close()
lines=[]
#*************************
count_Q={}
count_A={}
co_occur={}
I={}
counter=0
"""for w in vocabulary:
    count_Q[w] = 0
    count_A[w] = 0
"""
DOCUMENTS_COUNT=0
tmp_counter=0
for f in files:
    tmp_counter+=1
    if tmp_counter>10:
        break
    print(f+'\n')
    file_content=open(f,'r',encoding='utf8')
    lines=file_content.readlines()
    for line in lines:
        DOCUMENTS_COUNT+=1
        #tmp=line.split('Q_A_alignment')
        Q=line
        Q = ' '.join(set(Q.lower().split())) # remove duplicate from a question
        #A=tmp[1]
        tmp1 = {}
        tmp2 = {}
        first_time=0
        #for w1 in tokenize(Q):
        for i in range(0, len(tokenize(Q))):
            w1 = tokenize(Q)[i]
            if w1 not in count_Q:
                count_Q[w1]=1
            else:
                #if w1 not in tmp1:
                count_Q[w1] += 1
                 #   tmp1[w1]=1
                #else:
                 #   continue
            counter+=1
            if counter%10000==0:
                print('counter=',counter)

            """
            ##***************check
            Q = []
            A = []
            [Q.append(x) for x in tmp[0] if x not in Q]
            [A.append(x) for x in tmp[1] if x not in A]
            """
            #print('Q:',Q,'   A:',A)
            #for w1 in Q:
            tmp3 = {}
            #for w2 in tokenize(Q):
            for j in range(i+1,len(tokenize(Q))):
                w2=tokenize(Q)[j]
                if w2 not in count_Q:
                    count_Q[w2] = 1
                else:
                    count_Q[w2] += 1

                counter += 1
                #count_A[w]+=A.count(w)
                """if w2 not in count_A:
                    count_A[w2]=1
                else:
                    if not w2 in tmp2:
                        count_A[w2] += 1
                        tmp2[w2]=1
                """
                #if not w2 in tmp3:
                # tmp3[w2] = 1
                key=w1+' co_occur '+w2
                if not key in co_occur:
                    co_occur[key]=1
                else:
                    co_occur[key]+=1
                    #else:
                        #print`('here')
            tmp3.clear()
        tmp1.clear()
        #tmp2.clear()
    file_content.close()

#print('count:',count)
#print('co_occur:',co_occur)
#***************** MI calculation***************
#N=len(lines)           #*************************************check the truth
N=DOCUMENTS_COUNT
MI={}

#N=100
for w1 in count_Q:   # vocab of questions
    for w2 in count_Q:        #vocab of answers
        if w1==w2:
            continue
        P_w1_1=count_Q[w1]/N  #p(w1=1)
        P_w1_0=1- P_w1_1    #p(w1=0)

        P_w2_1=count_Q[w2]/N  #p(w2=1)
        P_w2_0=1- P_w2_1    #p(w2=0)

        if (not w1+' co_occur '+w2 in co_occur) and (not w2+' co_occur '+w1 in co_occur):
            continue
            #co_occur[w1 + ' co_occur ' + w2]=0
        if not w1+' co_occur '+w2 in co_occur:
            co_occur[w1 + ' co_occur ' + w2]=0
        if not w2+' co_occur '+w1 in co_occur:
            co_occur[w2 + ' co_occur ' + w1]=0
        P_1_1=(co_occur[w1+' co_occur '+w2]+co_occur[w2+' co_occur '+w1])/N  #P(w1=1,w2=1)
        t1=P_1_1
        P_1_0=(count_Q[w1]-(co_occur[w1+' co_occur '+w2]+co_occur[w2+' co_occur '+w1]))/N      #p(w1=1,w2=0)
        t2=P_1_0
        P_0_1=(count_Q[w2]-(co_occur[w1+' co_occur '+w2]+co_occur[w2+' co_occur '+w1]))/N      #p(w1=0,w2=1)
        t3=P_0_1
        #P_0_0= 1 - P_0_1 - P_1_0 - P_1_1
        P_0_0 = 1 -(t1+t2+t3)
        if t1<0 or t2<0 or t3<0 or P_0_0<0:
            print('error')
        """print('P_0_0=',P_0_0)
        print('P_0_1=', P_0_1)
        print('P_1_0=', P_1_0)
        print('P_1_1=', P_1_1)"""

        unit1=0
        unit2=0
        unit3=0
        unit4=0

        if P_0_0 != 0:
            if not (P_0_0/(P_w1_0*P_w2_0))>0:
                print('error is here')
            unit1=P_0_0 * math.log2(P_0_0/(P_w1_0*P_w2_0))
        if P_0_1!=0:
            if not (P_0_1/(P_w1_0*P_w2_1))>0:
                print('error is here, P_w1_0,P_w2_1',P_w1_0,' ',P_w2_1)
            unit2=P_0_1 * math.log2(P_0_1/(P_w1_0*P_w2_1))
        if P_1_0!=0:
            if not (P_1_0/(P_w1_1*P_w2_0))>0:
                print('error is here')
            unit3=P_1_0 * math.log2(P_1_0/(P_w1_1*P_w2_0))
        if P_1_1!=0:
            if not (P_1_1/(P_w1_1*P_w2_1))>0:
                print('error is here')
            unit4=P_1_1 * math.log2(P_1_1/(P_w1_1*P_w2_1))
        I_w1_w2=unit1+unit2+unit3+unit4
        #del co_occur[w1 + ' co_occur ' + w2]
        I[w1+' co_occur '+w2]=I_w1_w2 # for memory saving we put I in co_occur again
#co_occur={}
print('end of MI')
#****************************************************
"""P_MI={}
prob_check=0
for w1 in count_Q:
    if count_Q[w1] == 0:
        continue
    for w2 in count_A:
        denominator = 0
        if count_A[w2]==0:
            continue
        for w3 in count_Q:
            if count_Q[w3]==0:
                continue
            #if count_A[w2]==0 or count[w3]==0:
            if not w3+' co_occur '+w2 in MI:
                  continue
            else:
                denominator+=MI[w3+' co_occur '+w2]
        if denominator!=0 and w1+' co_occur '+w2 in MI:
            P_MI[w1+' co_occur '+w2]= MI[w1+' co_occur '+w2]/denominator
        prob_check+=P_MI[w1+' co_occur '+w2]
    print('prob_check:',prob_check)
    prob_check=0"""
#****************PMI calculation******************
#out=open('iis@172.16.145.90:~/hdd/ghasemi/probabilistic-dictionary-only-Q','w',encoding='utf8')
out=open('../../hdd/ghasemi/probabilistic-dictionary-only-Q-10','w',encoding='utf8')
P_MI={}
denominator={}
for w2 in count_Q:
    denominator[w2]=0
    for w1 in count_Q:
        if not w1 + ' co_occur ' + w2 in I:
            continue
        else:
            denominator[w2]+= I[w1 + ' co_occur ' + w2]
#*************
for w2 in count_Q:
    #print(w2,'\n')
    for w1 in count_Q:
        if not w1 + ' co_occur ' + w2 in I:
            continue
        else:
            P_MI[w1 + ' co_occur ' + w2]=I[w1 + ' co_occur ' + w2]/denominator[w2]
    #**************** sort and normalization********
    ii=0
    sum_of_ten=0
    most_probable={}
    for mykey in sorted(P_MI, key=P_MI.get, reverse=True):
        sum_of_ten+=P_MI[mykey]
        most_probable[mykey]=P_MI[mykey]
        ii+=1
        if ii==10:
            break
    P_MI={}
    for mykey in most_probable:
        out.write(str(mykey+'  '+str(most_probable[mykey]/sum_of_ten)+'\n'))
#****************write in the file*****************
"""out.write(str(MI))
out.write('\n')
out.write(str(P_MI))"""
out.close()