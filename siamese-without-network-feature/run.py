import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
nltk.download('stopwords')
stopword = set(stopwords.words('english'))

import re
import random
import os, errno
import sys
import csv
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

from data_partition import *
from model import *
from modeltrain import *
from testing import *

#set path
path = ''

head,body,label = load_train(reload = False,path = path)

#https://stackoverflow.com/a/52070223/8475746
glove = pd.read_csv(path+'glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)
t_glove = {key: val.values for key, val in glove.T.items()}
del glove

new_head = []
new_body = []
new_label = []
positive = 0
negitive = 0
for i in label:
  if i == 1:
    positive +=1
  else:
    negitive +=1
till = 50000 - negitive
count = 0 
for i in range(len(body)):
  if label[i]:
    if count < till:
      count +=1
      new_head.append(head[i])
      new_body.append(body[i])
      new_label.append(1)
  else:
    new_head.append(head[i])
    new_body.append(body[i])
    new_label.append(0)

def embedding(lst,lim,file_name):
  emb = []
  for i in lst:
    temp = []
    count = 0
    for t in i:
      if count < lim:
        try:
          temp.append(list(t_glove[t]))
        except:
          temp.append(list(t_glove['unk']))
        count += 1
    for t in range(count,lim):
      temp.append([0]*100)
    emb.append(list(temp[:lim]))

  torch.save(torch.FloatTensor(emb), file_name)

embedding(new_head,40,path+'head_vec.pt')
embedding(new_body[:10000],100,path+'body_vec1.pt')
embedding(new_body[10000:20000],100,path+'body_vec2.pt')
embedding(new_body[20000:30000],100,path+'body_vec3.pt')
embedding(new_body[30000:40000],100,path+'body_vec4.pt')
embedding(new_body[40000:50000],100,path+'body_vec5.pt')
torch.save(torch.LongTensor(new_label),path+'label_vec.pt')

#checking if cuda is avilable
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

head_vec = torch.load(path+'head_vec.pt').to(device)
body_vec1 = torch.load(path+'body_vec1.pt')
body_vec2 = torch.load(path+'body_vec2.pt')
body_vec3 = torch.load(path+'body_vec3.pt')
body_vec4 = torch.load(path+'body_vec4.pt')
body_vec5 = torch.load(path+'body_vec5.pt')
label_vec = torch.load(path+'label_vec.pt').to(device)

body_vec = torch.cat((body_vec1,body_vec2,body_vec3,body_vec4,body_vec5),dim = 0).to(device)
del body_vec1,body_vec2,body_vec3,body_vec4,body_vec5

model = main_model(100,device).to(device)
model.train(head_vec,body_vec,label_vec,500,False)

torch.save(model, path+"model")

test(model,path)
