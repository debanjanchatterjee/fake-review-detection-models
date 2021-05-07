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

head,body,label,features = load_train(reload = True, path = path)

# selection 50k points from dataset
new_head = []
new_body = []
new_features = []
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
      new_features.append(features[i])
      new_label.append(1)
  else:
    new_head.append(head[i])
    new_body.append(body[i])
    new_features.append(features[i])
    new_label.append(0)

def transform_featues(file_name):
  features_vec = torch.load(file_name)
  lst = []
  for i in features_vec:
    feat = []
    #user_median 0
    feat.append( int(i[0]) / 5 )
    #user_mean 1
    feat.append( float(i[1]) / 5 )
    #user_count 2
    feat.append( int(float(i[2])) / 400 )
    #prod_median 3
    feat.append( int(i[3]) / 5 )
    #prod_mean 4
    feat.append( float(i[4]) / 5 )
    #prod_count 5
    feat.append( int(i[5]) / 100 )
    #length 6
    feat.append( int(i[6]) / 400 )
    #rating 7
    feat.append( int(i[7]) / 5 )
    #median_diff 8
    feat.append( feat[3] - feat[0] )
    #mean_diff 9
    feat.append( feat[4] - feat[1] )
    #user_diff 10
    feat.append( feat[7] - feat[0] )
    #user_diff1 11
    feat.append( feat[7] - feat[2] )
    #prod_diff 12
    feat.append( feat[7] - feat[3] )
    #prod_diff1 13
    feat.append( feat[7] - feat[4] )
    lst.append(feat)
  torch.save(torch.FloatTensor(lst),file_name)

def load_vocab(file_name):
  '''loading vocab from file_name and generating word2index'''
  #reading file
  with open(file_name, 'r') as file:
    lst = file.read()
    lst = lst.split('\n')
    file.close()
  word2index = {'pad':0}
  # populating dictionary
  for i in range(len(lst)):
    word2index[lst[i]] = i +1
  return (word2index)

word2index= load_vocab(path+'vocab.csv')

#embedding
def embedding(lst,lim,file_name):
  emb = []
  for i in lst:
    temp = []
    count = 0
    for t in i:
      if count < lim:
        temp.append(word2index[t])
        count += 1
    for t in range(count,lim):
      temp.append(0)
    emb.append(list(temp[:lim]))
  
  torch.save(torch.FloatTensor(emb), file_name)

embedding(new_head,40,path+'head_vec.pt')
embedding(new_body,100,path+'body_vec.pt')
torch.save(torch.FloatTensor(new_features),path+'features_vec.pt')
torch.save(torch.LongTensor(new_label),path+'label_vec.pt')

transform_featues(path+'features_vec.pt')

#checking if cuda is avilable
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

head_in_vec = torch.load(path+'head_vec.pt').to(device)
body_in_vec = torch.load(path+'body_vec.pt').to(device)
features_vec = torch.load(path+'features_vec.pt').to(device)
label_vec = torch.load(path+'label_vec.pt').to(device)

model = main_model(189652,300,device).to(device)
model.train(head_in_vec,body_in_vec,features_vec,label_vec,500,False)

test(model,path)
