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


def load_vocab(file_name):
  '''loading vocab from file_name and generating word2index and index2word'''
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

def embedding(lst,lim):
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
  return torch.FloatTensor(emb)

#checking if cuda is avilable
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

head_vec = embedding(new_head,40).to(device)
body_vec = embedding(new_body,100).to(device)
label_vec = torch.LongTensor(new_label).to(device)

model = main_model(189652,300,device).to(device)
model.train(head_vec,body_vec,label_vec,100,False)

torch.save(model, path+"model")

test(model,path)
