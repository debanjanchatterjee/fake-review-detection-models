import pandas as pd
import numpy as np
import re
import random
import os, errno
import sys
import csv
import math


def partition(df):
  ratio = 3000/len(df.loc[df['Label'] == 0])
  df_test0 = df.loc[df['Label'] == 0].sample(frac = ratio)
  ratio = 3000/len(df.loc[df['Label'] == 1])
  df_test1 = df.loc[df['Label'] == 1].sample(frac = ratio)
  df_test = df_test1.append(df_test0)
  df_train = df.drop(df_test.index)
  df_test.reset_index(inplace = True)
  df_train.reset_index(inplace = True)
  return(df_train,df_test)

def save_list(lst,file_name):
  '''saves list of lists'''
  with open(file_name, 'w') as file:
    for i in lst:
      for t in i:
        file.write(t+",")
      file.write('\n')

def save_label(lst,file_name):
  '''saves list of lists'''
  with open(file_name, 'w') as file:
    for i in lst:
      file.write(str(i)+'\n')
      
def load_list(file_name):
  '''load list of lists'''
  lst = []
  with open(file_name, 'r') as file:
    t = file.read()
    t = t.split('\n')
    for i in t:
      lst.append(i.split(',')[:-1])
  return lst[:-1]

def load_label(file_name):
  '''load list of lists'''
  lst = []
  with open(file_name, 'r') as file:
    t = file.read()
    t = t.split('\n')
    for i in t:
      if len(i) > 0:
        lst.append(int(i))
  return lst

def load_feature(file_name):
  '''load list of lists'''
  lst = []
  with open(file_name, 'r') as file:
    t = file.read()
    t = t.split('\n')
    for i in t:
      temp = []
      for j in i.split(',')[:-1]:
        temp.append(float(j))
      lst.append(temp)
  return lst[:-1]

def load_train(reload,path):
  '''load train data,if true then partition dataset.csv else read from train.csv'''
  if reload:
    df = pd.read_csv(path+'dataset.csv')
    print('partitioning')
    df_train,df_test = partition(df)
    print("size of train: {}, test: {}".format(len(df_train),len(df_test)))
    df_train.to_csv(r''+path+'train.csv')
    df_test.to_csv(r''+path+'test.csv')
  else:
    df_train = pd.read_csv(path+'train.csv')
  head = [list(line.split(' ')) for line in df_train['Head']]
  body = [list(line.split(' ')) for line in df_train['Body']]
  features = []
  for line in df_train['Features']:
    lst = []
    for t in line.split(' '):
      lst.append(float(t))
    features.append(lst)
  label = [t for t in df_train['Label']]
  return (head,body,label,features)
