import pandas as pd
import numpy as np
import re
import random
import os, errno
import sys
import csv
import math

def partition(df):
  #ratio = 5000/len(df.loc[df['Label'] == -1])
  ratio = 0.10
  df_test0 = df.loc[df['Label'] == -1].sample(frac = ratio)
  #ratio = 5000/len(df.loc[df['Label'] == 1])
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

def load_train(reload):
  if not reload:
    head = load_list(path+'head.csv')
    body = load_list(path+'body.csv')
    label = load_label(path+'label.csv')
  else:
    print('partitioning')
    df_train,df_test = partition(load_new(reload = False))
    print("size of train: {}, test: {}".format(len(df_train),len(df_test)))
    df_train.to_csv(r''+path+'train.csv')
    df_test.to_csv(r''+path+'test.csv')
    head = [list(line.split(' ')) for line in df_train['Head']]
    body = [list(line.split(' ')) for line in df_train['Body']]
    label = []
    for t in df_train['Label']:
      if t == -1:
        label.append(0)
      else:
        label.append(1)
    print('saving')
    save_list(head,path+'head.csv')
    save_list(body,path+'body.csv')
    save_label(label,path+'label.csv')
  return (head,body,label)
