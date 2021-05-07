import pandas as pd
import numpy as np
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

#set path to loaction with Labelled Yelp Dataset.csv
path = ''
if path == '':
  print('set path to loaction with Labelled Yelp Dataset.csv')

def load_new(reload):

  '''Load data, True if loading data from scratch else False'''
  if reload:
    print("\nreading Labelled Yelp Dataset.csv")
    df = pd.read_csv(path+"Labelled Yelp Dataset.csv")

    df1 = df.filter(items=['User_id','Rating']).groupby(by ='User_id').median()
    user_rat_med = {df1.iloc[i].name:df1.iloc[i][0] for i in range(len(df1))}

    df1 = df.filter(items=['User_id','Rating']).groupby(by ='User_id').mean()
    user_rat_mean = {df1.iloc[i].name:df1.iloc[i][0] for i in range(len(df1))}

    df1 = df.filter(items=['User_id','Rating']).groupby(by ='User_id').count()
    user_rat_count = {df1.iloc[i].name:df1.iloc[i][0] for i in range(len(df1))}

    df1 = df.filter(items=['Product_id','Rating']).groupby(by ='Product_id').median()
    prod_rat_med = {df1.iloc[i].name:df1.iloc[i][0] for i in range(len(df1))}

    df1 = df.filter(items=['Product_id','Rating']).groupby(by ='Product_id').mean()
    prod_rat_mean = {df1.iloc[i].name:df1.iloc[i][0] for i in range(len(df1))}

    df1 = df.filter(items=['Product_id','Rating']).groupby(by ='Product_id').count()
    prod_rat_count = {df1.iloc[i].name:df1.iloc[i][0] for i in range(len(df1))}

    print("\npreparing dataset")
    data = []

    #progress
    end_count = len(df)
    percent = 0

    for i in range(len(df)):
      temp = sent_tokenize(df.iloc[i]['Review'])
      if len(temp) > 1:
        lst = []
        temp = [tokenizer(t) for t in temp]
        l = [len(t) for t in temp]
        head_length = 0
        skip = False
        head_t = []
        current = 0
        while(len(head_t) < 3  and not skip):
          head_t += temp[current]
          current += 1
          if current >= len(temp)-1:
            skip = True
        body_t = [word for t in temp[current+1:] for word in t]

        #additional network features
        t_lst = []
        t_lst.append(str( user_rat_med[df.iloc[i]['User_id']]) )
        t_lst.append(str( user_rat_mean[df.iloc[i]['User_id']]) )
        t_lst.append(str( user_rat_count[df.iloc[i]['User_id']]) )
        t_lst.append(str( prod_rat_med[df.iloc[i]['Product_id']]) )
        t_lst.append(str( prod_rat_mean[df.iloc[i]['Product_id']]) )
        t_lst.append(str( prod_rat_count[df.iloc[i]['Product_id']]) )
        t_lst.append(str( len(temp)) )
        t_lst.append(str( df.iloc[i]['Rating']) )

        if len(body_t) > 5 and len(head_t)  >= 3 :
          strl = ' '
          lst.append(strl.join(head_t))
          lst.append(strl.join(body_t))
          if df.iloc[i]['Label'] == -1:
            lst.append(0)
          else:
            lst.append(1)
          lst.append(strl.join(t_lst))
          data.append(lst)
      
      #tracking progress
      if(percent+1 == int((i/end_count)*100)):
        percent += 1
        print(".",end='')

    print("\nSaving data as Dataframe")
    column_names = ['Head','Body','Label','Features']
    df0 = pd.DataFrame(data, columns=column_names)

    print("saving as csv file for later use")
    df0.to_csv(path+'dataset.csv',index = False, header=True)
  else:
    print("reading dataset")
    df0 = pd.read_csv(path+"dataset.csv")

  return df0

def tokenizer(text):
  '''Removes stop words, symbols and tokenize''' 
  lst = []
  for i in word_tokenize(text):
    # remove symbols
    i = re.sub(r'[^\w]', '', i)
    # remove numbers
    i = re.sub(r'[0-9]+', '', i)
    # convert to lower case
    i = i.lower()
    if i not in stopword and len(i) > 1:
      lst.append(i)
  return lst

a = load_new(True)