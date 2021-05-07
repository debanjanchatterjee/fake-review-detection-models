from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

#checking if cuda is avilable
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

df_test = pd.read_csv(path+'test.csv')
test_head = [list(line.split(' ')) for line in df_test['Head']]
test_body = [list(line.split(' ')) for line in df_test['Body']]

test_label = [t for t in df_test['Label']]
test_head_vec = embedding(test_head).to(device)
test_body_vec = embedding(test_body).to(device)

predicted_label = []
test_batch_size = 1000
test_n_batch = int(len(test_body_vec)/test_batch_size)-1
for b in tqdm(range(test_n_batch)):
  predicted_label += model.predict(test_head_vec[b*test_batch_size:(b+1)*test_batch_size],
                                   test_body_vec[b*test_batch_size:(b+1)*test_batch_size],
                                   test_head_in_vec[b*test_batch_size:(b+1)*test_batch_size],
                                   test_body_in_vec[b*test_batch_size:(b+1)*test_batch_size],
                                   test_features_vec[b*test_batch_size:(b+1)*test_batch_size])
b = test_n_batch - 1
predicted_label += model.predict(test_head_vec[b*test_batch_size:],
                                   test_body_vec[b*test_batch_size:],
                                   test_head_in_vec[b*test_batch_size:],
                                   test_body_in_vec[b*test_batch_size:],
                                   test_features_vec[b*test_batch_size:])

print("f1_score(macro): {}".format(f1_score(test_label, predicted_label, average="macro")))
