from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

class bilstm(nn.Module):
  '''bi-LSTM'''
  def __init__(self,dict_sz,hid_sz,device,batch_size,num_layer = 2,inp_sz = 300):
    super(bilstm, self).__init__()
    
    #initalizing different asspects of encoder
    self.dict_sz = dict_sz
    self.inp_sz = inp_sz
    self.hid_sz = hid_sz
    self.num_layer = num_layer
    #self.batch_size = batch_size
    self.device = device

    self.embedding = nn.Embedding(self.dict_sz,self.inp_sz)
    #self.linear = nn.Linear(self.inp_sz,self.hid_sz)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.1)

    self.lstm = nn.LSTM(input_size = self.inp_sz,
                        hidden_size = self.hid_sz,
                        num_layers = self.num_layer,
                        batch_first = True,
                        bidirectional = True,
                        dropout = 0.1)

    self.linear1 = nn.Linear(self.hid_sz*2*4,self.hid_sz)
    self.linear2 = nn.Linear(self.hid_sz,self.hid_sz)
    self.linear3 = nn.Linear(self.hid_sz,2)
    self.log_softmax = nn.LogSoftmax(dim=1)

    for lin in [self.linear1, self.linear2, self.linear3]:
      nn.init.xavier_uniform_(lin.weight)
      nn.init.zeros_(lin.bias)
    
  def forward(self, head, body):
    batch_size = head.shape[0] 
    head_emm = self.dropout(self.embedding(head.long()))
    body_emm = self.dropout(self.embedding(body.long()))

    _,self.head_h = self.lstm(head_emm)
    _,self.body_h = self.lstm(body_emm)

    print(self.head_h[0].shape)
    print(self.body_h[0].shape)
    head_hid = self.head_h[0][-2:].transpose(0, 1).contiguous().view(batch_size, -1)
    body_hid = self.body_h[0][-2:].transpose(0, 1).contiguous().view(batch_size, -1)


    both = torch.cat((head_hid,body_hid,torch.abs(head_hid-body_hid),head_hid*body_hid),dim=1)
    
    output = self.relu(self.dropout(self.linear1(both)))
    output = self.relu(self.dropout(self.linear2(output)))
    output = self.log_softmax(self.linear3(output))
    return output
  
  def init_hidden(self, batch_sz):
    tensor1 = torch.zeros(self.num_layer, batch_sz, self.hid_sz ,device=self.device)
    tensor2 = torch.zeros(self.num_layer, batch_sz, self.hid_sz ,device=self.device)
    return (tensor1,tensor2)
