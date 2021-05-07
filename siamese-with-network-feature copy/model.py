from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

class siamese(nn.Module):
  '''LSTM'''
  def __init__(self,hid_sz,device,num_layer = 2,inp_sz = 100):
    super(siamese, self).__init__()
    
    #initalizing different asspects of encoder
    self.inp_sz = inp_sz
    self.hid_sz = inp_sz
    self.num_layer = num_layer
    self.device = device

    #self.embedding = nn.Embedding(self.dict_sz,self.inp_sz)
    #self.linear = nn.Linear(self.inp_sz,self.hid_sz)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.1)

    self.lstm = nn.LSTM(input_size = self.inp_sz,
                        hidden_size = self.hid_sz,
                        num_layers = self.num_layer,
                        batch_first = True,
                        bidirectional = True,
                        dropout = 0.1)
    
    self.linear0 = nn.Linear(self.hid_sz*2, self.hid_sz, bias= False)

    self.weight = nn.Parameter(torch.randn(self.hid_sz))

    # softmax layer for attention
    self.attn_softmax = nn.Softmax(dim=1)

    self.linear1 = nn.Linear(self.hid_sz*2*4,self.hid_sz)
    self.linear2 = nn.Linear(self.hid_sz,self.hid_sz)
    self.linear3 = nn.Linear(self.hid_sz,2)
    self.log_softmax = nn.LogSoftmax(dim=1)

    for lin in [self.linear1, self.linear2, self.linear3]:
      nn.init.xavier_uniform_(lin.weight)
      nn.init.zeros_(lin.bias)
    
  def forward(self, head, body):
    batch_size = head.shape[0]
    #head_emm = self.dropout(self.embedding(head.long()))
    #body_emm = self.dropout(self.embedding(body.long()))

    out_head,_ = self.lstm(head)
    out_body,_ = self.lstm(body)

    out_head_1 = torch.tanh(self.linear0(out_head))
    out_body_1 = torch.tanh(self.linear0(out_body))

    out_head_1 = self.attn_softmax(torch.matmul(out_head_1,self.weight))
    out_body_1 = self.attn_softmax(torch.matmul(out_body_1,self.weight))
    
    out_head_1 = torch.bmm(out_head.transpose(1, 2) ,out_head_1.view(batch_size,-1,1)).squeeze(2)
    out_body_1 = torch.bmm(out_body.transpose(1, 2) ,out_body_1.view(batch_size,-1,1)).squeeze(2)

    output = torch.cat((out_head_1,out_body_1,torch.abs(out_head_1-out_body_1),out_head_1*out_body_1),dim=1)

    output = self.relu(self.dropout(self.linear1(output)))
    output = self.relu(self.dropout(self.linear2(output)))
    output = self.log_softmax(self.linear3(output))
    return output
  
  def init_hidden(self, batch_sz):
    tensor1 = torch.zeros(self.num_layer, batch_sz, self.hid_sz ,device=self.device)
    tensor2 = torch.zeros(self.num_layer, batch_sz, self.hid_sz ,device=self.device)
    return (tensor1,tensor2)