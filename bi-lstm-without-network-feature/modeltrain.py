from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

from model import *

class main_model(nn.Module):
  def __init__(self,dict_sz,hid_sz,device):
    super(main_model, self).__init__()

    self.dict_sz = dict_sz
    self.hid_sz = hid_sz
    self.batch_size = 128
    self.device = device
  
    self.bilstm_model = bilstm(dict_sz,hid_sz,device,self.batch_size)

  def train(self,head,body,label,n_epochs,reload,batch_size = 128):
    # initalizing Adam optimizer and NLLLoss(Negitive log liklyhood loss)
    optimizer = optim.SGD(self.parameters(), lr = 0.001)
    criterion = nn.NLLLoss()

    # calculate number of batch iterations
    n_batches = int(len(head)/ self.batch_size)

    if reload:
      # if True, reload the model from saved checkpoint and retrain the model from epoch and batch last stoped training 
      optimizer, start_epoch, batch_id = self.load_ckp(path+"checkpoint.pt", optimizer)
      print("\nResuming training from epoch {}, batch {}".format(start_epoch,batch_id))
    else:
      # else start training from epoch 0
      start_epoch = 0
    
    start_epoch = n_epochs + 1
    for i in range(start_epoch,n_epochs):
      r = torch.randperm(head.shape[0])
      head = head[r[:, None]].squeeze(1)
      body = body[r[:, None]].squeeze(1)
      label = label[r[:, None]].squeeze(1)

      # epoch start time
      stime = datetime.now().timestamp()

      # initalize batch_loss to 0.
      batch_loss = 0.
      
      if reload:
        # if True, set start_batch to where left of
        if batch_id >= n_batches:
          start_batch = 0
        else:
          start_batch = batch_id
        # set reload to true so in next epoch batch starts from 0
        reload = False
      else:
        # else set to 0
        start_batch = 0

      for b in range(start_batch,n_batches):

        head_batch = head[b*batch_size: (b*batch_size) + batch_size]
        body_batch = body[b*batch_size: (b*batch_size) + batch_size]
        label_batch = label[b*batch_size: (b*batch_size) + batch_size]

        # initialize outputs tensor to save output from the training
        outputs = torch.zeros(batch_size,2, requires_grad=True).to(self.device)

        # zero the gradient
        optimizer.zero_grad()
        
        outputs = self.bilstm_model(head_batch,body_batch)
        
        loss = criterion(outputs,label_batch)

        batch_loss += loss.item()

        # backpropagation
        loss.backward()
        optimizer.step()

        #saving the checkpoint to resume training
        if (b+1 == n_batches):
          # save checkpoint if it's last batch
          self.save_ckp(i+1,b,optimizer,path+'checkpoint.pt')
        
        # show progress to training
        if (b+1)%(int((n_batches-start_batch)/5)) == 0:
          # print detailed progress only 10 times in a batch
          tempir = b - start_batch
          if tempir == 0:
            tempir = 1
          time_escaped = int(datetime.now().timestamp() - stime)
          time_expected = int((time_escaped/tempir)*(n_batches-start_batch))
          time_remaining = time_expected - time_escaped
          print("epoch {} batch {} of {}, time escaped = {}s, expected ={}s, remaining {}s,  batch loss = {}".format(i,b,n_batches,time_escaped,time_expected,time_remaining,batch_loss/(tempir)))
          self.save_ckp(i,b,optimizer,path+'checkpoint.pt')

      # loss for epoch 
      batch_loss = batch_loss/(n_batches-start_batch)
      # saves loss to file to loss change information is stores if training stops and training is resumed
      self.save_loss(i,batch_loss)

      # set endtime to calculate time taken for the epoch and print loss for the epoch
      etime = datetime.now().timestamp()
      print("\nepoch {} over, time taken = {}s, loss = {}".format(i,int(etime-stime),batch_loss))

  def save_loss(self,epoch,loss):
    '''saves loss value to loss.csv'''
    with open(path+'loss.csv', 'a') as file:
      file.write("{},{}\n".format(epoch,loss))
      file.close()

  #Thanks to https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
  # code have been modified as per the need of this model
  def save_ckp(self,epoch,batch_id,optimizer,checkpoint_dir):
    #save checkpoint to resume training
    checkpoint = {
        'batch_id':batch_id + 1,
        'epoch': epoch,
        'bilstm_state_dict': self.bilstm_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_dir)

  def load_ckp(self,checkpoint_fpath, optimizer):
    # reloads the model from the checkpoint
    checkpoint = torch.load(checkpoint_fpath)
    self.bilstm_model.load_state_dict(checkpoint['bilstm_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return ( optimizer, checkpoint['epoch'],checkpoint['batch_id'] )

  def predict(self,head,body):
    batch_size = head.shape[0]
    output = self.bilstm_model.forward( head , body )
    out = []
    #print(output.shape)
    for i in range(batch_size):
      #print(output[i].topk(1))
      topv, topi = output[i].topk(1)
      #print(topi[0])
      out.append(int(topi[0]))
    return out
