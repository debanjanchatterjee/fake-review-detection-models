from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence


class main_model(nn.Module):
  def __init__(self,dict_sz,hid_sz,device):
    super(main_model, self).__init__()

    self.dict_sz = dict_sz
    self.hid_sz = hid_sz
    self.device = device
  
    self.bilstm_model = bilstm(dict_sz,hid_sz,device)

  def train(self,head_i,body_i,features,label,n_epochs,reload,batch_size = 128):
    # initalizing SGD optimizer and NLLLoss(Negitive log liklyhood loss)

    optimizer = optim.SGD(self.parameters(), lr = 0.001)
    criterion = nn.NLLLoss()

    if reload:
      # if True, reload the model from saved checkpoint and retrain the model from epoch and batch last stoped training 
      optimizer, start_epoch = self.load_ckp(path+"checkpoint.pt", optimizer)
      print("\nResuming training from epoch {}".format(start_epoch))
    else:
      # else start training from epoch 0
      start_epoch = 0
    
    return

    # calculate number of batch iterations
    n_batches = int(len(head_i)/batch_size)

    # start time
    stime = datetime.now().timestamp()
    
    #start_epoch = n_epochs + 1
    for i in range(start_epoch,n_epochs):

      #shuffle
      r = torch.randperm(head_i.shape[0])
      head_i = head_i[r[:, None]].squeeze(1)
      body_i = body_i[r[:, None]].squeeze(1)
      features = features[r[:, None]].squeeze(1)
      label = label[r[:, None]].squeeze(1)

      # initalize batch_loss to 0.
      batch_loss = 0.

      for b in range(n_batches):

        head_i_batch = head_i[b*batch_size: (b*batch_size) + batch_size]
        body_i_batch = body_i[b*batch_size: (b*batch_size) + batch_size]
        features_batch = features[b*batch_size: (b*batch_size) + batch_size]
        label_batch = label[b*batch_size: (b*batch_size) + batch_size]

        # initialize outputs tensor to save output from the training
        outputs = torch.zeros(batch_size,2, requires_grad=True).to(self.device)

        # zero the gradient
        optimizer.zero_grad()
        
        outputs = self.bilstm_model(head_i_batch,body_i_batch,features_batch)
        
        loss = criterion(outputs,label_batch)

        batch_loss += loss.item()

        # backpropagation
        loss.backward()
        optimizer.step()

        #saving the checkpoint to resume training
      self.save_ckp(i,optimizer,path+'checkpoint.pt')

      # loss for epoch 
      batch_loss = batch_loss/(n_batches)
      # saves loss to file to loss change information is stores if training stops and training is resumed
      self.save_loss(i,batch_loss)

      # show progress to training
      time_escaped = int(datetime.now().timestamp() - stime)
      epoch_passed = i - start_epoch
      if epoch_passed < 1:
        epoch_passed = 1
      time_expected = int((time_escaped/epoch_passed)*(n_epochs - start_epoch))
      time_remaining = time_expected - time_escaped
      print("epoch {} of {}, time escaped = {}s, expected ={}s, remaining {}s, loss = {}"
            .format(i,n_epochs,time_escaped,time_expected,time_remaining,batch_loss))

  def save_loss(self,epoch,loss):
    '''saves loss value to loss.csv'''
    with open(path+'loss.csv', 'a') as file:
      file.write("{},{}\n".format(epoch,loss))
      file.close()

  #Thanks to https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
  # code have been modified as per the need of this model
  def save_ckp(self,epoch,optimizer,checkpoint_dir):
    #save checkpoint to resume training
    checkpoint = {
        'epoch': epoch + 1,
        'bilstm_state_dict': self.bilstm_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_dir)

  def load_ckp(self,checkpoint_fpath, optimizer):
    # reloads the model from the checkpoint
    checkpoint = torch.load(checkpoint_fpath)
    self.bilstm_model.load_state_dict(checkpoint['bilstm_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return ( optimizer, checkpoint['epoch'])

  def predict(self,head_i,body_i,features):
    batch_size = head_i.shape[0]
    output = self.bilstm_model.forward(head_i, body_i, features)
    out = []
    #print(output.shape)
    for i in range(batch_size):
      #print(output[i].topk(1))
      topv, topi = output[i].topk(1)
      #print(topi[0])
      out.append(int(topi[0]))
    return out