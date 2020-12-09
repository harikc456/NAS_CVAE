#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.special import softmax
from cnas import *
from weight_initializer import *
import matplotlib.pyplot as plt


# In[2]:


batch_size = 64
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST('./mnist/', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,shuffle=True, num_workers=2, batch_size = batch_size)

testset = torchvision.datasets.MNIST('./mnist/', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,shuffle=False, num_workers=2, batch_size = batch_size)


# In[3]:


input_dim = 784
cond_dim = 10


# In[4]:


def loss_function(recon, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# In[5]:


population_size = 5
nas = NAS(input_dim, cond_dim, population_size, trainloader = trainloader, testloader = testloader)


# In[6]:


nas.start(3000)


# In[12]:

best_model = nas.best_model
weight_initialiser = Weight(best_model, loss_function, trainloader, tunable = False)


# In[13]:


n_generations = 1000
ascent = weight_initialiser.start(n_generations)


# In[ ]:


best_weight_loss = []
model = weight_initialiser.model
epochs = 100
running_loss = 0.0
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        x, y = data
        #x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        recon, mu, logvar = model(x, y)
        BCE = F.binary_cross_entropy(recon, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            best_weight_loss.append(running_loss)
            running_loss = 0.0

# In[14]:


PATH = 'nas.pt'
torch.save(nas,PATH)


# In[ ]:


PATH = 'vae.pt'
torch.save(model,PATH)

