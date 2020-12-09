import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self, input_dim, encoder_layers, decoder_layers, encoder_neurons, decoder_neurons, latent_dim, cond_dim):
        super(Network, self).__init__()
        self.input_dim = input_dim 
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.encoder = None
        self.mu_fc = None
        self.logvar_fc = None
        self.decoder = None
        self.relu = nn.ReLU()
        self.construct_network()
        
        
    def construct_network(self):
        modules = []
        prev_layer = [self.input_dim + self.cond_dim]
        #prev_layer = [self.input_dim]
        
        ## Construct the encoder
        
        for i in range(len(self.encoder_layers)):
            if self.encoder_layers[i] == 1:
                no_of_neuron = self.encoder_neurons[i]
                modules.append(nn.Linear(prev_layer[-1], no_of_neuron))
                modules.append(self.relu)
                prev_layer.append(no_of_neuron)
        self.encoder = nn.Sequential(*modules)
        self.mu_fc = nn.Linear(prev_layer[-1], self.latent_dim)
        self.logvar_fc = nn.Linear(prev_layer[-1], self.latent_dim)
        
        ## Construct the decoder
        
        modules = []
        prev_layer.append(self.latent_dim + + self.cond_dim)
        for i in range(len(self.decoder_layers)):
            if self.decoder_layers[i] == 1:
                no_of_neuron = self.decoder_neurons[i]
                modules.append(nn.Linear(prev_layer[-1], no_of_neuron))
                modules.append(self.relu)
                prev_layer.append(no_of_neuron)
        modules.append(nn.Linear(prev_layer[-1], self.input_dim))
        self.decoder = nn.Sequential(*modules)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, c):
        z = torch.cat([z,c], dim = 1)
        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x
        
    def forward(self, x, y):
        c = nn.functional.one_hot(y, num_classes = self.cond_dim)
        x = torch.cat([x.view(-1, 784), c.float()], dim = 1)
        x = self.encoder(x)
        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, c)
        return x, mu, logvar