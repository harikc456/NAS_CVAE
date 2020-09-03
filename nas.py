import torch
import numpy as np
import torch.nn as nn
from vae_network import *
import torch.nn.functional as F
from scipy.special import softmax

class NAS():
    
    def __init__(self, input_dim, c_dim, size, layer_upper_bound = 5, neuron_upper_bound = 1000,
                 neuron_lower_bound = 50, latent_upper_bound = 100, latent_lower_bound = 20, 
                 trainloader = None, testloader = None):
        
        self.input_dim = input_dim
        self.cond_dim = c_dim
        self.size = size
        self.layer_upper_bound = layer_upper_bound
        self.neuron_upper_bound = neuron_upper_bound
        self.neuron_lower_bound = neuron_lower_bound
        self.latent_upper_bound = latent_upper_bound - c_dim
        self.latent_lower_bound = latent_lower_bound
        self.encoder_layer_population = np.random.randint(0, 2, (size, self.layer_upper_bound))
        self.decoder_layer_population = np.random.randint(0, 2, (size, self.layer_upper_bound))
        self.encoder_neuron_population = np.random.randint(self.neuron_lower_bound, self.neuron_upper_bound, (size, self.layer_upper_bound))
        self.decoder_neuron_population = np.random.randint(self.neuron_lower_bound, self.neuron_upper_bound, (size, self.layer_upper_bound))
        self.latent_population = np.random.randint(self.latent_lower_bound, self.latent_upper_bound, size)
        self.trainloader = trainloader
        self.testloader = testloader
        self.fitness = []
        self.best_model = None
        
    def check_layer_constraints(self, candidate):	
        if min(candidate) < 0 or max(candidate) > 1:
            return False
        return True
        
    def check_neuron_constraints(self, candidate):
        if min(candidate) >= self.neuron_lower_bound and max(candidate) <= self.neuron_upper_bound:
            return True
        return False
    
    def check_latent_constraints(self, candidate):
        if candidate >= self.latent_lower_bound and candidate <= self.latent_upper_bound:
            return True
        return False
    
    def bit_flipping(self, child):
        mutated_child = []
        for i in child:
            if np.random.uniform(0, 1) > 0.5:
                mutated_child.append(np.invert(i))
            else:
                mutated_child.append(i)
        return mutated_child
            
    def creep_mutation(self, child):
        creep = np.random.randint(-3, 3)
        return child + creep
    
    def uniform_crossover(self, parent_1, parent_2):
        mask = np.random.uniform(0, 1, len(parent_1)) > 0.5
        child_1 = (mask * parent_1) + (~mask * parent_2)
        child_2 = (mask * parent_2) + (~mask * parent_1)
        return child_1, child_2
    
    def check_fitness(self, model, epochs = 2, verbose = False):
        
        ## Loss function for VAE
        def loss_function(recon_x, x, mu, logvar):
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD
        
        optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader):
                x, y = data
                #x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                recon, mu, logvar = model(x, y)
                loss = loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print("Epoch {}, Loss {}".format(epoch, running_loss/len(self.trainloader)))
                
        test_loss = 0
        model.eval()
        for i, data in enumerate(self.testloader):
            x, y = data
            #x, y = x.cuda(), y.cuda()
            recon, mu, logvar = model(x, y)
            loss = loss_function(recon, x, mu, logvar)
            test_loss += loss.item()
        
        return 1/(test_loss/len(self.testloader))
        
        
    def start(self, n_generations = 250, n_mating = 1, cr_prob = 1.0, mut_prob = 1.0, verbose = True):
        
        fitness_trail = []
        ## fitness calculation for initial candidates
        for i in range(self.size):
            temp_model = Network(self.input_dim, self.encoder_layer_population[i], self.decoder_layer_population[i],
                                self.encoder_neuron_population[i], self.decoder_neuron_population[i], self.latent_population[i],
                                self.cond_dim)
            self.fitness.append(self.check_fitness(temp_model))
        
        ## Begining the genetic algorithm
        for gen in range(n_generations):
            for mate in range(n_mating):
                index_1 = np.random.choice([*range(self.size)], p = softmax(self.fitness))
                index_2 = np.random.choice([*range(self.size)], p = softmax(self.fitness))
                while index_1 == index_2:
                    index_2 = np.random.choice([*range(self.size)], p = softmax(self.fitness))
                    
                ## Loading the mating pool
                encoder_parent_1 = self.encoder_layer_population[index_1]
                encoder_parent_2 = self.encoder_layer_population[index_2]
                
                decoder_parent_1 = self.decoder_layer_population[index_1]
                decoder_parent_2 = self.decoder_layer_population[index_2]

                encoder_neuron_parent_1 = self.encoder_neuron_population[index_1]
                encoder_neuron_parent_2 = self.encoder_neuron_population[index_2]
                
                decoder_neuron_parent_1 = self.decoder_neuron_population[index_1]
                decoder_neuron_parent_2 = self.decoder_neuron_population[index_2]
                
                latent_dim_parent_1 = self.latent_population[index_1]
                latent_dim_parent_2 = self.latent_population[index_2]
                
                ## Performing crossover and mutation
                cr_p = np.random.uniform(0,1) 
                mut_p = np.random.uniform(0,1)
                if cr_p < cr_prob:
                    encoder_child_1, encoder_child_2 = self.uniform_crossover(encoder_parent_1, encoder_parent_2)
                    decoder_child_1, decoder_child_2 = self.uniform_crossover(decoder_parent_1, decoder_parent_2)
                    
                    encoder_neuron_child_1, encoder_neuron_child_2 = self.uniform_crossover(encoder_neuron_parent_1, encoder_neuron_parent_2)
                    decoder_neuron_child_1, decoder_neuron_child_2 = self.uniform_crossover(decoder_neuron_parent_1, decoder_neuron_parent_2)
                    
                    if mut_p < mut_prob:
                        encoder_child_1 = self.bit_flipping(encoder_child_1)
                        encoder_child_2 = self.bit_flipping(encoder_child_2)
                        
                        decoder_child_1 = self.bit_flipping(decoder_child_1)
                        decoder_child_2 = self.bit_flipping(decoder_child_2)
                        
                        encoder_neuron_child_1 = self.creep_mutation(encoder_neuron_child_1)
                        encoder_neuron_child_2 = self.creep_mutation(encoder_neuron_child_2)
                        
                        decoder_neuron_child_1 = self.creep_mutation(decoder_neuron_child_1)
                        decoder_neuron_child_2 = self.creep_mutation(decoder_neuron_child_2)
                        
                if mut_p < mut_prob:
                        latent_dim_child_1 = self.creep_mutation(latent_dim_parent_1)
                        latent_dim_child_2 = self.creep_mutation(latent_dim_parent_2)
                        
                ## Check whether the new generation satisfies the constraints
                
                if not self.check_layer_constraints(encoder_child_1):
                    encoder_child_1 = encoder_parent_1
                    
                if not self.check_layer_constraints(encoder_child_2):
                    encoder_child_2 = encoder_parent_2
                
                if not self.check_neuron_constraints(encoder_neuron_child_1):
                    encoder_neuron_child_1 = encoder_neuron_parent_1
                    
                if not self.check_neuron_constraints(encoder_neuron_child_2):
                    encoder_neuron_child_2 = encoder_neuron_parent_2
                    
                if not self.check_layer_constraints(decoder_child_1):
                    decoder_child_1 = decoder_parent_1
                    
                if not self.check_layer_constraints(encoder_child_2):
                    decoder_child_2 = decoder_parent_2
                    
                if not self.check_neuron_constraints(decoder_neuron_child_1):
                    decoder_neuron_child_1 = decoder_neuron_parent_1
                    
                if not self.check_neuron_constraints(decoder_neuron_child_2):
                    decoder_neuron_child_2 = decoder_neuron_parent_2
                    
                if not self.check_latent_constraints(latent_dim_child_1):
                    latent_dim_child_1 = latent_dim_parent_1
                    
                ## Creating and evaluating new model
                
                temp_model = Network(self.input_dim, encoder_child_1, decoder_child_1, encoder_neuron_child_1,
                                     decoder_neuron_child_1, latent_dim_child_1, self.cond_dim)
                child_fitness_1 = self.check_fitness(temp_model)
                
                if child_fitness_1 > min(self.fitness):
                    replace_index = self.fitness.index(min(self.fitness))
                    self.encoder_layer_population[replace_index] = encoder_child_1
                    self.decoder_layer_population[replace_index] = decoder_child_1
                    self.encoder_neuron_population[replace_index] = encoder_neuron_child_1
                    self.decoder_neuron_population[replace_index] = decoder_neuron_child_1
                    self.latent_population[replace_index] = latent_dim_child_1
                    self.fitness[replace_index] = child_fitness_1
                
                temp_model = Network(self.input_dim, encoder_child_2, decoder_child_2, encoder_neuron_child_2, 
                                     decoder_neuron_child_2, latent_dim_child_2, self.cond_dim)
                child_fitness_2 = self.check_fitness(temp_model)
                
                if child_fitness_2 > min(self.fitness):
                    replace_index = self.fitness.index(min(self.fitness))
                    self.encoder_layer_population[replace_index] = encoder_child_2
                    self.decoder_layer_population[replace_index] = decoder_child_2
                    self.encoder_neuron_population[replace_index] = encoder_neuron_child_2
                    self.decoder_neuron_population[replace_index] = decoder_neuron_child_2
                    self.latent_population[replace_index] = latent_dim_child_2
                    self.fitness[replace_index] = child_fitness_2
                    
            if verbose:
                print("Generation {} completed. Best fitness value is {}".format(gen,max(self.fitness)))
                
            fitness_trail.append(max(self.fitness))
            
        ## Save the best model
        best_index = self.fitness.index(max(self.fitness))
        self.best_model = Network(self.input_dim, self.encoder_layer_population[best_index], self.decoder_layer_population[best_index],
                                self.encoder_neuron_population[best_index], self.decoder_neuron_population[best_index], 
                                self.latent_population[best_index], self.cond_dim)
        return fitness_trail