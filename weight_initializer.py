import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Weight():
    
    def __init__(self, model, criterion, testloader, population_size = 10):
        self.model = model
        self.population_size = population_size
        self.dims = self.get_dims()
        self.loss_fn = criterion
        self.testloader = testloader
        self.population = self.initialize_population()
        self.fitness = self.check_fitness()
        self.fitness_probs = softmax(self.fitness)
     
    
    ## Get the dimensions of each weight matrix in the model
    def get_dims(self):
        dimensions = []
        for p in self.model.parameters():
            if p.requires_grad and len(p.size())>1:
                dimensions.append(p.size())
        return dimensions
    
    ## Use the dimesions to generate the initial population 
    def initialize_population(self):
        population = []
        for pop in range(self.population_size):
            layer = []
            for dim in self.dims:
                temp_tensor = torch.randn(dim) * torch.sqrt(torch.tensor(2 / dim[0]))
                layer.append(temp_tensor)
            population.append(layer)
        return population
    
    ## Used to check fitness of the population
    def check_fitness(self):
        fitness_list = []
        for pop_index in range(self.population_size):
            fitness_list.append(self.fitness_of(self.population[pop_index]))
        fitness_list = [p.item() for p in fitness_list]
        return fitness_list
    
    ## Used to check fitness of one candidate
    def fitness_of(self, child):
        tensor_index = 0
        state_dict = self.model.state_dict()
        with torch.no_grad():
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = nn.Parameter(child[tensor_index]) 
                    tensor_index += 1
        self.model.load_state_dict(state_dict)
        loss = 0
        for data,y in self.testloader:
            y_hat = self.model(data, y)
            loss += self.loss_fn(y_hat, y)
        return 1/loss
    
    ## Decide to stop differential evolution or not
    def early_stop(self, trend, num_values = 30):
        fit = np.array(trend[-num_values:])
        key = fit[0]
        print(fit,key)
        print(fit==key)
        if np.sum(fit==key) == len(fit):
            return True
        return False
    
    ## Differential Mutation
    def mutation(self, f = 0.4):
        child = []
        index_1 = np.random.randint(0, self.population_size)
        index_2 = np.random.randint(0, self.population_size)
        index_3 = np.random.randint(0, self.population_size)
        parent_1 = self.population[index_1]
        parent_2 = self.population[index_2]
        parent_3 = self.population[index_3]
        for tensor_index in range(len(parent_1)):
            subparent_1 = parent_1[tensor_index] 
            subparent_2 = parent_2[tensor_index] 
            subparent_3 = parent_2[tensor_index] 
            subchild = subparent_1 + f * (subparent_2 - subparent_3)
            child.append(subchild)
        return child
        
    ## Crossover with a probability
    def crossover(self, parent, child, cr = 0.8):
        child_1 = []
        for tensor_index in range(len(child)):
            subparent_1 = parent[tensor_index]
            subparent_2 = child[tensor_index]
            prob_tensor = torch.rand(subparent_1[tensor_index].size())
            mask_tensor = prob_tensor > cr
            subchild_1 = (subparent_1 * mask_tensor) + (subparent_2 * ~mask_tensor)
            child_1.append(subchild_1)
        return child_1
            
    ## Applying the weights to the model
    def apply_weights(self,pop_index):
        tensor_index = 0
        state_dict = self.model.state_dict()
        with torch.no_grad():
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = nn.Parameter(self.population[pop_index][tensor_index]) 
                    tensor_index += 1
        self.model.load_state_dict(state_dict)
        return self.model
    
    ## Start the differential evolution
    def start(self, n_generations, n_weights = 1, verbose = True, warmup = 50):
        trend = []
        for gen in range(n_generations):
            for c in range(self.population_size):
                parent = self.population[c]
                child = self.mutation()
                child = self.crossover(parent, child)
                child_fitness = self.fitness_of(child)
                if child_fitness > self.fitness[c]:
                    self.population[c] = child
                    self.fitness[c] = child_fitness
            if verbose:
                print("Generation {} Best Fitness {}".format(gen, max(self.fitness)))
            trend.append(max(self.fitness).item())
            if gen > warmup:
                if self.early_stop(trend):
                    print("Early stopping initiated")
                    break
        best_fitness = max(self.fitness)
        self.apply_weights(self.fitness.index(best_fitness))
        return trend