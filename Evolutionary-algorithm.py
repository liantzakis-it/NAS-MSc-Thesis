import matplotlib.pyplot as plt
import numpy as np
import random
import time

import tensorflow as tf
from copy import deepcopy
from collections import namedtuple
from operator import attrgetter

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Tensorflow   :', tf.__version__)
tf.test.gpu_device_name()

import math
from nord.neural_nets import BenchmarkEvaluator, NeuralDescriptor

evaluator = BenchmarkEvaluator('nasbench', False)

def mutate(init_d, add_node_rate, add_connection_rate):

    layer_no = len(init_d.layers) - 1 # supposing that 2 layers always exist (input and output)
    d = deepcopy(init_d)
    prev_d = deepcopy(d)

    r = np.random.uniform()

    # Add a node
    if r < add_node_rate:
        # Choose an origin node at random
        origin = np.random.choice(list(d.connections.keys()))
        # If there are no outgoing connections
        # (for example if the output node is chosen), resample a node
        while len(d.connections[origin]) < 1:
            origin = np.random.choice(list(d.connections.keys()))
        # Choose an outgoing connection (destination node) at random
        destination = np.random.choice(d.connections[origin])

        # Choose one of the available operations at random (1x1 convolution, 3x3 convolution, 3x3 max pooling)
        new_node = np.random.choice(evaluator.get_available_ops())
        # Add the layer and connect it to the origin and destination
        d.add_layer(new_node, {}, str(layer_no))
        d.connect_layers(origin, str(layer_no))
        d.connect_layers(str(layer_no), destination)
        # Remove the original connection
        d.disconnect_layers(origin, destination)  # TODO: Crashing here
        # -- End of add a node
    # Add a connection
    elif r < add_node_rate + add_connection_rate:
        # Choose an origin node at random
        origin = np.random.choice(list(d.connections.keys()))
        # Choose a destination node at random
        destination = np.random.choice(list(d.connections.keys()))
        # Connect the layers
        d.connect_layers(origin, destination)
        # -- End of add a connection

    return d

NeuralArchitecture = namedtuple(
    'NeuralArchitecture',
    ('arch', 'reward')
)

class PopulationMemory():
    def __init__(self):
        self.history = []
        self.population = []
    
    def sample_from_population(self, size_of_sample):
        return random.sample(self.population, size_of_sample)
    
    def get_best_parent(self, sample):
        return sorted(sample, key=attrgetter('reward'), reverse = True)[0]
    
    def save_architecture(self, architecture_entry):
        self.population.append(architecture_entry)
        self.history.append(architecture_entry)
    
    def discard_the_oldest(self):
        self.population.pop(0)

    def get_population_size(self):
        return len(self.population)
    
    def get_history_size(self):
        return len(self.history)




# PARAMETERS
add_node_rate = 0.05
add_connection_rate = 0.03
population_size = 100
sample_size = 25
evolve_cycles = 500
init_rounds = 2
rewardList = []

# initializations
population_memory = PopulationMemory()


while population_memory.get_population_size() < population_size:
    n = NeuralDescriptor()
    n.add_layer_sequential('input', {}, '0')
    n.add_layer_sequential('output', {}, '6')

    # mutate for a certain amount of times e.g. 4
    # create a random architecture
    for i in range(init_rounds):
        n = mutate(n, add_node_rate, add_connection_rate)

    # evaluate it
    reward, time_taken = evaluator.descriptor_evaluate(n)
    rewardList.append(reward*100)

    architecture = NeuralArchitecture(n, reward*100)

    population_memory.save_architecture(architecture)

while population_memory.get_history_size() < evolve_cycles:
    sample = population_memory.sample_from_population(sample_size)
    parent_set = population_memory.get_best_parent(sample)
    parent_arch = parent_set.arch

    child_arch = mutate(parent_arch, add_node_rate, add_connection_rate)

    reward, time_taken = evaluator.descriptor_evaluate(child_arch)
    rewardList.append(reward*100)

    child_architecture = NeuralArchitecture(child_arch, reward*100)

    population_memory.save_architecture(child_architecture)
    population_memory.discard_the_oldest()