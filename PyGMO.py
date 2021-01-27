import pygmo as pg
from nord.neural_nets import BenchmarkEvaluator, NeuralDescriptor
import numpy as np

evaluator = BenchmarkEvaluator('nasbench', False)

class nas_problem:
    def fitness(self, x):
        '''
        len(x) == 26
        x[:5] ---> the 5 layer types, from 0 through 3 (0 = no layer, 1 = conv1x1, 2 = conv3x3, 3 = maxpool3x3)
        x[5:] ---> all possible connections in sequence: x01, x02, x12, x03, x13, x23, x04, x14, x24, x34, x05, x15, x25, x35, x45, x06, x16, x26, x36, x46, x56.
                   with x01 meaning a connection from the input layer (0) to the 1st hidden layer
                   and x06 meaning a connection from the input layer (0) to the output layer (6)
                   The indices above are a triangular matrix, the top half of the adjacency matrix of all 7 layers. 
                   So, in x[5:], the 1st index is the possible input of layer 1,
                   the next 2 indices are the possible inputs of layer 2, the next 3 indices are the possible indices of layer 3,
                   the next 4 indices are  the possible inputs of layer 4, the next 5 indices are the possible inputs of layer 5
                   and the last 6 indices are the possible inputs of the output layer.
        '''
        # construct the Neural Descriptor
        n = NeuralDescriptor()
        
        # add the input layer
        n.add_layer('input', {}, '0')

        inputs_slice = x[5:]
        start_idx = 0
        end_idx = 0

        for iteration in range(1,6):
            end_idx = start_idx + iteration
            # check for chosen layer type
            if x[iteration-1] != 0:
                type_idx = x[iteration-1]
                layer_to_add = evaluator.get_available_ops()[int(type_idx)-1]
                n.add_layer(layer_to_add, {}, str(iteration))

                curr_slice = inputs_slice[start_idx:end_idx]
                indices = [i for i,j in enumerate(curr_slice) if j]
                for idx in indices:
                    if (str(idx) in n.layers.keys()):
                        n.connect_layers(str(idx), str(iteration))
            # else:
            #     start_idx = end_idx
            #     continue
            start_idx = end_idx

        n.add_layer('output', {}, '6')
        outputs_slice = inputs_slice[-6:]
        output_indices = [i for i,j in enumerate(outputs_slice) if j]
        for idx in output_indices:
            if (str(idx) in n.layers.keys()):
                n.connect_layers(str(idx), '6')

        #print(n)
        # obj func, need to maximize it so a minus is needed since pygmo minimizes it
        obj = 0
        try:
            obj = - (evaluator.descriptor_evaluate(n)[1]) # returns params, reward, time_taken # , acc='test_accuracy'
        except:
            pass

        # constraint to enforce the maximum of 9 connections, so sum(all_input_vars) <= 9
        ci1 = sum(x[5:]) - 9

        # constraint to ensure that there's at least 1 input to the output layer
        # adding x06, x16, x26, x36, x46, x56 and asserting they are >= 1
        ci2 = 1 - sum(x[-6:])

        return [obj,ci1,ci2]
    

    def get_bounds(self):
        return ([0]*5 + [False]*21,[3]*5 + [True]*21)
    
    # Inequality constraints
    def get_nic(self):
        return 2
    
    # Integer dimension
    def get_nix(self):
        return 26



# Unconstrained version of the nas_problem class 
# for the problems that cannot accept constrained formulatation
class nas_problem_unconstrained:
    def fitness(self, x):
        '''
        len(x) == 26
        x[:5] ---> the 5 layer types, from 0 through 3 (0 = no layer, 1 = conv1x1, 2 = conv3x3, 3 = maxpool3x3)
        x[5:] ---> all possible connections in sequence: x01, x02, x12, x03, x13, x23, x04, x14, x24, x34, x05, x15, x25, x35, x45, x06, x16, x26, x36, x46, x56.
                   with x01 meaning a connection from the input layer (0) to the 1st hidden layer
                   and x06 meaning a connection from the input layer (0) to the output layer (6)
                   The indices above are a triangular matrix, the top half of the adjacency matrix of all 7 layers. 
                   So, in x[5:], the 1st index is the possible input of layer 1,
                   the next 2 indices are the possible inputs of layer 2, the next 3 indices are the possible indices of layer 3,
                   the next 4 indices are  the possible inputs of layer 4, the next 5 indices are the possible inputs of layer 5
                   and the last 6 indices are the possible inputs of the output layer.
        '''
        # construct the Neural Descriptor
        n = NeuralDescriptor()
        
        # add the input layer
        n.add_layer('input', {}, '0')

        inputs_slice = x[5:]
        start_idx = 0
        end_idx = 0

        for iteration in range(1,6):
            end_idx = start_idx + iteration
            # check for chosen layer type
            if x[iteration-1] != 0:
                type_idx = x[iteration-1]
                layer_to_add = evaluator.get_available_ops()[int(type_idx)-1]
                n.add_layer(layer_to_add, {}, str(iteration))

                curr_slice = inputs_slice[start_idx:end_idx]
                indices = [i for i,j in enumerate(curr_slice) if j]
                for idx in indices:
                    if (str(idx) in n.layers.keys()):
                        n.connect_layers(str(idx), str(iteration))
            # else:
            #     start_idx = end_idx
            #     continue
            start_idx = end_idx

        n.add_layer('output', {}, '6')
        outputs_slice = inputs_slice[-6:]
        output_indices = [i for i,j in enumerate(outputs_slice) if j]
        for idx in output_indices:
            if (str(idx) in n.layers.keys()):
                n.connect_layers(str(idx), '6')

        #print(n)
        # obj func, need to maximize it so a minus is needed since pygmo minimizes it
        obj = 0
        try:
            obj = - (evaluator.descriptor_evaluate(n)[1]) # returns params, reward, time_taken # , acc='test_accuracy'
        except:
            pass

        return [obj]
    

    def get_bounds(self):
        return ([0]*5 + [False]*21,[3]*5 + [True]*21)
    
    # Integer dimension
    def get_nix(self):
        return 26



# Ant Colony Optimization
prob = pg.problem(nas_problem())
algo = pg.algorithm(pg.gaco(50, 13, 1.0, 1e9, 0.0, 1, 7, 100000, 100000, 0.0, False))

pop = pg.population(prob = nas_problem(), size = 30)
algo.set_verbosity(1)
pop = algo.evolve(pop)

print(pop.champion_f)
print(pop.champion_x)

my_uda = algo.extract(pg.gaco)
my_uda.get_log()


# Particle Swarm Optimization
prob = pg.problem(nas_problem_unconstrained())
algo = pg.algorithm(uda = pg.pso(gen=50, omega=0.7298, eta1=2.05, eta2=2.05, max_vel=0.5, variant=1, neighb_type=1, neighb_param=4, memory=False))

pop = pg.population(prob = nas_problem_unconstrained(), size = 30)
algo.set_verbosity(1)
pop = algo.evolve(pop)

print(pop.champion_f)
print(pop.champion_x)

my_uda = algo.extract(pg.pso)
my_uda.get_log()


# Artificial Bee Colony
prob = pg.problem(nas_problem_unconstrained())
algo = pg.algorithm(uda = pg.bee_colony(gen=50, limit=20))

pop = pg.population(prob = nas_problem_unconstrained(), size = 30)
algo.set_verbosity(1)
pop = algo.evolve(pop)

print(pop.champion_f)
print(pop.champion_x)

my_uda = algo.extract(pg.bee_colony)
my_uda.get_log()