import matplotlib.pyplot as plt
import numpy as np
import random
import time

import keras
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from copy import deepcopy
from collections import namedtuple
from operator import attrgetter

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
tf.test.gpu_device_name()

import math
from nord.neural_nets import BenchmarkEvaluator, NeuralDescriptor

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, SeparableConv2D, Lambda, LSTM, GRU, Attention, Embedding
from tensorflow.keras.layers import Activation, Flatten, Dropout, Reshape


def keras_convnet(input_shape=(None,12)): # was (None,12 for the LSTM)
    """
    The Keras Convolutional Network that parses the incoming pixel data.
    Consists of 3 Convolutional Layers, and 2 Dense Layers, plus 1 softmax output layer
    Note that we use SeparableConv2D layers ala MobileNets for memory efficiency
    
    Inputs:
    input_shape -- shape of the input tensor, (height, width, channels)
    
    Outputs:
    model -- keras.models.Model instance
    
    """
    inp = Input(shape=input_shape)

    H = LSTM(units=128)(inp)
    X = Dense(units = 4, activation = 'tanh', name='action')(H) # [1,0,0,0]
    Y = Dense(units = 7, activation = 'tanh', name='input1')(H) # [1,0,0,0,0,0,0]
    Z = Dense(units = 7, activation = 'tanh', name='input2')(H) # [1,0,0,0,0,0,0]

    model = Model(inputs=inp, outputs=[X,Y,Z]) 
    
    return model



# PREDICTION DQN
prediction_net = keras_convnet()
print(prediction_net.summary())

# DQN runtime parameters
def sum_squared_loss(yTrue, yPred):
    return K.sum(K.square(yTrue - yPred))
loss = sum_squared_loss
optimizer = tf.keras.optimizers.Adam(lr=0.001)
# optimizer = tf.keras.optimizers.Adamax(lr=0.001)

# Compile keras model
prediction_net.compile(loss=loss, optimizer=optimizer, loss_weights={'action':1, 'input1':0.01, 'input2':0.01})



def agent_step(old_position:np.array, action:int, layer_num:int, first_input:int, second_input:int, num_actions:int, total_edges:int):
    """
    Possible 'action' values: 0,1,2,3 = no_layer, conv1x1, conv3x3, maxpool3x3
    If 'action' == -1 then we are at the output layer

    first_input & second_input: the INDEX of the layers to be the first and second inputs

    layer_num: the number of the layer we are currently working on

    num_actions: the number of possible actions (e.g. if possible actions are 0,1,2,3 then num_actions = 4)

    If first and second input layer are equal, and they are either 'input' or 'no_input',
    one of them gets 'input' assigned and one of them gets 'no_input'

    If they are equal but neither 'input' or 'no_input', one of them keeps its value
    and the other one gets 'no_input'
    
    """

    max_edges = 9
    edges_connected = 0
    new_position = np.copy(old_position)

    # HANDLE THE INPUT LAYERS
    # if the inputs are invalid, set them to 'no_input'
    if first_input >= layer_num:
        first_input = 0
    if second_input >= layer_num:
        second_input = 0

    if first_input == second_input:
        if first_input == 0 or first_input == 6:
            first_input = 0
            seconnd_input = 6
            edges_connected += 1
        elif (total_edges / layer_num) < 1.5:
            first_input = 0
            edges_connected += 2
        else:
            first_input = 6
            edges_connected += 1
    else:
        if first_input != 6:
            edges_connected += 1
        if second_input != 6:
            edges_connected += 1
        
    if action == 0:
        if layer_num <= 5:
            # new_position[0,layer_num-1,0:num_actions] = [1,0,0,0,0]
            new_position[0,layer_num-1,0] = 1
        new_position[0,new_position.shape[1]-1,0] = 0
        new_position[0,new_position.shape[1]-1,num_actions] = 1
        new_position[0,new_position.shape[1]-1,num_actions+first_input] = 1
        new_position[0,new_position.shape[1]-1,num_actions+second_input] = 1
    else:
        # last index 0 for the flattened and dense layer versions, same above whenever applicable
        new_position[0,layer_num-1,0] = 0
        new_position[0,layer_num-1,action] = 1
        new_position[0,layer_num-1,num_actions+first_input] = 1
        new_position[0,layer_num-1,num_actions+second_input] = 1
    


    
    return new_position, edges_connected

# EXPERIENCE REPLAY

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'reward', 'next_state')
)

def break_down_experiences(experiences):
    # the "experiences" param will be the batch of samples returned from "sample_from_memory" function
    sample_batch = Experience(*zip(*experiences))

    # part where states, rewards, actions, next_states are broken down into separate arrays
    stacked_states = np.stack(sample_batch.state)
    stacked_actions = np.stack(sample_batch.action)
    rewards_tuple = sample_batch.reward
    stacked_next_states = np.stack(sample_batch.next_state)

    return stacked_states, stacked_actions, rewards_tuple, stacked_next_states

class ExperienceReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def can_provide_sample(self, size_of_sample:int):
        return len(self.memory) >= size_of_sample

    def push_into_memory(self, experience_entry):
        if len(self.memory) < self.capacity:
            self.memory.append(experience_entry)
        else:
            self.memory[self.push_count % self.capacity] = experience_entry
        self.push_count += 1

    def sample_from_memory(self, size_of_sample):
        # sorted_rewards_list = sorted(self.memory, key=attrgetter('reward'), reverse=True)
        # return sorted_rewards_list[0:size_of_sample]
        return random.sample(self.memory, size_of_sample)

# Params
gamma = 0.999
max_reward = 100
min_reward = 0
target_update = 13
epsilon = 0.75
epsilon_decay = 0.001
total_time = 0
reward_reduction = 90
layer_num = 0
epochs_to_train = 1

num_episodes = 2001
rewardList = []

# experience replay parameters
memory_capacity = 15000
sample_size = 1024 # possibly increase this

# NASBench restrictions and more
num_actions = 5
max_layers = 5  # setting it to 5, excluding the input and output layers that add up to 7
max_edges = 9
possible_inputs = 7  # 'input' layer, 5 hidden layers and 'no_input'
edge_termination_threshold = 7 # at how many edges does the building stop and add the output layer with 1 or 2 inputs

actionKeys = {0:"Stop", 1:"CONV1X1", 2:"CONV3X3", 3:"MAXPOOL3X3"}

# Initial state of the neural network being built. Each row represents a layer. 
# Indices from 0 to 3 represent "No Layer", "CONV1X1", "CONV3X3", "MAXPOOL3X3" respectively. Index 4 is the output layer
# Indices from 5 to 11 represent the inputs for each hidden layer, from 'input', to the 5 hidden layer, to 'no_input', 7 in total
# The last row represents the output layer
# shape: (6,12)
initial_state = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

#reshape to -1,6,12 for the LSTM
# initial_state = initial_state.reshape(-1, initial_state.shape[0], initial_state.shape[1])
initial_state = initial_state.reshape(-1, max_layers + 1, num_actions + possible_inputs) #adding +1 to the hidden layers to count for the last row / output layer
# initial_state = initial_state.reshape(-1,6,12,1)

evaluator = BenchmarkEvaluator('nasbench', False)

replay_memory = ExperienceReplayMemory(memory_capacity)


for episode in range(num_episodes):
# for episode in range():
    # start with an empty network
    curr_pos = np.copy(initial_state)
    n = NeuralDescriptor()
    layer_num = 0
    total_edges = 0
    done = False
    reward = 0

    # add the input layer
    n.add_layer('input', {}, '0')

    while not done:
        layer_num += 1

        # current state Q-values
        curr_state_qv = prediction_net.predict(x = curr_pos, batch_size = 1)
        curr_state_best_action = np.argmax(curr_state_qv[0][0])
        # taking argmax up until the previous layer, and the no_input
        indices = [x for x in range(layer_num)]
        curr_state_best_input1 = np.argmax(curr_state_qv[1][0][[*indices,-1]]) # [[*indices,-1]]
        curr_state_best_input2 = np.argmax(curr_state_qv[2][0][[*indices,-1]]) # [[*indices,-1]]
        if curr_state_qv[1][0][6] > curr_state_qv[1][0][curr_state_best_input1]:
            curr_state_best_input1 = 6
        if curr_state_qv[2][0][6] > curr_state_qv[2][0][curr_state_best_input2]:
            curr_state_best_input2 = 6

        # take action
        action = curr_state_best_action
        inp1 = curr_state_best_input1
        inp2 = curr_state_best_input2
        if np.random.rand(1) < epsilon:
            action = int(np.random.uniform(low=0, high=num_actions-1))
            inp1 = int(np.random.uniform(low=0, high=possible_inputs))
            inp2 = int(np.random.uniform(low=0, high=possible_inputs))
        
        # check here if layer_num is > 5 or total edges >= 7 and make the action zero to terminate the process
        if layer_num > 5 or total_edges >= edge_termination_threshold:
            action = 0

        next_pos, edges_conn = agent_step(curr_pos, action, layer_num, inp1, inp2, num_actions, total_edges)
        total_edges += edges_conn

        if action != 0:
            # add the respective layer to the desriptor
            # this is done supposing that get_available_ops() returns the 3 layers
            # in this order: [CONV1X1, CONV3X3, MAXPOOL3X3]
            layer_to_add = evaluator.get_available_ops()[np.argmax(next_pos[0][layer_num-1]) - 1]
            the_inputs = np.where(next_pos[0,layer_num-1,num_actions:] == 1)
            if len(the_inputs[0]) > 0:
                inp1 = the_inputs[0][0]
            else:
                inp1 = 6
            if len(the_inputs[0]) > 1:
                inp2 = the_inputs[0][1]
            else:
                inp2 = 6
            n.add_layer(layer_to_add, {}, str(layer_num))
            if inp1 < 6:
                n.connect_layers(str(inp1), str(layer_num))
            if inp2 < 6:
                n.connect_layers(str(inp2), str(layer_num))

        if action == 0:
            done = True

            # add the output layer
            n.add_layer('output', {}, '6')

            # connect to the inputs
            the_inputs = np.where(next_pos[0,next_pos.shape[1]-1,num_actions:] == 1)
            if len(the_inputs[0]) > 0:
                inp1 = the_inputs[0][0]
            else:
                inp1 = 6
            if len(the_inputs[0]) > 1:
                inp2 = the_inputs[0][1]
            else:
                inp2 = 6
            if inp1 < 6:
                n.connect_layers(str(inp1), '6')
            if inp2 < 6:
                n.connect_layers(str(inp2), '6')

            # evaluate the constructed descriptor and get the reward
            params, orig_reward, time_taken = evaluator.descriptor_evaluate(n)

            # scale and clip the reward, supposing the returned accuracy is in range [0,1]
            # reward = np.exp(reward) * 100
            reward = max(min_reward, min(orig_reward*100, max_reward)) - reward_reduction

            rewardList.append(reward)
            total_time += time_taken

        # action is now a set of 3 things: layer_type, input1, input2
        if reward != -90:
            action_inputs = np.array([action, inp1, inp2], dtype=np.int32)
            e = Experience(curr_pos, action_inputs, reward, next_pos)
            replay_memory.push_into_memory(e)

        curr_pos = next_pos


    if replay_memory.can_provide_sample(sample_size):
        experiences = replay_memory.sample_from_memory(sample_size)
        # states dim: (sample_size, 1, 6, 12)
        # actions dim: (sample_size, 3), 3 for layer_time, inp1, inp2
        states, actions, rewards, next_states = break_down_experiences(experiences)
        # target_qvs = []
        target_actions = np.zeros((sample_size, 1, num_actions-1), dtype=np.float32)
        target_inputs1 = np.zeros((sample_size, 1, possible_inputs), dtype=np.float32)
        target_inputs2 = np.zeros((sample_size, 1, possible_inputs), dtype=np.float32)

        target_qvs = {'action':target_actions, 'input1':target_inputs1, 'input2':target_inputs2}

        states = states.reshape(sample_size, max_layers+1, num_actions + possible_inputs)
        next_states = next_states.reshape(sample_size, max_layers+1, num_actions + possible_inputs)

        crt_states_qv = prediction_net.predict(x = states)
        nxt_states_qv = prediction_net.predict(x = next_states)

        # loop through the experiences and re-calculate the target q-values for all of them
        for exp in range(sample_size):
            crt_state = states[exp]
            nxt_state_qv = [nxt_states_qv[0][exp], nxt_states_qv[1][exp], nxt_states_qv[1][exp]]
            target_qv = [crt_states_qv[0][exp], crt_states_qv[1][exp], crt_states_qv[1][exp]]
            target_qv[0][actions[exp,0]] = rewards[exp] + gamma*np.max(nxt_state_qv[0][0])
            target_qv[1][actions[exp,1]] = rewards[exp] + gamma*np.max(nxt_state_qv[1][0])
            target_qv[2][actions[exp,2]] = rewards[exp] + gamma*np.max(nxt_state_qv[2][0])
            target_qvs['action'][exp] = target_qv[0][0]
            target_qvs['input1'][exp] = target_qv[1][0]
            target_qvs['input2'][exp] = target_qv[2][0]

        # reshape the x,y arrays
        # states = states.reshape(sample_size, max_layers+1, num_actions + possible_inputs)
        # reshape all the target qvs, for actions, input1 and input2
        target_qvs['action'] = target_qvs['action'].reshape(sample_size, num_actions-1)
        target_qvs['input1'] = target_qvs['input1'].reshape(sample_size, possible_inputs)
        target_qvs['input2'] = target_qvs['input2'].reshape(sample_size, possible_inputs)

        prediction_net.fit(x = states, y = target_qvs, verbose = 0, batch_size = 64, epochs = 1)
 
        if epsilon > 0:
            epsilon -= epsilon_decay