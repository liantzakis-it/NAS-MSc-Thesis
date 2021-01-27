import matplotlib.pyplot as plt
import numpy as np
import random

import keras
import tensorflow as tf
from copy import deepcopy
from collections import namedtuple

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
tf.test.gpu_device_name()

import math
from nord.neural_nets import BenchmarkEvaluator, NeuralDescriptor


from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Input, SeparableConv2D, Lambda
from keras.layers.core import Activation, Flatten, Dropout, Reshape


def keras_convnet(input_shape=(5,4,1)):
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

    X = Flatten()(inp)
    X = Dense(units=512, activation='relu', use_bias=False)(X)
    X = Dense(units=256, activation='relu', use_bias=False)(X)
    X = Dense(units=128, activation='relu', use_bias=False)(X)
    X = Dense(units=4, activation='tanh')(X)
    # X = Lambda(lambda X: X * 100)(X)
    
    model = Model(inputs=inp, outputs=X) 
    
    return model


# PREDICTION DQN
prediction_net = keras_convnet()
print(prediction_net.summary())

# DQN runtime parameters
def sum_squared_loss(yTrue, yPred):
    return K.sum(K.square(yTrue - yPred))
loss = sum_squared_loss
optimizer = keras.optimizers.Adam(lr=0.001)

# Compile keras model
prediction_net.compile(loss=loss, optimizer=optimizer)



def agent_step(old_position:np.array, action:int, layer_num:int):
    """
    Take a step somewhere, unless you'll go through the wall!
    
    Inputs:
    old_position -- a numpy array, usually of shape (1,5,4,1), with the 2nd dim representing the num of
                    layers and the 3rd dim representing the possible layer types per layer
    action -- an integer from 0 to 3 representing either Stop, CONV1X1, CONV3X3, MAXPOOL3X3
    layer_num -- an integer from 1 to 7 representing the index of the layer we need to take a step at 
                 (need to subtract one to use as an actual index)
    
    Outputs:
    new_position -- a numpy array, usually of shape (1,5,4,1), each row representing a one hot vector

    """
    actionKeys = {0:"Stop", 1:"CONV1X1", 2:"CONV3X3", 3:"MAXPOOL3X3"}
    new_position = np.copy(old_position)

    if actionKeys[action] == "Stop":
        new_position[0,layer_num-1,0,0] = 1
    else:
        new_position[0,layer_num-1,0,0] = 0
        new_position[0,layer_num-1,action,0] = 1
    
    return new_position



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
    actions_tuple = sample_batch.action
    rewards_tuple = sample_batch.reward
    stacked_next_states = np.stack(sample_batch.next_state)

    return stacked_states, actions_tuple, rewards_tuple, stacked_next_states

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
        return random.sample(self.memory, size_of_sample)





actionKeys = {0:"Stop", 1:"CONV1X1", 2:"CONV3X3", 3:"MAXPOOL3X3"}

# Initial state of the neural network being built. Each row represents a layer. 
# Indices from 0 to 3 represent "No Layer", "CONV1X1", "CONV3X3", "MAXPOOL3X3" respectively
initial_state = np.array([[1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [1, 0, 0, 0],
                          [1, 0, 0, 0]], dtype=np.int32)
initial_state = initial_state.reshape(-1,5,4,1)

# Params
gamma = 0.999
max_reward = 100
min_reward = 0
target_update = 13
max_layers = 5
epsilon = 0.75
epsilon_decay = 0.008
total_time = 0
reward_reduction = 90

num_episodes = 1001
rewardList = []

# experience replay parameters
memory_capacity = 2500
sample_size = 128


evaluator = BenchmarkEvaluator('nasbench', False)

replay_memory = ExperienceReplayMemory(memory_capacity)


for episode in range(num_episodes):
# for episode in range(1):
    # start with an empty network
    curr_pos = np.copy(initial_state)
    n = NeuralDescriptor()
    layer_num = 0
    done = False
    reward = 0

    # add the input layer
    n.add_layer_sequential('input', {}, '0')

    while not done:
        layer_num += 1

        # current state Q-values
        curr_state_qv = prediction_net.predict(x = curr_pos, batch_size = 1)
        curr_state_best_action = np.argmax(curr_state_qv)
        # curr_state_max_q = np.max(curr_state_qv)

        # take action
        action = curr_state_best_action
        if np.random.rand(1) < epsilon:
            action = int(np.random.uniform(low=0, high=4))
        
        next_pos = agent_step(curr_pos, action, layer_num)

        if action != 0:
            # add the respective layer to the desriptor
            # this is done supposing that get_available_ops() returns the 3 layers
            # in this order: [CONV1X1, CONV3X3, MAXPOOL3X3]
            layer_to_add = evaluator.get_available_ops()[np.argmax(next_pos[0][layer_num-1]) - 1]
            n.add_layer_sequential(layer_to_add, {}, str(layer_num))

        # if at terminal state
        if layer_num == max_layers or action == 0:
            done = True
            # add the output layer
            n.add_layer_sequential('output', {}, '9')

            # evaluate the constructed descriptor and get the reward
            reward, time = evaluator.descriptor_evaluate(n)
            if(reward == 0):
                print(n)

            # scale and clip the reward, supposing the returned accuracy is in range [0,1]
            reward = max(min_reward, min(reward*100, max_reward)) - reward_reduction

            rewardList.append(reward)
            total_time += time

        e = Experience(curr_pos, action, reward, next_pos)
        replay_memory.push_into_memory(e)

        curr_pos = next_pos


    if replay_memory.can_provide_sample(sample_size):
        experiences = replay_memory.sample_from_memory(sample_size)
        states, actions, rewards, next_states = break_down_experiences(experiences)
        target_qvs = np.zeros((sample_size, 1, 4), dtype=np.float32)

        # loop through the experiences and re-calculate the target q-values for all of them
        for exp in range(sample_size):
            crt_state = states[exp]
            nxt_state = next_states[exp]
            crt_state_qv = prediction_net.predict(x = crt_state, batch_size = 1)
            nxt_state_qv = prediction_net.predict(x = nxt_state, batch_size = 1)
            target_qv = crt_state_qv
            target_qv[0,actions[exp]] = rewards[exp] + gamma*np.max(nxt_state_qv)
            target_qvs[exp] = target_qv

        # reshape the x,y arrays
        states = states.reshape(sample_size, max_layers, 4, 1)
        target_qvs = target_qvs.reshape(sample_size, 4)

        prediction_net.fit(x = states, y = target_qvs, verbose = 0, batch_size = 64)
 
    if epsilon > 0:
        epsilon -= epsilon_decay
