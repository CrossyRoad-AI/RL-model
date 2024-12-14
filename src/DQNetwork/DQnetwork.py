import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from constants.constants import *
from utils.singleton import Singleton

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        
        self.conv1 = nn.Conv2d(self.input_dims[0], 32, 3, stride=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1) 
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        conv_out_size = self._get_conv_out(self.input_dims)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(conv_out_size, self.fc1_dims) # its a way of unpacking the list of the elements of the observation space
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("Using device:", self.device)
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = T.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, state):
        state = state.to(self.device,dtype=T.float32)
        if len(state.shape) == 3:  # If the input is 3D, add a batch dimension
            state = state.unsqueeze(0)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(state.size(0), -1)

        X = F.relu(self.fc1(x)) # we want to pass the state through the first layer
        X = F.relu(self.fc2(X)) # pass the output of the first layer through the second layer
        actions = self.fc3(X) # pass the output of the second layer through the third layer
                            # we don't use an activation function here because we want the raw Q values
        return actions

class Agent(metaclass = Singleton):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)] # list of all possible actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_ctr = 0 #  to keep track of the position of the first available memory for storing the agent's memory
        
        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=512, fc2_dims=512) # the Q network that the agent uses to learn, fc1_dims and fc2_dims are the number of neurons in the first and second hidden layers which are 256 by default
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) # the memory of the states
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)    # the memory of the new states
        
        # what the agent wants to know for the temporal difference update rule is what is the value of current state, the value of the next state, 
        # the reward it recieved and the action it took and to get that you have to pass in a memory of the states that resulted from its actions 
        # because deep q learning is a 1.model free, 2.bootstrapped, 3.off policy learning method

        # 1. model free means that we don't need to know anything about the dynmaics of the environment, how the game works, we're gonna figure that 
        # out by playing the game 

        # 2. boostrapped means that you  are going to construct estimates of action value functions meaning the value of each action given you're 
        # in some state based on earlier estimates (you're using one estimate to update another (you're pulling yourseld up by the bootstraps))

        # 3. off policy means that you have a policy that you use to generate actions which is epsilon greedy meaning that we use the hyperparameter that 
        # we defined (epsilon) to determine the proportion of time that the agent is taking random VS greedy actions and you're going to use that policy 
        # to generate data for updating the purely greedy policy meaning the agent's estimate of the maximum action value function

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32) # the memory of the actions    
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) # the memory of the rewards
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)


    def store_transition(self, state, action, reward, state_, done):
        index= self.mem_ctr % self.mem_size # using the modulus has the property that this will wrap around so once we go from memory 0 up to 99999 
                                            # that 100,000 memory we store will go all the way back in position 0 so we rewrite the agents earliest 
                                            # memories with new memories and thats because the memory is finite
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_ctr +=1  


    def choose_action(self, observation):
        if np.random.random() > self.epsilon: # we want to take the best known action
            state = T.tensor([observation]).to(self.Q_eval.device) # we need these brackets around the observation because of the way the DNN is setup
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item() # we want argmax to tell us the integer that corresponds to the maximal action for that set or state and 
                                            # you have dereference it with ".items()" because it returns a tensor  and we're gonna be passing this back 
                                            # to our environement which only takes integers or numpy arrays as input
        else: # we want to take a random action
            action = np.random.choice(self.action_space)
        
        return action
    

    
    def learn(self): 
        # in the beginning we have a memory filled up with zeros and we can't learn anything from zeros so how do we deal with that?
        # 1. you can let the agent play a bunch of games randomly until it fills up the whole entirety of its memory and then you can go ahead and start learning
        # (you're not selecting actions intelligently you're just doing it at random)
        # 2. another possibility is to start learning as soon as you filled up the batch size of memory
        if self.mem_ctr < self.batch_size:
            return # we're gonna call the learn function every iteration of our game loop and if we have not filled up the batch size of our memory you just go 
                # ahead and return, don't bother learning  

        # the first thing that we're going to do in the event that we are going to try to learn is 0 the gradient on our optimizer      
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_ctr, self.mem_size) # we calculate the position of the maximum memory because we want to select a subset of our memories but we only 
                                                # want to select up to the last filled memory and so we want to take the minimum of mem_ctr or mem_size 
        batch = np.random.choice(max_mem,self.batch_size, replace=False) # we want replace=False because we don't want to keep selecting the same memories more than 
                                                                        # once (it's a problem in the case that we've stored a small amount of memories)
        batch_index = np.arange(self.batch_size, dtype=np.int32) # to perform the proper index silicing (?)


        # we're converting the numpy array subset of our agent's memory into a pytorch tensor
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device) 
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device) 

        action_batch = self.action_memory[batch]

        # now we have to perform the feed forwards through our DNN to get the relevant parameters for our loss function 
        # we want to be moving the agent's estimate for the value of the current state towards the maximal value for the next state
        # in other words we want to tilt it towards selecting maximal actions

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] # we have to do the dereferencing (array slicing) because we want to get the values of the 
                                                                            # actions we actually took those are the only ones we're interested in, you can't really update 
                                                                            # the values of actions you didn't take because you didn't sample those 
        q_next = self.Q_eval.forward(new_state_batch) # we want the agent's estimate for the next state as well 
        q_next[terminal_batch] = 0.0

        # maximum value of the next state and that is the purely greedy action and it's what we want for updating our loss function
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] # this is where we want to update our estimates towards (the zeroth element because the max function 
                                                                    # returns the value as well as the index and we only want the value)

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward() # back propagation
        
        # Gradient clipping
        for param in self.Q_eval.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)  # Clip gradients between -1 and 1

        self.Q_eval.optimizer.step()

        # the next thing we have to handle is the epsilon decrement so each time we learn we're gonna decrease epsilon by 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min