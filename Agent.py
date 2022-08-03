import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
from collections import namedtuple

# this file implements the agent (MEC server)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class input_layer(nn.Module):
    '''
    this is a self defined laer, used to transfer the user request state, as shown in the paper.
    '''
    def __init__(self, user_num, task_num):
        super(input_layer, self).__init__()
        self.user_num = user_num
        self.task_num =task_num
        self.w = nn.Parameter(torch.randn(task_num, user_num))
        self.b = nn.Parameter(torch.randn(task_num))

    def forward(self, x):
        batch_sz = x.shape[0]

        First_opt = np.zeros([batch_sz, self.task_num])
        for batch_id in range(batch_sz):
            sample = x[batch_id]
            for user_ind in range(self.user_num):
                index = int(sample[user_ind] - 1)
                if index >=0:
                    First_opt[batch_id, index] += self.w[index, user_ind]

            for file_ind in range(self.task_num):
                First_opt[batch_id, file_ind] += self.b[file_ind]

        tensor_opt = torch.from_numpy(First_opt).type(torch.FloatTensor).to(device)

        return tensor_opt

class FullNet(nn.Module):
    '''
    this class define the DNN used in the DDQN of our paper
    '''
    def __init__(self, state_num, n_mid1, n_mid2, n_mid3, n_mid4, n_mid5, task_num):
        super(FullNet, self).__init__()
        self.fc1 = input_layer(state_num, task_num)
        self.fc2 = nn.Linear(task_num, n_mid1)
        self.fc3 = nn.Linear(n_mid1, n_mid2)
        self.fc4 = nn.Linear(n_mid2, n_mid3)
        self.fc5 = nn.Linear(n_mid3, n_mid4)
        self.fc6 = nn.Linear(n_mid4, n_mid5)
        self.fc7 = nn.Linear(n_mid5, task_num)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        output = self.fc7(h6)
        return output

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:
    '''
    memory, store historical data for experience replay
    '''
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        '''save the transition = (state, action, state_next, reward)'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1)%self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 32
CAPACITY = 10000
class BSAgentBrain:
    '''
    this class implements the agent brain
    '''
    def __init__(self, state_num, action_num, MEC_C, File_num, D_f, learning_rate=0.0001, GAMMA=0.9):
        self.state_num = state_num
        self.action_num = action_num
        self.memory = ReplayMemory(CAPACITY)
        self.MEC_C = MEC_C
        self.File_num = File_num
        self.Df = D_f
        self.GAMMA = GAMMA
        n_in, n_mid1, n_mid2, n_mid3, n_mid4, n_mid5, n_out = state_num, 512, 512, 256, 256, 128, action_num
        self.main_q_network = FullNet(n_in, n_mid1, n_mid2, n_mid3, n_mid4, n_mid5, n_out).to(device)
        self.target_q_network = FullNet(n_in, n_mid1, n_mid2, n_mid3, n_mid4, n_mid5, n_out).to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=learning_rate)
        
    def action_selection(self, last_layer_out):
        '''
        this funtion implements the optimal caching action based on the output of the first layer of TLA (TLA is shown in paper)
        '''
        capacity = int(self.MEC_C / (10 ** 8))
        Df = self.Df / (10 ** 8)
        last_out = torch.squeeze(last_layer_out)
        file_num = self.File_num
        caching_vector = np.zeros(file_num)
        W_r = np.zeros((file_num, capacity+1))
        W_value = np.zeros((file_num, capacity+1))
        for f in range(file_num):
            if f < file_num-1:
                for q in range(capacity+1):
                    if f == 0:
                        if q < Df[f]:
                            W_r[f, q] = 0
                            W_value[f, q] = 0
                        else:
                            W_r[f, q] = 1
                            W_value[f, q] = last_out[f]
                    else:
                        if q < Df[f]:
                            W_r[f, q] = 0
                            W_value[f, q] = W_value[f-1, q]
                        else:
                            dim2_ind = int(q-Df[f])
                            caching_v = last_out[f] + W_value[f-1, dim2_ind]
                            if caching_v > W_value[f-1, q]:
                                W_r[f, q] = 1
                                W_value[f, q] = caching_v
                            else:
                                W_r[f, q] = 0
                                W_value[f, q] = W_value[f-1, q]
            else:
                dim2_ind = int(capacity-Df[f])
                caching_v = last_out[f] + W_value[f-1, dim2_ind]
                if caching_v > W_value[f-1, capacity]:
                    W_r[f, capacity] = 1
                    W_value[f, capacity] = caching_v
                else:
                    W_r[f, capacity] = 0
                    W_value[f, capacity] = W_value[f-1, capacity]
        caching_vector[file_num-1] = W_r[file_num-1, capacity]
        temp_L = caching_vector[file_num-1] * Df[file_num-1]
        temp_L = int(temp_L)
        posi_index = range(file_num-1)
        inver_index = sorted(posi_index, reverse=True)
        for index in inver_index:
            dim2_ind = int(capacity-temp_L)
            caching_vector[index] = W_r[index, dim2_ind]
            temp_L += caching_vector[index] * Df[index]
            temp_L = int(temp_L)

        caching_vector = torch.from_numpy(caching_vector).type(torch.FloatTensor)
        caching_vector = torch.unsqueeze(caching_vector, 0)
        caching_action = caching_vector
        return caching_action

    def decide_action(self, state, training=True):
        '''
        this function implements the action selection, use \epsilon-greedy policy
        '''
        if np.random.uniform(0, 1) < 0.5 and training == True:
            action = np.zeros(self.File_num)
            shuffle_index = [i for i in range(self.File_num)]
            np.random.shuffle(shuffle_index)
            residual_C = self.MEC_C
            for ind in range(self.File_num):
                file_ind = shuffle_index[ind]
                if residual_C > 0 and self.Df[file_ind] < residual_C:
                    action[file_ind] = 1
                    residual_C = residual_C - self.Df[file_ind]
            action = torch.from_numpy(action).type(torch.FloatTensor)
            action = torch.unsqueeze(action, 0)
        else:
            self.main_q_network.eval()
            with torch.no_grad():
                state = state.to(device)
                output = self.main_q_network(state)
                action = self.action_selection(last_layer_out=output)
        return action

    def replay(self):
        '''
        experience replay for training
        '''
        if len(self.memory) < BATCH_SIZE:
            return
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.next_state_batch = self.make_minibatch()
        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def make_minibatch(self):
        '''
        sampling a batch of data from experience memory
        '''
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)
        return batch, state_batch, action_batch, reward_batch, next_state_batch

    def get_expected_state_action_values(self):
        '''
        solve the expected state-action value
        '''
        self.main_q_network.eval()
        self.target_q_network.eval()
        batch_out = self.main_q_network(self.state_batch)
        s_a_values = torch.zeros(BATCH_SIZE).to(device)
        for batch_ind in range(BATCH_SIZE):
            s_a_values[batch_ind] = torch.matmul(batch_out[batch_ind], self.action_batch[batch_ind])

        self.state_action_values = s_a_values
        next_batch_out = self.target_q_network(self.next_state_batch)
        next_s_a_values = torch.zeros(BATCH_SIZE).to(device)
        for batch_ind in range(BATCH_SIZE):
            temp = self.action_selection(next_batch_out[batch_ind]).to(device)
            next_s_a_values[batch_ind] = torch.matmul(next_batch_out[batch_ind], temp.t())
        expected_state_action_values = self.reward_batch + self.GAMMA * next_s_a_values
        return expected_state_action_values

    def update_main_q_network(self):
        '''
        backpropagate for agent training
        '''
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        '''
        weight copy to target network
        '''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())



class BSAgent:
    '''
    this class implements agent, based on BSAgentBrain
    '''
    def __init__(self, state_num, action_num, MEC_C, File_num, D_f, learning_rate=0.0001, GAMMA=0.9):
        self.brain = BSAgentBrain(state_num=state_num, action_num=action_num, MEC_C=MEC_C, File_num=File_num, D_f=D_f, learning_rate=learning_rate, GAMMA=GAMMA)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, training=True):
        action = self.brain.decide_action(state, training=training)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()

