import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
from Environment import Environment
from Agent import BSAgent

# The annotations' name is the same as the paper's variables
MEC_C = 20*(10**8)                      # MEC server’s cache size: C
MEC_fc = 50*(10**9)                     # MEC server’s CPU capability: f_{\text{C}}
User_num = 20                           # Number of users: K
User_fc = 1*(10**9)                     # User k’s CPU capability: f_k
User_e_coe = 5*(10**(-27))              # Energy coefficient of mobile devices: \zeta
User_power = 0.5 * np.ones(User_num)    # User k’s transmit power: p_k
Task_num = 50                           # Number of tasks: F
Task_If_low = 1                         # Input parameters’ min size of task f: I_{\max}
Task_If_high = 5                        # Input parameters’ max size of task f: I_{\max}
Task_Df_low = 1                         # Input parameters’ min size of task f: I_{\min}
Task_Df_high = 5                        # Max data size of the task f’s software: D_{\max}
Task_Sf_low = 1
Task_Sf_high = 5                        # Max computation load of task f: S_{\max}
Tau = 5
Bandwidth = 50*(10**6)                  # Wireless transmission bandwidth: B
Channel_num = 10                        # Number of subchannels: M
area_length = 200.0                     # Cell region: 200*200
sigma2 = 2*(10**(-13))                  # Noise
zipf_gamma = 0.7                        # Environment parameter in (24): \delta
zipg_R = 0.1                            # Environment parameter in (24): R
zipf_N = 3                              # Environment parameter in (24): N

BATCH_SIZE = 32                         # Batch size
CAPACITY = 10000                        # Replay memory
GAMMA = 0.9                             # Discount factor

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
env = Environment(User_num=User_num, User_fc=User_fc, User_zeta=User_e_coe, User_P=User_power, MEC_fc=MEC_fc, MEC_C=MEC_C, bandw=Bandwidth,\
                  Ch_num=Channel_num, Sf_max=Task_Sf_high, Sf_min=Task_Sf_low, Df_max=Task_Df_high, Df_min=Task_Df_low, tau=Tau, \
                  If_max=Task_If_high, If_min=Task_If_low, Task_num=Task_num, area_len=area_length, sigma2=sigma2,zipf_gamma=zipf_gamma,\
                  zipf_R=zipg_R, zipf_N=zipf_N)
num_states = User_num
num_actions = Task_num
agent = BSAgent(state_num=num_states, action_num=num_actions, MEC_C=MEC_C, File_num=Task_num, D_f=env.Task_Df)#

def play_gameDQN(slots, train=True):
    '''
    this function runs one time slot, the detailed steps as Figure 1(b) in our work
    '''
    episode_reward = 0
    observation = env.User_Req
    state = observation
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = torch.unsqueeze(state, 0)
    action1 = agent.get_action(state, training=train)
    action = torch.squeeze(action1)
    action = action.numpy()
    reward, next_observation = env.step(action)
    next_state = torch.from_numpy(next_observation).type(torch.FloatTensor)
    next_state = torch.unsqueeze(next_state, 0)
    episode_reward += reward
    reward = torch.FloatTensor([reward])
    if train:
        agent.memorize(state, action1, next_state, reward)
        agent.update_q_function()
        if slots % 5 == 0:
            agent.update_target_q_function()
    return episode_reward

# Training:
slots = 2000 # training episodes
eva_step_slots = 5  # evaluate the reward every 5 episodes
evaluate_slot_nums = 10 # evaluation times every eva_step_slots, the reward is the average value of evaluate_slot_nums times reward

for slot in range(slots+eva_step_slots):
    if slot % 5 == 0:
        episode = int(slot / 5)
        reward_sum = 0.
        for ave_times in range(evaluate_slot_nums):
            episode_reward = play_gameDQN(slot, train=False)
            reward_sum += episode_reward
        reward_eva = reward_sum / evaluate_slot_nums
        print('episode=', episode, 'reward=', reward_eva)
    else:
        episode_reward = play_gameDQN(slot, train=True)


# # inference
# inference_times = 1000 # the inference times
# for slot in range(inference_times):
#     state = env.User_Req
#     state = torch.from_numpy(state).type(torch.FloatTensor)
#     state = torch.unsqueeze(state, 0)
#     action1 = agent.get_action(state, training=False)
#     action = torch.squeeze(action1)
#     action = action.numpy()
#     reward, _ = env.step(action)
#     print('reduced energy consumption=', reward)



