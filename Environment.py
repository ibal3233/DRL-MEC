import numpy as np

# this file implements the computation environment
class Multi_off:
    '''
    this class implements the multi-user computation offloading game,
    '''
    def __init__(self, User_num, User_fc, User_e_coe, User_P,  MEC_fc, MEC_C, Band, Channel_num, Channel_Gain, Sigma2, Task_Sf, Task_Df, Tau, Task_If):
        self.user_num = User_num
        self.user_fc = User_fc
        self.user_e_coe = User_e_coe
        self.user_power = User_P
        self.MEC_fc = MEC_fc
        self.MEC_C = MEC_C
        self.bandwidth = Band
        self.channel_num = Channel_num
        self.channel_gain = Channel_Gain
        self.sigma2 = Sigma2
        self.task_sf = Task_Sf
        self.task_df = Task_Df
        self.tau = Tau
        self.task_If = Task_If
        self.subB = Band / Channel_num
        self.game_error = 10**(-20) # the terminate condition of computation game

    def multi_game(self, MEC_caching_state, User_requset, init_off, off_vec_tag=False):
        if off_vec_tag:
            offloading_vector = init_off
        else:
            offloading_vector = np.zeros(self.user_num)
        game_continue_Info = True
        last_cost = 10.0 ** 15
        while game_continue_Info:
            for user_index in range(self.user_num):
                if User_requset[user_index] == 0:
                    offloading_vector[user_index] = 0
                else:
                    Channel_cost = np.zeros(self.channel_num + 1)
                    for chann in range(self.channel_num+1):
                        offloading_vector[user_index] = chann
                        temp_cost = 0.
                        for user_x in range(self.user_num):
                            if User_requset[user_x] != 0:
                                temp_ind = int(User_requset[user_x] - 1)
                                if offloading_vector[user_x] == 0:
                                    temp_cost += self.user_e_coe * (self.task_sf[temp_ind] ** 3) / (self.tau ** 2)
                                else:
                                    temp_interference = 0
                                    for user_y in range(self.user_num):
                                        if offloading_vector[user_y] == offloading_vector[user_x] and user_y != user_x:
                                            temp_interference += self.user_power[user_y] * self.channel_gain[user_y]
                                    temp_rate = self.user_power[user_x] * self.channel_gain[user_x] / (self.sigma2 + temp_interference)
                                    temp_rate = self.subB * np.log2(1 + temp_rate)
                                    temp_rate = temp_rate / 8  # byte/s
                                    if MEC_caching_state[temp_ind] == 0:
                                        temp_cost += self.user_power[user_x] * (self.task_If[temp_ind] + self.task_df[temp_ind]) / temp_rate
                                    else:
                                        temp_cost += self.user_power[user_x] * self.task_If[temp_ind] / temp_rate
                        Channel_cost[chann] = temp_cost
                    temp_a = np.argmin(Channel_cost)
                    offloading_vector[user_index] = temp_a
            cost = self.cost_estimation(offloading_vector, User_requset, MEC_caching_state)
            if np.abs(last_cost-cost) < self.game_error:
                game_continue_Info = False
            last_cost = cost
        return last_cost, offloading_vector

    def cost_estimation(self, offloading_vector, User_requset, MEC_caching_state):
        '''
        compute the energy consumption under given offloading policy, MEC server caching state, user request state
        '''
        cost = 0.
        for user_x in range(self.user_num):
            if User_requset[user_x] != 0:
                temp_ind = int(User_requset[user_x] - 1)
                if offloading_vector[user_x] ==0:
                    cost += self.user_e_coe * (self.task_sf[temp_ind] ** 3) / (self.tau ** 2)
                else:
                    temp_interference = 0
                    for user_y in range(self.user_num):
                        if offloading_vector[user_y] == offloading_vector[user_x] and user_y != user_x:
                            temp_interference += self.user_power[user_y] * self.channel_gain[user_y]
                    temp_rate = self.user_power[user_x] * self.channel_gain[user_x]/(self.sigma2 + temp_interference)
                    temp_rate = self.subB * np.log2(1 + temp_rate)
                    temp_rate = temp_rate / 8  # byte/s
                    if MEC_caching_state[temp_ind] == 0:
                        cost += self.user_power[user_x] * (self.task_If[temp_ind] + self.task_df[temp_ind]) / temp_rate
                    else:
                        cost += self.user_power[user_x] * self.task_If[temp_ind] / temp_rate
        return cost

class Environment:
    '''
    This class implements the wireless computing environment, including computation offloading game,
    '''
    def __init__(self, User_num, User_fc, User_zeta, User_P, MEC_fc, MEC_C, bandw, Ch_num, Sf_max, Sf_min, Df_max, Df_min, tau,\
                 If_max, If_min, Task_num, area_len, sigma2, zipf_gamma, zipf_R, zipf_N):
        self.user_num = User_num
        self.User_fc = User_fc
        self.zeta = User_zeta
        self.User_P = User_P
        self.MEC_fc = MEC_fc
        self.MEC_C = MEC_C
        self.Bandwidth = bandw
        self.ch_num = Ch_num
        self.Sf_max = Sf_max
        self.Sf_min = Sf_min
        self.Df_max = Df_max
        self.Df_min = Df_min
        self.tau = tau
        self. If_max = If_max
        self.If_min = If_min
        self.sigma2 = sigma2
        self.task_num = Task_num
        self.area_length = area_len
        self.zipf_gamma = zipf_gamma
        self.zipg_R = zipf_R
        self.zipf_N = zipf_N
        self.reset()
    def reset(self):
        self.Task_If = np.random.randint(self.If_min, self.If_max + 1, size=self.task_num) * (10.0 ** 6)
        self.Task_Df = np.random.randint(self.Df_min, self.Df_max + 1, size=self.task_num) * (10.0 ** 8)
        self.Task_Sf = np.random.randint(self.Sf_min, self.Sf_max + 1, size=self.task_num) * (10.0 ** 9)
        self.User_Req = np.random.randint(0, self.task_num+1, size=self.user_num) # -----------
        self.caching_state = np.zeros(self.task_num)
        residual_c = self.MEC_C
        for ind in range(self.task_num):
            if self.Task_Df[ind] < residual_c:
                self.caching_state[ind] = 1
                residual_c = residual_c - self.Task_Df[ind]
        BS_x = self.area_length / 2
        BS_y = self.area_length / 2
        self.Channel_gain = np.zeros(self.user_num)
        User_x = np.random.randint(1, self.area_length, size=self.user_num)
        User_y = np.random.randint(1, self.area_length, size=self.user_num)
        for user_index in range(self.user_num):
            distance = np.sqrt((User_x[user_index] - BS_x) ** 2 + (User_y[user_index] - BS_y) ** 2)
            self.Channel_gain[user_index] = distance ** (-4)
        self.potential_game = Multi_off(User_num=self.user_num, User_fc=self.User_fc, User_e_coe=self.zeta, User_P=self.User_P, \
                                        MEC_fc=self.MEC_fc, MEC_C=self.MEC_C, Band=self.Bandwidth, Channel_num=self.ch_num, \
                                        Channel_Gain=self.Channel_gain, Sigma2=self.sigma2, Task_Sf=self.Task_Sf, Task_Df=self.Task_Df, \
                                        Task_If=self.Task_If, Tau=self.tau)
    def user_Request_transfer(self):
        '''
        user request transfer based on the used model in the simulation part
        '''
        temp_req = np.zeros(self.user_num)
        zipf_sum = 0
        for i in range(self.task_num):
            zipf_sum = zipf_sum + 1/((i + 1)**self.zipf_gamma)
        for user_index in range(self.user_num):
            req_prob = np.zeros(self.task_num+1, dtype=float)
            req_prob[0] = self.zipg_R  # request nothing
            if self.User_Req[user_index] == 0:
                for req_task_index in range(self.task_num):
                    req_prob[req_task_index + 1] = (1/((req_task_index + 1)**self.zipf_gamma))/zipf_sum
            else:
                for ind in range(self.zipf_N):
                    if ind + 1 + self.User_Req[user_index] <= self.task_num:
                        index = int(ind + 1 + self.User_Req[user_index])
                        req_prob[index] = (1 - self.zipg_R) * (1 / self.zipf_N)
                    else:
                        index = int(ind + 1 + self.User_Req[user_index] - self.task_num)
                        req_prob[index] = (1 - self.zipg_R) * (1 / self.zipf_N)
            rand_prob = np.random.uniform(0, 1)
            accmulate_prob = 0
            for ind in range(self.task_num+1):
                accmulate_prob += req_prob[ind]
                if accmulate_prob > rand_prob:
                    temp_req[user_index] = int(ind)
                    break
        self.User_Req = temp_req

    def caching_state_transfer(self, caching_action):
        '''
        caching state transfer
        '''
        self.caching_state = caching_action

    def step(self, caching_action):
        '''
        environmwnt step, return reduced energy consumption, and next user request
        '''
        self.caching_state_transfer(caching_action)
        self.user_Request_transfer()
        temp_off = np.zeros(self.user_num)
        cost_caching, off_vector = self.potential_game.multi_game(self.caching_state, self.User_Req, init_off=temp_off, off_vec_tag=False)
        zero_caching = np.zeros(self.task_num)
        cost_non_caching, off_non_vector = self.potential_game.multi_game(zero_caching, self.User_Req, init_off=temp_off, off_vec_tag=False)
        while cost_caching > cost_non_caching:
            cost_caching, _ = self.potential_game.multi_game(self.caching_state, self.User_Req,init_off=off_non_vector, off_vec_tag=True)
        reward = cost_non_caching - cost_caching
        return reward, self.User_Req
