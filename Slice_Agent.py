import numpy as np
import torch
from NeuralNet import *
from user_distribution_within_coverage import *

class Slice_Agent:

    def __init__(self, env,target_bs,target_app):
        self.n_inputs=2 #dimension of the input
        self.n_outputs=23 #dimenion of the output
        self.model=Net(self.n_inputs,self.n_outputs)
        self.env=env
        self.target_bs=target_bs
        self.target_app=target_app # index of target_app
        self.optimizer=optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion=nn.MSELoss()
        self.best_payoff=0
        self.eps=0.2 #
        self.bs_configs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1],[1,0,1], [1, 1, 1]]

    def softmax(self,data, tau=1.2):
        softm = np.exp(data / tau) / np.sum(np.exp(data / tau))
        return softm

    def normalize(self,probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

    def step_and_action(self,epochs,beta,agent_idx):

        state = self.env.get_new_state(self.target_bs, self.target_app, epochs)

        state_reshape = np.reshape(state, (1, state.size))
        scaler = MinMaxScaler(feature_range=(0, 1))  # put the data into the range 0 and 1
        state_fit = scaler.fit_transform(state_reshape)

        state_input = torch.FloatTensor(state_fit)

        self.rewards_pred = self.model(state_input)

        self.rewards_pred_numpy=self.rewards_pred.data.numpy().copy()

        action_probas = self.softmax(self.rewards_pred_numpy)

        action_prob = action_probas.flatten()

        df_user_traffic_info=self.env.user_traffic_info(self.target_bs,self.target_app)

        take_step = df_user_traffic_info.iloc[epochs]
        num_user_count_ = take_step[2]
        num_user_count = math.ceil(num_user_count_)

        neighbor_bs_set=self.env.neighbor_basestation_list(self.target_bs)

        config = self.env.generate_config(neighbor_bs_set)

        #user_sets_loc = self.env.generate_user_location_associated_with_neighbours(self.target_bs, self.target_app,num_user_count, eps)
        self.user_sets_loc=generate_user_location_within_radius(num_user_count,self.target_bs,self.target_app)

        selected_slice = []
        selected_bs = []
        index_config = []

        #update action prob with slice-on-off status from large time scale
        if epochs % 10 == 0 and epochs != 0:
            df_bs_config=pd.read_csv(f'slice_on_and_off_{beta}.csv')
            for i in range(len(neighbor_bs_set)):
                bs_config_idx=df_bs_config[f'config_{neighbor_bs_set[i]}bs'].iloc[0]
                bs_config=self.bs_configs[bs_config_idx]
                slice_on_off=bs_config[self.target_app]
                if slice_on_off==0:
                    action_prob[i]=action_prob[i]+action_prob[i + len(neighbor_bs_set)]    #if slice is off, its corresponding probabilities is 0
                else:
                    action_prob[i]=action_prob[i]


        for i in range(len(neighbor_bs_set)):

            p = np.random.random()
            if self.target_bs == neighbor_bs_set[i]:

                set_configs = [i, i + len(neighbor_bs_set)]
                action_prob_ = [action_prob[i], action_prob[i + len(neighbor_bs_set)]]

                if p <= self.eps:
                    index_config_ = np.random.choice(set_configs, p=action_prob_)
                else:  # exploi  action
                    index_config_ = set_configs[np.argmax(action_prob_)]
            else:

                set_configs = [0, i, 0 + len(neighbor_bs_set), i + len(neighbor_bs_set)]
                action_prob_ = [action_prob[0], action_prob[i], action_prob[0 + len(neighbor_bs_set)],
                                action_prob[i + len(neighbor_bs_set)]]

                if p <= self.eps:
                    index_config_ = np.random.choice(set_configs, p=action_prob_)
                else:
                    index_config_ = set_configs[np.argmax(action_prob_)]

            index_config.append(index_config_)
            selected_slice_, selected_bs_ = config[index_config_]  # get the real config setting from the index_config
            selected_slice.append(selected_slice_)
            selected_bs.append(selected_bs_)

        df_user_assoication = pd.DataFrame(
            {f'{self.target_bs}bs_{self.target_app}app_slice': selected_slice, f'{self.target_bs}bs_{self.target_app}app_bs': selected_bs})
        df_user_association_index= pd.DataFrame({f'{agent_idx}agent_indexConfig': index_config})

        return df_user_assoication,df_user_association_index,self.target_bs,self.target_app

    def reward_and_regret(self,epochs,agent_idx):

        df_user_traffic_info_=self.env.user_traffic_info(self.target_bs,self.target_app)

        rewards, Avg_QoS, energy_slice,energy_slice_nor, energy_eco, energy_eco_nor = self.env.choose_config(self.target_bs, self.target_app, df_user_traffic_info_, epochs,agent_idx)

        if rewards> self.best_payoff:
            self.best_payoff = rewards
        else:
            self.best_payoff= self.best_payoff

        regrets= self.best_payoff- rewards

        return rewards,regrets,Avg_QoS,energy_slice,energy_slice_nor,energy_eco,energy_eco_nor

    def loss_fn(self,global_reward,agent_idx):

        df = pd.read_csv('User_BS_associated_index.csv')
        true_rewards= self.rewards_pred.data.numpy().copy()
        true_rewards.put([x for x in df[f'{agent_idx}agent_indexConfig'].tolist() if math.isnan(x) == False],global_reward)
        self.loss = self.criterion(self.rewards_pred,torch.FloatTensor(true_rewards))

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss






