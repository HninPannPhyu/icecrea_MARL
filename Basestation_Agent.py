import math
from NeuralNet import *

class Basestation_Agent:

    def __init__(self, target_bs):
        self.n_inputs=2 # state input: Overall Energy and QoS in average
        self.n_outputs=8 #output is reward distribution for possible configuration
        self.model=Net(self.n_inputs,self.n_outputs)
        self.target_bs=target_bs
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.best_payoff = 0
        self.configs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1],[1,0,1],[1, 1, 1]]
        self.eps=0.1 #explore or exploit

    def softmax(self,data, tau=1.2):
        softm = np.exp(data / tau) / np.sum(np.exp(data / tau))
        return softm

    def normalize(self,probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

    def step_and_action(self,epochs,agent_idx,avg_qos,energy):

        state_qos=avg_qos[epochs-1][agent_idx-1]
        state_energy=energy[epochs-1][agent_idx-1]
        state=np.array([state_energy,state_qos],dtype=float)
        state_reshape = np.reshape(state, (1, state.size))
        scaler = MinMaxScaler(feature_range=(0, 1))  # put the data into the range 0 and 1
        state_fit = scaler.fit_transform(state_reshape)

        state_input = torch.FloatTensor(state_fit)
        self.rewards_pred = self.model(state_input)

        action_probas = self.softmax(self.rewards_pred.data.numpy().copy())

        action_prob_ = action_probas.flatten()

        action_prob_nor=self.normalize(action_prob_)
        p = np.random.random()

        n_configs=range(len(self.configs))

        if p <= self.eps:
            self.config_idx = np.random.choice(n_configs, p=action_prob_nor)
        else:  # exploi action
            self.config_idx= n_configs[np.argmax(action_prob_nor)]

        return self.config_idx,self.target_bs

    def loss_fn(self, global_reward,agent_idx,beta): # get the config idx from dataframe using target_bs
        df=pd.read_csv(f'slice_on_and_off_{beta}.csv')
        true_rewards = self.rewards_pred.data.numpy().copy()
        true_rewards[:, df[f'config_{self.target_bs}bs'].iloc[0]] = global_reward
        self.loss = self.criterion(self.rewards_pred, torch.FloatTensor(true_rewards))

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

