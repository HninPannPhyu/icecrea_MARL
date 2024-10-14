from Basestation_Agent import *
from Slice_Agent import *
import matplotlib.pyplot as plt
from numpy import inf

def running_mean(data,window=50):
    c = data.shape[0] - window
    smoothened = np.zeros(c)
    conv = np.ones(window)
    for i in range(c):
        smoothened[i] = (data[i:i+window] @ conv)/window
    return smoothened

def sum_of_triplets(arr):
    result = []
    for i in range(0, len(arr) - 2, 3):
        triplet = arr[i:i+3]
        triplet_sum = sum(triplet)
        result.append(triplet_sum)
    return result

def normalize(value, min_value, max_value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value

def create_agents(bs_set,app_set,beta):
    agents_sts = [None]  # 1 dimensional-index agents for small time scale
    agents_lts=[None]   # 1 dimensional-index agents for large time scale
    for bs in bs_set:
        for app in app_set:
            if app == 'Facebook':
                app_ = 0
            elif app == 'Youtube':
                app_ = 1
            else:
                app_ = 2

            cenv=Env(bs,app_,beta)

            agents_sts.append(Slice_Agent(cenv, bs, app_))
        agents_lts.append(Basestation_Agent(bs))
    return agents_sts,agents_lts

def Average(lst):
    return sum(lst) / len(lst)

def train_network(epochs,beta):
    allBS = range(62)
    exclu = np.array([1, 35,36, 47, 48]) #lots of missing data for those base station number
    bs_set = np.setdiff1d(allBS, exclu)
    app_set = ['Facebook', 'Youtube', 'Google']
    n_agents_sts = len(bs_set) * len(app_set)
    n_agents_lts=len(bs_set)

    avg_qos_lts_save = []
    energy_lts_save=[]

    avg_qos_bef_lts=np.zeros(n_agents_sts,dtype=float)
    energy_slice_bef_lts=np.zeros(n_agents_sts,dtype=float)
    energy_eco_bef_lts=np.zeros(n_agents_sts,dtype=float)


    rewards_lts = np.zeros((n_agents_lts+1 , round((epochs)/10)+1), dtype=float)
    regrets_lts = np.zeros((n_agents_lts+1 , round((epochs)/10)+1), dtype=float)
    loss_lts = np.zeros((n_agents_lts+1, round((epochs)/10)+1), dtype=float)
    avg_qos_lts = np.zeros((n_agents_lts + 1, round((epochs) / 10)+1), dtype=float)
    energy_slice_lts = np.zeros((n_agents_lts + 1, round((epochs) / 10)+1), dtype=float)

    config_lts = np.zeros((n_agents_lts + 1, round((epochs) / 10) + 1), dtype=float)

    loss_sts= np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    avg_qos=np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    energy_slice=np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    energy_eco=np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    energy_slice_nor = np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    energy_eco_nor = np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    rewards_sts=np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    regrets_sts=np.zeros((n_agents_sts + 1, epochs + 1), dtype=float)
    global_rewards_sts=[]
    global_rewards_lts=[]

    global_reward_lts=0

    agents_sts,agents_lts = create_agents(bs_set, app_set,beta)  # 1-index agents

    for e in range(0, epochs + 1):

        if e%10 ==0 and e!=0:
            config_lst=[]
            for j in range(1,n_agents_lts+1):
                agent_lts=agents_lts[j]
                config,target_bs=agent_lts.step_and_action(e // 10, j, avg_qos_lts_save, energy_lts_save)
                config_lts[j][e // 10]=config
                modified_df = pd.DataFrame({f'config_{target_bs}bs': config},index=[0])
                config_lst.append(pd.Series(modified_df[f'config_{target_bs}bs']))
            config_lst=pd.concat(config_lst,axis=1)
            config_lst.to_csv(f'slice_on_and_off_{beta}.csv')

        else:
            pass


        rewards_=np.zeros(n_agents_sts, dtype=float)

        result = []
        result1=[]

        for i in range(1, n_agents_sts + 1):
            agent_sts=agents_sts[i]
            user_bs_association, user_bs_association_index,target_bs,target_app=agent_sts.step_and_action(e, beta,i)
            result.append(pd.Series(user_bs_association[f'{target_bs}bs_{target_app}app_slice']))
            result.append(pd.Series(user_bs_association[f'{target_bs}bs_{target_app}app_bs']))
            result1.append(pd.Series(user_bs_association_index[f'{i}agent_indexConfig']))
        result = pd.concat(result, axis=1)
        result1 = pd.concat(result1, axis=1)
        result.to_csv('User_BS_associated_information.csv')
        result1.to_csv('User_BS_associated_index.csv')


            #for large time scale reward calculation ,calculate the 10 mins of avg and energy

        j=0
        for i in range(1, n_agents_sts + 1):
            agent_sts = agents_sts[i]
            rewards_[j], regrets_, avg_qos_, energy_slice_,energy_slice_nor_, energy_eco_,energy_eco_nor_ = agent_sts.reward_and_regret(e,i)
            rewards_sts[i][e] = rewards_[j]  # rewards_[j] is to keep all the rewards_sts for n_agents_sts in order to calculate the global reward easily
            regrets_sts[i][e] = regrets_

            avg_qos[i][e] = avg_qos_
            energy_slice[i][e] = energy_slice_
            energy_eco[i][e] = energy_eco_
            energy_slice_nor[i][e] = energy_slice_nor_
            energy_eco_nor[i][e] = energy_eco_nor_
            print(
                f'Epoch {e + 1}  Agent{i} \t\t   Rewards: {format(rewards_[j], ".3f")}  \t\t Regrets: {format(regrets_, ".3f")} \t\t   QoS: {format(avg_qos_, ".3f")} \t\t   Energy: {format(energy_slice_, ".3f")}')
            j=j+1

        if e % 9 == 0 and e != 0:
            for i in range(1, n_agents_sts + 1):
                avg_qos_bef=avg_qos[i][:]
                avg_qos_bef[np.isinf(avg_qos_bef)]=0   #replace inf with zero
                avg_qos_bef_lts[i-1]=Average(avg_qos_bef[e-9:e])
                energy_slice_bef=energy_slice[i][:] 
                energy_slice_bef[np.isinf(energy_slice_bef)]=0 #replace inf with zero
                energy_slice_bef_lts[i-1]=sum(energy_slice_bef[e-9:e])
                energy_eco_bef=energy_eco[i][:] 
                energy_eco_bef[np.isinf(energy_eco_bef)]=0 
                energy_eco_bef_lts[i-1]=sum(energy_eco_bef[e-9:e])

            avg_qos_lts_ = np.mean(avg_qos_bef_lts.reshape(-1, 3),
                                   axis=1)  # Average of every consecutive triplet (3) of elements of a given array
            sum_energy_lts = sum_of_triplets(
                energy_slice_bef_lts)  # sum of every consecutive triplet of elements of a given array
            sum_energy_eco = sum_of_triplets(energy_eco_bef_lts)
            sum_energy_incl_eco = np.add(sum_energy_lts, sum_energy_eco)  # summation of two array by index
            avg_qos_lts_save.append(avg_qos_lts_)
            energy_lts_save.append(sum_energy_incl_eco)


        if e % 10 == 0 and e != 0:
            k=0
            rewards_lts_list=[]
            best_payoff_lts=0
            for j in range(1,n_agents_lts+1):

                avg_qos_lts[j][e//10]=avg_qos_lts_save[(e // 10)-1][j - 1]
                energy_slice_lts[j][e // 10]=energy_lts_save[(e // 10) - 1][j - 1]
                rewards_lts[j][e // 10]= 1 / (energy_lts_save[(e // 10) - 1][j - 1]) + beta * avg_qos_lts_save[(e // 10) - 1][j - 1]
                rewards_lts_list.append(rewards_lts[j][e // 10])
                if rewards_lts[j][e//10] > best_payoff_lts:
                    best_payoff_lts = rewards_lts[j][e//10]
                else:
                    best_payoff_lts = best_payoff_lts

                regrets_lts[j][e//10] = best_payoff_lts - rewards_lts[j][e//10]

                k=k+1
                print(
                    f'Epoch {e + 1}  LargeAgent{j} \t\t   Rewards: {format(rewards_lts[j][e//10], ".3f")}  \t\t Regrets: {format(regrets_lts[j][e//10], ".3f")}')

            global_reward_lts=sum(rewards_lts_list)
            global_rewards_lts.append(global_reward_lts)

            print(f'Epoch {e + 1}  \t\t   Large_Global Reward: {format(global_reward_lts, ".3f")}')

        global_reward_sts=sum(rewards_)
        global_rewards_sts.append(global_reward_sts)
        print(f'Epoch {e + 1}  \t\t   Small_Global Reward: {format(global_reward_sts, ".3f")}')

        if e % 10 == 0 and e != 0:
            for j in range(1, n_agents_lts + 1):
                agent_lts = agents_lts[j]
                loss_ = agent_lts.loss_fn(global_reward_lts, j,beta)
                loss_lts[j][e//10] = loss_.item()
                print(f'Epoch {e + 1}  LargeAgent{j} \t\t   Loss: {format(loss_, ".3f")}')

        for i in range(1, n_agents_sts + 1):
            agent_sts=agents_sts[i]
            loss_=agent_sts.loss_fn(global_reward_sts, i)
            loss_sts[i][e]=loss_.item()
            print( f'Epoch {e + 1}  Agent{i} \t\t   Loss: {format(loss_, ".3f")}')

    return ( np.array(rewards_lts),np.array(rewards_sts),np.array(regrets_lts), np.array(regrets_sts), np.array(global_rewards_lts), np.array(global_rewards_sts), np.array(avg_qos_lts), np.array(avg_qos), np.array(energy_slice_lts), np.array(energy_slice), np.array(energy_eco), np.array(loss_lts),np.array(loss_sts),np.array(config_lts), n_agents_lts,n_agents_sts)


def main() -> None:
    epochs =3000
    beta=5 #qos impacting factor
    Rewards_lts,Rewards_sts,Regrets_lts,Regrets_sts,Global_Rewards_lts,Global_Rewards_sts,Avg_QoS_lts,Avg_QoS_sts,Energy_Slice_lts,Energy_Slice_sts,Energy_Eco,Loss_lts,Loss_sts,config_lts,n_agents_lts,n_agents_sts= train_network(epochs, beta)


    Fig_main = plt.figure(f'./Small_DMAB_globalreward_{beta}beta.png')
    plt.plot(running_mean(Global_Rewards_sts, window=50), label="Global reward")
    plt.xlabel('Epochs')
    plt.ylabel('Global Rewards Small Time Scale')
    plt.title('Global Rewards Small Time Scale')
    plt.legend()
    plt.show()

    Fig_main = plt.figure(f'./Large_DMAB_globalreward_{beta}beta.png')
    plt.plot(running_mean(Global_Rewards_lts, window=50), label="Global reward")
    plt.xlabel('Epochs')
    plt.ylabel('Global Rewards Large Time Scale')
    plt.title('Global Rewards Large Time Scale')
    plt.legend()
    plt.show()

    for j in range(1, n_agents_lts + 1):
        Fig_main1 = plt.figure(
            f'./Large_DMAB_reward_{j}_{beta}beta.png')
        plt.plot(running_mean(Rewards_lts[j], window=50), label="avg reward lts")
        plt.xlabel('Epochs')
        plt.ylabel(f'Rewards_lts{j}')
        plt.title(f'EQ_Rewards_{j}_basestation_{beta}beta')
        plt.legend()
        plt.show()
        
        Fig_main2 = plt.figure(
            r'./Large_DMAB_Regrets_{}_{}beta.png'.format(j, beta))
        plt.plot(running_mean(Loss_lts[j], window=50), label="Regrets_lts")
        plt.xlabel('Epochs')
        plt.ylabel(f'Regrets_lts{j}')
        plt.title(f'Regrets_lts{j}')
        plt.legend()
        # plt.show()
        
        Fig_main3 = plt.figure(
            r'./Large-DMAB_qos_{}_{}beta.png'.format(j, beta))
        plt.plot(running_mean(Avg_QoS_lts[j], window=50), label="Avg_QoS_lts")
        plt.xlabel('Epochs')
        plt.ylabel(f'QoS{j}')
        plt.title(f'QoS{j}')
        plt.legend()
        plt.show()
       
    for i in range (1, n_agents_sts + 1):
        Fig_main1 = plt.figure(f'./Small_DMAB_reward_{i}_{beta}beta.png')
        plt.plot(running_mean(Rewards_sts[i], window=50), label="avg reward")
        plt.xlabel('Epochs')
        plt.ylabel(f'Rewards_sts{i}')
        plt.title(f'EQ_Rewards_{i}_agent_{beta}beta')
        plt.legend()
        #plt.show()
        
        Fig_main2 = plt.figure(r'./Small_DMAB_Loss_{}_{}beta.png'.format(i,beta))
        plt.plot(running_mean(Loss_sts[i], window=50), label="Loss_sts")
        plt.xlabel('Epochs')
        plt.ylabel(f'Loss_sts{i}')
        plt.title(f'Loss_sts{i}')
        plt.legend()
        plt.show()

        Fig_main3 = plt.figure(r'./Small-DMAB_qos_{}_{}beta.png'.format(i,beta))
        plt.plot(running_mean(Avg_QoS_sts[i], window=50), label="Avg_QoS_sts")
        plt.xlabel('Epochs')
        plt.ylabel(f'QoS{i}')
        plt.title(f'QoS{i}')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
