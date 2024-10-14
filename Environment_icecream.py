import random
import itertools
import pandas as pd
import numpy as np
from user_distribution_within_coverage import *

def normalize(value, min_value, max_value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value

class Env(object):

    def __init__(self,target_bs,target_app,beta):

       self.slice=target_app
       self.target=target_bs

       self.eco_transmit_power_ratio=1#can also assume eco slice transmit power is 50% or some percentage of normal slice transmit power0.3
       self.slice_instance = [5, 1, 10] #embb, urllc, mmtc in ms (the guaranteed delay in ms)
       self.power_traffic = 742  # load depended power in Watts
       self.power_static = 139  # static power of slice instance to run their corresponding functions in Watts
       self.power_basic = 18  # cooling the circuit power in Watts (need to change to Joule) *60
       self.psi = [1.5,1.8,1.6,1]  # eMBB, URLLC, mMTC, EcoSlice, power impact factor of slice instances , the larger value means more power consumption
       self.AWGN_dBm = -174  # dBm/Hz #W/Hz
       self.AWGN_W = (10 ** (self.AWGN_dBm / 10)) / 1000  # dBm to Watts conversion
       self.theta= 1#QoS stepness control
       #self.slice_bandwidth_ = [8e+06,10e+06,10e+06,0.5e+06] # 5MHz
       self.slice_bandwidth_ = [15e+06, 20e+06, 15e+06, 5e+06]  # 5MHz
       self.sub_channels = 1 #we dont use sub_channels in this scenario so just put 1
       self.hand_over_bs_slice=-0.0002
       self.hand_over_bs= -0.0001
       self.hand_over_slice=-0.00005
       self.configs_bs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
       self.qos_impacting_factor = beta# QoS impact factor on the overall reward
       self.energy_impacting_factor=1
       self.nei_target_bs = self.neighbor_basestation_list(target_bs)
       self.max_energy_sts=np.zeros(len(self.psi), dtype=float)
       self.min_energy_sts = np.zeros(len(self.psi), dtype=float)

    def Average(self,lst):
        return sum(lst) / len(lst)

    def generate_config(self,neighbouring_bs):
        slice_ = [0, 1] # on and off
        config_list = itertools.product(slice_, neighbouring_bs)
        return list(config_list)

    def user_traffic_info(self,target_bs,target_app):
        if target_app==0:
            app_='Facebook'
        elif target_app==1:
            app_='Youtube'
        else:
            app_='Google'
        directory = r'.\df{}OneWeek05_{}_TrafficUsercount.parquet.gzip'.format(target_bs, app_)
        user_info = pd.read_parquet(directory)
        user_info.reset_index(inplace=True)
        user_traffic_info = user_info.replace(np.nan, 0)
        return user_traffic_info

    def neighbor_basestation_list(self,target_bs):
        self.df_nei_bs = pd.read_csv('Neigboring_BS_distancethr_2_km.csv')
        if self.df_nei_bs[f'df{target_bs}05'].count()==0: #some basestations might not have neighbours
            neighbor_bs_set=[target_bs]
        else:
            df_nei = self.df_nei_bs[f'df{target_bs}05'].str[2:-2]
            df_nei = df_nei.dropna()
            df_nei = df_nei.astype(int)
            df_nei = df_nei.iloc[0:]
            neighbor_bs_set = df_nei.tolist()  # neighboring set of BS
            neighbor_bs_set.insert(0, target_bs)

        return neighbor_bs_set

    def get_length_excluding_zeros(self,lst):
        filtered_list = [x for x in lst if x != 0]
        return len(filtered_list)

    def normalize(self,value, min_value, max_value):
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    def get_reward(self, target_bs, target_app,df_user_traffic_info,e,agent_idx):#total bs load is no longer used because we consider max bs load

        user_bs_association_info=pd.read_csv('User_BS_associated_information.csv')

        df_energy_uti = pd.read_csv(f'energy_utilization_slice_{self.qos_impacting_factor}.csv')
        take_step = df_user_traffic_info.iloc[e]
        num_user_count_ = take_step[2]
        num_user_count = math.ceil(num_user_count_)
        self.user_sets_loc = generate_user_location_within_radius(num_user_count, target_bs, target_app)

        self.df_user_sets_loc=pd.read_csv('user_location_{}bs_{}app_radius.csv'.format(target_bs,target_app)) #.format(target_bs,target_app) #location of users
        self.df_bs_loc = pd.read_csv(r'./base_station_location_France.csv')

        self.df_bs_loc = self.df_bs_loc.set_index('BS')
        self.max_bs_load = self.df_bs_loc.loc[f'df{target_bs}05', 'Max_load']

        self.find_potential_bs_for_user_=find_potential_bs_for_user(target_bs,target_app,self.df_user_sets_loc,self.df_nei_bs,self.df_bs_loc)#distribut user set of target bs respective to neighboring bs
        user_with_potential_bs=pd.read_csv(f'user_set_with_potential_{target_bs}bs_{target_app}app.csv')

        self.Avg_QoS=0
        self.avg_qoS_slice_for_each_nei_bs=np.zeros(len(self.nei_target_bs),dtype=float)
        self.energy_slice_for_each_nei_bs=np.zeros(len(self.nei_target_bs),dtype=float)
        self.energy_slice_eco_for_each_nei_bs = np.zeros(len(self.nei_target_bs), dtype=float)
        z = 0  # for the index of nei_target_bs set

        used_nei_bs = len(self.nei_target_bs)
        for i in self.nei_target_bs:

            user_cnt=user_with_potential_bs['Lon_{}'.format(i)].count()
            take_step_user_info = df_user_traffic_info.iloc[e]
            per_user_demand = take_step_user_info[3]  # [1] TotalVol_Traffic  [2]Total_UserCount  [3]Per_User_Usage [4] New_UserCount_static  (with reset index)

            if user_cnt==0: #means no user of target bs is potentially associated to that neighboring BS
                z=z+1
                used_nei_bs=used_nei_bs-1
            else:
                trafficload_slice = per_user_demand * user_cnt
                achieved_user_delay = np.zeros(user_cnt, dtype=float)
                QoS = np.zeros(user_cnt, dtype=float)

                sel_slice_, sel_bs_ = user_bs_association_info['{}bs_{}app_slice'.format(target_bs, target_app)].iloc[z], \
                user_bs_association_info['{}bs_{}app_bs'.format(target_bs, target_app)].iloc[z]
                sel_slice = int(sel_slice_)
                sel_bs = int(sel_bs_)
                transmit_pwr_watt = dbm_to_watt(self.df_bs_loc.loc[f'df{sel_bs}05', 'Transmit_power'])
                transmit_pwr_watt_for_eco = transmit_pwr_watt * self.eco_transmit_power_ratio  # assume eco slice transmit power is 50% of normal slice transmit power

                if sel_slice == 1:
                    for k in range(user_cnt):
                        k_loc = (user_with_potential_bs.loc[k, f'Lon_{i}'], user_with_potential_bs.loc[k, f'lat_{i}'])
                        sel_bs_lat = self.df_bs_loc.loc[f'df{sel_bs}05', 'lat']  # long of target BS
                        sel_bs_lon = self.df_bs_loc.loc[f'df{sel_bs}05', 'Lon']
                        loc_sel_bs = (sel_bs_lon, sel_bs_lat)
                        if len(self.nei_target_bs) ==1:
                            interf=0
                        else:
                            df_sel_bs = self.df_nei_bs['df{}05'.format(sel_bs)].str[2:-2]

                            df_sel_bs = df_sel_bs.dropna()
                            df_sel_bs = df_sel_bs.astype(np.int64)
                            df_sel_bs = df_sel_bs.iloc[0:]
                            neighbor_bs_sel_bs = df_sel_bs.tolist()

                            interf = interference(k_loc ,neighbor_bs_sel_bs)
                        dist_x = distance_in_km(k_loc, loc_sel_bs)
                        PL_x = pathloss(dist_x)
                        CG_x = channelgain(PL_x)
                        sinr_x = sinr(transmit_pwr_watt, CG_x, self.AWGN_W, interf)

                        user_throughput_x = achievable_rate(self.slice_bandwidth_[target_app], user_cnt, sinr_x)#bps
                        user_throughput_x = round(user_throughput_x, 4) #Round a number to only two decimals:

                        achieved_user_delay[k] = (
                                        per_user_demand / ((user_throughput_x) / 8))  # divided by 1000 to convert to ms

                        if target_bs == sel_bs:  # target slice #to add handover case
                            QoS[k] = QoS[k] + 1 / (
                                    1 + np.exp(-self.theta * (self.slice_instance[target_app] - achieved_user_delay[k])))
                        else:
                            QoS[k] = QoS[k] + 1 / (1 + np.exp(-self.theta * (self.slice_instance[target_app] - achieved_user_delay[
                                    k]))) + self.hand_over_bs
                    self.avg_qoS_slice_for_each_nei_bs[z] = (sum(QoS) / user_cnt)

                    traffic_portion=trafficload_slice/self.max_bs_load
                    df_energy_uti[f'{sel_bs}bs_{target_app}app_energy_slice'].iloc[0] = df_energy_uti[f'{sel_bs}bs_{target_app}app_energy_slice'].iloc[0]+(traffic_portion) *self.psi[target_app] *self.power_traffic+ self.psi[target_app] *self.power_static
                    df_energy_uti.to_csv(f'energy_utilization_slice_{self.qos_impacting_factor}.csv')

                else:
                    for k in range(user_cnt):
                        k_loc = (user_with_potential_bs.loc[k, f'Lon_{i}'], user_with_potential_bs.loc[k, f'lat_{i}'])
                        sel_bs_lat = self.df_bs_loc.loc[f'df{sel_bs}05', 'lat']  # long of target BS
                        sel_bs_lon = self.df_bs_loc.loc[f'df{sel_bs}05', 'Lon']
                        loc_sel_bs = (sel_bs_lon, sel_bs_lat)

                        if len(self.nei_target_bs) == 1:
                            interf = 0
                        else:
                            df_sel_bs = self.df_nei_bs['df{}05'.format(sel_bs)].str[2:-2]

                            df_sel_bs = df_sel_bs.dropna()
                            df_sel_bs = df_sel_bs.astype(np.int64)
                            df_sel_bs = df_sel_bs.iloc[0:]
                            neighbor_bs_sel_bs = df_sel_bs.tolist()

                            interf = interference(k_loc, neighbor_bs_sel_bs)

                        dist_x = distance_in_km(k_loc, loc_sel_bs)
                        PL_x = pathloss(dist_x)
                        CG_x = channelgain(PL_x)

                        sinr_x = sinr(transmit_pwr_watt_for_eco, CG_x, self.AWGN_W, interf)

                        user_throughput_x = achievable_rate(self.slice_bandwidth_[3], user_cnt, sinr_x)
                        user_throughput_x = round(user_throughput_x, 4)  # Round a number to only two decimals:

                        achieved_user_delay[k] = per_user_demand / ((user_throughput_x) / 8)

                        if target_bs == sel_bs:  # target slice #to add handover case
                            QoS[k] = QoS[k] + 1 / (1 + np.exp(-self.theta * (self.slice_instance[target_app]-achieved_user_delay[k] )))+self.hand_over_slice

                        else:
                            QoS[k] = QoS[k] + 1 / (1 + np.exp(-self.theta * (self.slice_instance[target_app]-achieved_user_delay[k] ))) + self.hand_over_bs_slice

                    self.avg_qoS_slice_for_each_nei_bs[z] = (sum(QoS) / user_cnt)


                    traffic_portion=trafficload_slice/self.max_bs_load

                    df_energy_uti[f'{sel_bs}bs_energy_eco'].iloc[0] = df_energy_uti[f'{sel_bs}bs_energy_eco'].iloc[
                                                                          0] + (traffic_portion) * \
                                                                      self.psi[3] * self.power_traffic + self.psi[
                                                                          3] * self.power_static

                    df_energy_uti[f'{agent_idx}agent_energy_eco'].iloc[0] = \
                    df_energy_uti[f'{agent_idx}agent_energy_eco'].iloc[0] + (traffic_portion) * \
                    self.psi[3] * self.power_traffic + self.psi[3] * self.power_static
                    df_energy_uti.to_csv(f'energy_utilization_slice_{self.qos_impacting_factor}.csv')
                z = z + 1

        self.Avg_QoS=(sum(self.avg_qoS_slice_for_each_nei_bs)/used_nei_bs)

        return float(self.Avg_QoS),float(self.max_energy_sts[target_app]),float(self.min_energy_sts[target_app]),float(self.max_energy_eco),float(self.min_energy_eco)

    def get_new_state(self, target_bs, target_app,step,agent_idx):
        if step == 0: # initially we don't have info for overall energy, qos
            self.Overall_Energy=0
            self.Average_QoS=0
            self.state_tuple = [self.Overall_Energy, self.Average_QoS]
        else:
            df_energy_uti = pd.read_csv(f'energy_utilization_slice_{self.qos_impacting_factor}.csv')
            self.Energy_Slice=df_energy_uti[f'{target_bs}bs_{target_app}app_energy_slice'].iloc[0]
            self.Energy_Eco = df_energy_uti[f'{agent_idx}agent_energy_eco'].iloc[0]
            self.Overall_Energy=self.Energy_Slice
            self.Overall_Energy=normalize(self.Overall_Energy, self.min_energy_sts[target_app], self.max_energy_sts[target_app])
            self.state_tuple = [self.Overall_Energy,self.Average_QoS]
        return np.array(self.state_tuple,dtype=float)

    def choose_config(self,target_bs,target_app,df_user_traffic_info,e,agent_idx):
        self.Average_QoS,self.max_energy_sts_,self.min_energy_sts_,self.max_energy_eco,self.min_energy_eco = self.get_reward(target_bs,target_app,df_user_traffic_info,e,agent_idx)
        return self.Average_QoS,self.max_energy_sts_,self.min_energy_sts_,self.max_energy_eco,self.min_energy_eco




