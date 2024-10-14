import numpy as np
import pandas as pd
import haversine as hs
import math
import random


import math

def watt_to_dbm(watt):
    dbm = 10 * math.log10(watt * 1000)
    return dbm

def sinr(transmitpower,channel_gain,AWGN,interference): # in dB but need to convert to watts
    z= (transmitpower*abs(channel_gain)**2)/(interference+AWGN)
    return z

def pathloss(distance):# in dB
    y = 128.1 + 37.6 * np.log(distance) #distance in km
    return y

def channelgain(pathloss):
    channel_gain=10**(-pathloss/10)
    return channel_gain

def achievable_rate(slice_bandwidth,connected_user,sinr): #bits per second , slice_bandwidth in Hz
    achi_rate = (slice_bandwidth / connected_user) * math.log2(1 + sinr)
    return achi_rate

def distance_in_km(userlocation,bslocation):
    loc_bs=bslocation
    loc_user=userlocation
    distance_km = hs.haversine(loc_bs,loc_user)
    return distance_km

def interference(loc_user_i, neigbor_bs_set):
    interfer=0
    for bs in neigbor_bs_set:
        df_bs_loc = pd.read_csv(f'./base_station_location_France.csv')
        df_bs_loc = df_bs_loc.set_index('BS')
        loc_bs = (df_bs_loc.loc[f'df{bs}05', 'Lon'], df_bs_loc.loc[f'df{bs}05', 'lat'])
        distance=distance_in_km(loc_user_i,loc_bs)
        PL=pathloss(distance)
        CG=channelgain(PL)
        interfer = interfer + df_bs_loc.loc[f'df{bs}05', 'Transmit_power'] * abs(CG)**2

    return interfer

def dbm_to_watt(dbm):
    watt = 10 ** ((dbm - 30) / 10)
    return watt

def find_transmit_power(distance_km,user_received_pwr): #user_received_pwr_watt
    pathloss_=pathloss(distance_km)
    channel_gain=channelgain(pathloss_)
    transmit_pwr=user_received_pwr/channel_gain
    return transmit_pwr #watt

