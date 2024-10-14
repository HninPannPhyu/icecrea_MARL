import math
import random
from geopy.point import Point
from geopy.distance import distance
import pandas as pd
from rate_model import *
import haversine as hs
from math import radians, sin, cos, sqrt

def generate_user_location_within_radius(num_users,target_bs,target_app):
    # Center coordinates (latitude, longitude)
    df_bs_loc = pd.read_csv(f'./base_station_location_France.csv')
    df_bs_loc = df_bs_loc.set_index('BS')
    center_latitude = df_bs_loc.loc[f'df{target_bs}05', 'lat']
    center_longitude = df_bs_loc.loc[f'df{target_bs}05', 'Lon']

    # Radius in meter
    radius_m = (df_bs_loc.loc[f'df{target_bs}05', 'Coverage_km'])/1000

    lat_fn=[]
    long_fn=[]
    for _ in range(num_users):
        # Generate a random angle within the circle
        r = radius_m * sqrt(random.uniform(0, 1))
        theta = random.uniform(0, 1) * 2 * math.pi

        lat = center_latitude + r * cos(theta)
        lon = center_longitude + r * sin(theta)

        lat_fn.append(lat)
        long_fn.append(lon)

    save_data = pd.DataFrame(
        {f'Lon_{target_bs}': long_fn,
         f'lat_{target_bs}': lat_fn})

    hist_csv_file1 = 'user_location_{}bs_{}app_radius.csv'.format(target_bs,target_app)
    with open(hist_csv_file1, mode='w', newline="") as f:
        save_data.to_csv(f)

    return hist_csv_file1
#
def coordinate_set(neigh_bs_set,df_bs_loc):

    coordinates = [
        (df_bs_loc.loc[f'df{i}05', 'lat'] , df_bs_loc.loc[f'df{i}05', 'Lon']) for i in neigh_bs_set
    ]
    return coordinates

def find_nearest_pair(lat1, lon1, coordinates_set):

    min_distance = float('inf')
    nearest_pair = None
    nearest_pair_indx=None
    # Iterate through each pair of coordinates
    for i in range(len(coordinates_set)):
        lat2, lon2 = coordinates_set[i]
        # Calculate the distance between the coordinates
        loc1 = (lat1, lon1)
        loc2 = (lat2, lon2)
        distance = hs.haversine(loc1, loc2)
        # Check if the distance is the new minimum
        if distance < min_distance:
            min_distance = distance
            nearest_pair = (lat2, lon2)
            nearest_pair_indx=i
    return nearest_pair, nearest_pair_indx, min_distance


def find_potential_bs_for_user(target_bs,target_app,df_user_sets_loc_,df_nei_bs_,df_bs_loc_):

    df_user_sets_loc=df_user_sets_loc_
    df_nei_bs=df_nei_bs_

    user_cnt=df_user_sets_loc['Lon_{}'.format(target_bs)].count()
    result=[]
    if df_nei_bs[f'df{target_bs}05'].count() == 0:
        neighbor_bs_set = [target_bs]
        result=None
        lists_lon_target=df_user_sets_loc[f'Lon_{target_bs}']
        lists_lat_target=df_user_sets_loc[f'lat_{target_bs}']
        df_for_target = pd.DataFrame({f'Lon_{target_bs}': lists_lon_target,
                                      f'lat_{target_bs}': lists_lat_target})
    else:
        df_nei = df_nei_bs[f'df{target_bs}05'].str[2:-2]

        df_nei = df_nei.dropna()
        df_nei = df_nei.astype(int)
        df_nei = df_nei.iloc[0:]
        neighbor_bs_set_exclu_target_bs = df_nei.tolist()  # neighboring set of BS


        df_bs_loc = df_bs_loc_

        coordinates_lst=coordinate_set(neighbor_bs_set_exclu_target_bs,df_bs_loc)

        lists_lat = [[] for _ in range(len(neighbor_bs_set_exclu_target_bs))]
        lists_lon = [[] for _ in range(len(neighbor_bs_set_exclu_target_bs))]
        lists_lat_target=[]
        lists_lon_target=[]
        for k in range(user_cnt):
            k_lon = df_user_sets_loc.loc[k, f'Lon_{target_bs}']
            k_lat=df_user_sets_loc.loc[k, f'lat_{target_bs}']
            # k_loc = (k_lon,k_lat)
            nearest_pair,nearest_pair_indx,distance_km=find_nearest_pair(k_lat,k_lon,coordinates_lst)
           
            val_lat = [k_lat]
            val_lon = [k_lon]
            if distance_km < df_bs_loc.loc[f'df{neighbor_bs_set_exclu_target_bs[nearest_pair_indx]}05', 'Coverage_km']:

                lists_lat[nearest_pair_indx].extend(val_lat)
                lists_lon[nearest_pair_indx].extend(val_lon)
            else:
                lists_lat_target.extend(val_lat)
                lists_lon_target.extend(val_lon)


        for i in range(len(neighbor_bs_set_exclu_target_bs)):
            df_for_neighbor=pd.DataFrame({f'Lon_{neighbor_bs_set_exclu_target_bs[i]}': lists_lon[i],
                        f'lat_{neighbor_bs_set_exclu_target_bs[i]}': lists_lat[i]} )

            result.append(pd.Series(df_for_neighbor[f'Lon_{neighbor_bs_set_exclu_target_bs[i]}']))
            result.append(pd.Series(df_for_neighbor[f'lat_{neighbor_bs_set_exclu_target_bs[i]}']))

        result = pd.concat(result, axis=1)

        df_for_target=pd.DataFrame({f'Lon_{target_bs}':lists_lon_target,
                                f'lat_{target_bs}':lists_lat_target})

    df_finale=pd.concat([df_for_target,result],axis=1)
    df_finale.to_csv(f'user_set_with_potential_{target_bs}bs_{target_app}app.csv')
    return df_finale



