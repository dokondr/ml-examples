#!/usr/bin/python
# coding: utf-8

"""
Count total distances covered by a self-driving car in autopilot mode and in manual control mode
"""

from urllib.request import urlopen
from math import radians
import numpy as np
import pandas as pd
import sys

def distance(s_lat, s_lng, e_lat, e_lng):
    """  Find distance between two latitude-longitude coordinates with Haversine formula
    
        Args:
            s_lat, s_lng - coordinates of the first point
            e_lat, e_lng - coordinates of the second point
            
        Returns:
            distances in km
    """      
    
    R = 6371 # approximate radius of earth in km
    
    s_lat = np.deg2rad(s_lat)                    
    s_lng = np.deg2rad(s_lng)     
    e_lat = np.deg2rad(e_lat)                       
    e_lng = np.deg2rad(e_lng)  
    
    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(
        e_lat) * np.sin((e_lng - s_lng)/2)**2
    
    return 2 * R * np.arcsin(np.sqrt(d)) 

def get_switch(s):
    """ Get control_switch_on state
    
        Args:
            s - string
            
        Returns:
            time 
            switch state: True/False
    """
    switch, ts = s.split(',')
    _, control = switch.split(':')
    control = control == 'true'
    _, ts = ts.split(':')
    ts = float(ts.replace('}\n',''))
    return(ts, control)

def get_geo(s):
    """Get geo location
    
        Args:
            s - string
            
        Returns:
            time, lat, long
            
    """
    geo, ts = s.split('},')
    _, ts = ts.split(':')
    ts = ts.replace('}\n','')
    lat, lon = geo.split(',')
    _, lon = lon.split(':')
    _, lat = lat.split('"lat":')
    return(float(ts), float(lat), float(lon))

def main():
    
    #url = "https://sdcimages.s3.yandex.net/test_task/data"
    
    url = sys.argv[1]
    print("*** url: ", url)
    
    file  = urlopen(url)

    count = 0
    switch_times = []
    switch_on = []
    geo_times = []
    lats = []
    lons = []
    for string in file: 
        count += 1
        line = string.decode("utf-8")
        if line.find('control_switch_on') > 0: 
            ts, on = get_switch(line)
            switch_times.append(ts)
            switch_on.append(on)
        elif line.find('geo') > 0:
            ts, lat, lon = get_geo(line)
            geo_times.append(ts)
            lats.append(lat)
            lons.append(lon)
        else:
            print("*** Unknown format: ", line)
            break;
            
    print("*** Total lines: ", count)

    switch_on_df = pd.DataFrame({'time':switch_times, 'on':switch_on})
    switch_on_df = switch_on_df.sort_values(by='time').copy()
    print('*** control_switch_on:')
    print(switch_on_df.info())

    print('*** Switch value counts:')
    print(switch_on_df['on'].value_counts())

    geo_df = pd.DataFrame({'time':geo_times, 'lat':lats, 'lon':lons})
    geo_df = geo_df.sort_values(by='time').copy()
    print("*** locations: ")
    print(geo_df.info())

    # Left join locations with `control_switch_on` matching on nearest time
    df = pd.merge_asof(geo_df, switch_on_df.assign(time=switch_on_df["time"].astype(float)), on="time")

    print("*** Left join locations with `control_switch_on` ")
    print(df.info())
    #print(df.head())

    print("*** Switch value counts (joined with locations):\n",df['on'].value_counts())

    no_switch_info = df[df['time'] <= min(switch_on_df['time'].values)].copy()
    print("*** Number of location records for which value of `control_switch_on` is not konown: ",
          len(no_switch_info))

    # Drop records with unknown switch
    df = df.dropna()

    # Calculate the distances between adjacent points
    df['dist'] = distance(df['lat'], df['lon'], 
                          df['lat'].shift(-1), df['lon'].shift(-1))

    total_dist = df['dist'].sum() # total distance
    auto_pilot_on_distance = df.loc[df['on'],'dist'].sum() # with autopilot 
    auto_pilot_off_distance = total_dist - auto_pilot_on_distance

    print("*** Distances: ")
    print("    - On autopilot (control_switch_on = true): {} km ".format(auto_pilot_on_distance))
    print("    - With manual control (control_switch_on = false): {} km".format(auto_pilot_off_distance))


if __name__=="__main__": 
    main() 




