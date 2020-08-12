#!/usr/bin/env python
import time
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dT", default=0.3, help="delta time")
    parser.add_argument("--i", default="incidents.csv", help="input csv file with incidents")
    parser.add_argument("--o", default="counts.csv", help="output csv file with counts")
    args = parser.parse_args()
    dt = float(args.dT)
    input_path = args.i
    output_path = args.o
    print("dT: ",dt, " input: ", input_path, " output: ", output_path)

    def add_to_dict(d, key, val):
        """Add element to a list that has a key in dictionary.
            d - dictionary
            key - list key
            val - value to add to list
       """
        if key in d:
           val_lst = d[key]
           val_lst.append(val)
        else:
           d[key] = [val]

    start_time = time.time()    
    df = pd.read_csv(input_path)

    # Put (id,time) of records that have matching feature values in a dictionary      
    dct = {}
    df_len = len(df)
    for i in range(len(df)):
        row = df.iloc[i,:]
        add_to_dict(dct, (row['feature1'], row['feature2']), (row['id'], row['time']))


    # For every id count ids with time less that the given id has
    cnt_dict = {}
    for tup_lst in dct.values() :
        # Sort list of (id, time) tuples by time
        tup_lst.sort(key = lambda x: x[1], reverse=True) 
        t_len = len(tup_lst)
        for i,tup in enumerate (tup_lst):
            id = tup[0]
            cnt = 0
            k = i+1
            while k < t_len:
                next_tup = tup_lst[k]
                delta = tup[1] - next_tup[1] 
                if delta <= dt:
                    cnt += 1
                k += 1
                
            cnt_dict[id] = cnt

    # Save results
    cnt_df = pd.DataFrame(list(cnt_dict.items()))
    rs = df.merge(cnt_df.rename(columns={0:'id', 1:'cnt'}), on='id')
    rs.to_csv(output_path)
    
    print("Elapsed time (min): ", (time.time() - start_time) / 60)
    
    #print(rs)
    
    
if __name__ == '__main__':
  main()