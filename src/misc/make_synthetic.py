import pandas as pd


import numpy as np
import random
np.random.seed(1000)
random.seed(1000)

def make_synthetic_data(tgi_path, OS_path, ptnm_path):


    tgi_df = pd.read_csv(tgi_path)
    OS_df = pd.read_csv(OS_path)
    ptnm_df = pd.read_csv(ptnm_path)

    # randomize NMID
    unique_nmid = pd.unique(tgi_df.NMID)
    print(unique_nmid)

    random_ids = random.sample(range(len(unique_nmid)), len(unique_nmid))
    mapping = dict(zip(unique_nmid, random_ids))

    # before vs after
    print(tgi_df.NMID[:15])
    tgi_df.NMID = tgi_df.NMID.map(mapping)
    print(tgi_df.NMID[:15])

    # also need to make random mapping of PTNM to random ID, 
     # randomize NMID
    unique_ptnm = pd.unique(ptnm_df.PTNM)
    print(unique_ptnm)
    random_ptnm = random.sample(range(min(unique_ptnm), max(unique_ptnm)), len(unique_ptnm))
    ptnm_mapping = dict(zip(unique_ptnm, random_ptnm))

    # change the OS df and 
    print(OS_df.PTNM[:15])
    OS_df.PTNM = OS_df.PTNM.map(ptnm_mapping)
    print(OS_df.PTNM[:15])

    ptnm_df.PTNM = ptnm_df.PTNM.map(ptnm_mapping)
    ptnm_df.NMID = ptnm_df.NMID.map(mapping)

    # perturb the SLd values by 
    tgi_df.SLD = tgi_df.SLD.apply(lambda x: x + np.random.normal(loc=0, scale=0.05*max(x, -1 * x)))
    OS_df.OS = OS_df.OS.apply(lambda x: x + np.random.normal(loc=0, scale=0.05*(max((x, -1 * x)))))


    tgi_df.to_csv("assets/data/tgi_synthetic.csv")
    OS_df.to_csv("assets/data/OS_synthetic.csv")
    ptnm_df.to_csv("assets/data/pt_data_synthetic.csv")









if __name__=="__main__":
    tgi_path = "assets/data/tgi.csv"
    OS_path = "assets/data/TGIOS_Data_IMPower150.csv"
    ptnm_path = "assets/data/IMPower150_tumor_20180420.csv"
    make_synthetic_data(tgi_path, OS_path, ptnm_path)

