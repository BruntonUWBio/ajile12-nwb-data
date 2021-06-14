import pdb
import pandas as pd
import numpy as np

def project_power(dat, roi_ind, roi_proj_loadpath, rem_bad_chans=True,
                  part_ind=0, bad_chans=[], atlas='aal', elec_dens_thresh=0):
    #Project to ROI's
    df = pd.read_csv(roi_proj_loadpath+atlas+'_'+str(part_ind+1).zfill(2)+'_elecs2ROI.csv')
    chan_ind_vals = np.nonzero(df.transpose().mean().values!=0)[0][1:]

    # Remove bad electrodes by zeroing out their projection values
    if rem_bad_chans:
        if len(bad_chans)==1:
            bad_chans = bad_chans[0]

        df.iloc[bad_chans] = 0
        sum_vals = df.sum(axis=0).values
        for s in range(len(sum_vals)):
            df.iloc[:,s] = df.iloc[:,s]/sum_vals[s]
    
    normalized_weights = np.asarray(df.iloc[chan_ind_vals,roi_ind])
    power_norm = np.dot(normalized_weights, dat[chan_ind_vals-1,:])
    return power_norm