import glob
import natsort
import pandas as pd
import numpy as np
from scipy import interpolate
from tqdm import tqdm as tqdm
from neurodsp.spectral import compute_spectrum

from pynwb import NWBHDF5IO
from ndx_events import LabeledEvents, AnnotatedEventsTable, Events
from spec_utils import project_power

# Set parameters
win_spec_len = 30  # sec
large_win = 30*60  # sec
fs = 500  # Hz
freq_range = [3, 125]  # Hz
sp = '/data1/users/stepeter/mvmt_init/data_spec/'
data_lp = '/data2/users/stepeter/files_nwb/downloads/000055/'
roi_proj_loadpath = '/data1/users/stepeter/mvmt_init/data_release/roi_proj_matlab/'
n_parts = 12  # number of participants

# Load ROI projection matrices
atlas = 'aal'
elec_dens_thresh = 3 #threshold for dipole density
for s in range(n_parts):
    df = pd.read_csv(roi_proj_loadpath+atlas+'_'+str(s+1).zfill(2)+'_elecs2ROI.csv')
    if s==0:
        elec_densities = df.iloc[0]
    else:
        elec_densities += df.iloc[0]
elec_densities = elec_densities/n_parts 
#Select ROI's that have electrode density above threshold
good_rois = np.nonzero(np.asarray(elec_densities)>elec_dens_thresh)[0]
roi_labels = df.columns.tolist()
print('Selected '+str(len(good_rois))+' regions')

selected_rois = np.arange(len(good_rois)).tolist() #0

win_n_samps = int(win_spec_len * fs)
large_win_samps = int(large_win * fs)
freqs = np.arange(freq_range[0],freq_range[1]+1)
n_freqs = len(freqs)

for part_ind in tqdm(range(n_parts)):
    for selected_roi in range(len(good_rois)):
        fids = natsort.natsorted(glob.glob(data_lp+'sub-'+str(part_ind+1).zfill(2)+'/*.nwb'))
        pows_sbj = None
        for j, fid in enumerate(fids):
            io = NWBHDF5IO(fid, mode='r', load_namespaces=False)
            nwb = io.read()

            good_chans = nwb.electrodes['good'][:].astype('int')
            bad_chans = np.nonzero(1-good_chans)[0]

            N_dat = len(nwb.acquisition['ElectricalSeries'].data)
            window_starts_ep = np.arange(0, N_dat, large_win_samps)
            n_windows = len(window_starts_ep)

            for i in tqdm(range(n_windows)):
                end_ind = (window_starts_ep[i]+large_win_samps) if i<(n_windows-1) else -1
                first_elec = nwb.acquisition['ElectricalSeries'].data[window_starts_ep[i]:end_ind, 0]
                if np.sum(np.isnan(first_elec)) == 0:
                    dat = nwb.acquisition['ElectricalSeries'].data[window_starts_ep[i]:end_ind, :].T

                    # Compute power using Welch's method
                    f_welch, spg = compute_spectrum(dat, fs, method='welch',
                                                  avg_type='median', nperseg=win_n_samps,
                                                  f_range=freq_range)

                    # Interpolate power to integer frequencies
                    f = interpolate.interp1d(f_welch, spg)
                    spg_new = f(freqs)

                    # Project power to ROI of interest
                    spg_proj = project_power(spg_new, good_rois[selected_roi], roi_proj_loadpath,
                                             part_ind=part_ind, bad_chans=bad_chans,
                                             atlas=atlas, elec_dens_thresh=elec_dens_thresh)

                    # Append result to final list
                    if pows_sbj is None:
                        pows_sbj = spg_proj[np.newaxis, :].copy()
                    else:
                        pows_sbj = np.concatenate((pows_sbj,
                                                   spg_proj[np.newaxis, :].copy()),
                                                  axis=0)

        # Save power result
        roi_curr = roi_labels[good_rois[selected_roi]][:-2]
        np.save(sp+'P'+str(part_ind+1).zfill(2)+'_'+roi_curr+'.npy',pows_sbj)
        del pows_sbj
