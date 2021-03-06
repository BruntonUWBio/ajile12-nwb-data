"""Compute ECoG power spectra and save files for later plotting.

Computes spectral power of whole-day ECoG recordings, applying Welch's
method to compute power. Spectral power is then saved to NPY files for
later plotting. Use when streaming data directly from DANDI instead of
running on local NWB files.


Author
------
Steven Peterson


Modification history
--------------------
02/16/2022 - Add comments and header
07/16/2021 - Created script


License
-------
Copyright (c) 2020 CatalystNeuro
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import glob
import natsort
import numpy as np
from scipy import interpolate
from tqdm import tqdm as tqdm
from neurodsp.spectral import compute_spectrum

from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
from ndx_events import LabeledEvents, AnnotatedEventsTable, Events
from spec_utils import project_power, proj_mat_compute

# Set parameters
win_spec_len = 30  # sec
large_win = 30*60  # sec
fs = 500  # Hz
freq_range = [3, 125]  # Hz
sp = '/data1/users/stepeter/mvmt_init/data_spec/'
data_lp = '/data2/users/stepeter/files_nwb/downloads/000055/'
hgrid_fid = '/home/stepeter/AJILE/ajile12-nwb-data/headGrid.mat'
aal_fid = '/home/stepeter/AJILE/ajile12-nwb-data/aal_rois.mat'
n_parts = 12  # number of participants

# Determine all file paths
with DandiAPIClient() as client:
    paths = []
    for file in client.get_dandiset("000055", "draft").get_assets_under_path(''):
        paths.append(file.path)
paths = natsort.natsorted(paths)

# Create ROI projection matrices
elec_dens_thresh = 3 #threshold for dipole density
proj_mats = []
for s in range(n_parts):
    fid = [val for val in paths if 'sub-'+str(s+1).zfill(2) in val][0]
    with DandiAPIClient() as client:
        asset = client.get_dandiset("000055", "draft").get_asset_by_path(fid)
        s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

    io = NWBHDF5IO(s3_path, mode='r', load_namespaces=True, driver='ros3')
    nwb = io.read()
    elec_locs = np.vstack((nwb.electrodes['x'][:],
                           nwb.electrodes['y'][:],
                           nwb.electrodes['z'][:])).T

    elec_locs[elec_locs[:,0]>0, 0] = -elec_locs[elec_locs[:,0]>0, 0] # flip all electrodes to left hemisphere
    
    keep_inds = (1-np.isnan(elec_locs[:,0])).nonzero()[0]
    elec_locs = elec_locs[keep_inds,:]  # remove NaN electrode locations
    good_chans = nwb.electrodes['good'][:].astype('int')
    bad_chans = np.nonzero(1-good_chans[keep_inds])[0]
    
    tot_elec_density, weight_mat, roi_labels = proj_mat_compute(elec_locs, hgrid_fid,
                                                                fwhm=20, bad_chans=bad_chans,
                                                                aal_fid=aal_fid)
    if s==0:
        elec_densities = tot_elec_density.copy()
    else:
        elec_densities += tot_elec_density.copy()
    proj_mats.append(weight_mat)
        
elec_densities = elec_densities/n_parts
#Select ROI's that have electrode density above threshold
good_rois = np.nonzero(elec_densities>elec_dens_thresh)[0]
proj_mats = np.asarray(proj_mats)
print('Selected '+str(len(good_rois))+' regions')

selected_rois = np.arange(len(good_rois)).tolist() #0

win_n_samps = int(win_spec_len * fs)
large_win_samps = int(large_win * fs)
freqs = np.arange(freq_range[0],freq_range[1]+1)
n_freqs = len(freqs)

for part_ind in tqdm(range(n_parts)):
    for selected_roi in range(len(good_rois)):
        fids = [val for val in paths if 'sub-'+str(part_ind+1).zfill(2) in val]
        pows_sbj = None
        for j, fid in enumerate(fids):
            with DandiAPIClient() as client:
                asset = client.get_dandiset("000055", "draft").get_asset_by_path(fid)
                s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

            io = NWBHDF5IO(s3_path, mode='r', load_namespaces=True, driver='ros3')
            nwb = io.read()

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
                    spg_proj = project_power(spg_new, proj_mats[part_ind], good_rois[selected_roi])

                    # Append result to final list
                    if pows_sbj is None:
                        pows_sbj = spg_proj[np.newaxis, :].copy()
                    else:
                        pows_sbj = np.concatenate((pows_sbj,
                                                   spg_proj[np.newaxis, :].copy()),
                                                  axis=0)

        # Save power result
        roi_curr = roi_labels[good_rois[selected_roi]][:-2]
        np.save(sp+'P'+str(part_ind+1).zfill(2)+'_'+roi_curr+'_new.npy',pows_sbj)
        del pows_sbj
