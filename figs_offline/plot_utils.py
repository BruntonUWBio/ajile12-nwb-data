"""Utility functions for plotting.

Subfunctions used in Jupyter notebooks to process and
plot NWB data, including ECoG, pose, movement events,
and behavior labels.


Author
------
Steven Peterson


Modification history
--------------------
02/17/2022 - Add comments and header
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import medfilt
from nilearn import plotting as ni_plt

from pynwb import NWBHDF5IO
from ndx_events import LabeledEvents, AnnotatedEventsTable, Events
from nwbwidgets.utils.timeseries import align_by_times, timeseries_time_to_ind

def prune_clabels(clabels_orig, targeted=False,
                  targ_tlims=[13, 17], first_val=True,
                  targ_label='Eat'):
    '''Modify coarse behavior labels based on whether
    looking at whole day (targeted=False) or specific
    hours (targeted=True). When selecting specific
    hours, can look at either the first (first_val=True)
    or last (first_val=False) label if there are multiple
    overlapping activity labels.'''
    clabels = clabels_orig.copy()
    if not targeted:
        for i in range(len(clabels_orig)):
            lab = clabels_orig.loc[i, 'labels']
            if lab[:5] == 'Block':
                clabels.loc[i, 'labels'] = 'Blocklist'
            elif lab == '':
                clabels.loc[i, 'labels'] = 'Blocklist'
            elif lab not in ['Sleep/rest', 'Inactive']:
                clabels.loc[i, 'labels'] = 'Active'
    else:
        for i in range(len(clabels_orig)):
            lab = clabels_orig.loc[i, 'labels']
            if targ_label in lab.split(', '):
                clabels.loc[i, 'labels'] = targ_label
            else:
                clabels.loc[i, 'labels'] = 'Blocklist'
#             if lab[:5] == 'Block':
#                 clabels.loc[i, 'labels'] = 'Blocklist'
#             elif lab == '':
#                 clabels.loc[i, 'labels'] = 'Blocklist'
#             elif first_val:
#                 clabels.loc[i, 'labels'] = lab.split(', ')[0]
#             else:
#                 clabels.loc[i, 'labels'] = lab.split(', ')[-1]

    if targeted:
        start_val, end_val = targ_tlims[0]*3600, targ_tlims[1]*3600
        clabels = clabels[(clabels['start_time'] >= start_val) &\
                          (clabels['stop_time'] <= end_val)]
        clabels.reset_index(inplace=True)
    uni_labs = np.unique(clabels['labels'].values)
    return clabels, uni_labs


def plot_clabels(clabels, uni_labs, targeted=False, first_val=True,
                 targ_tlims=[13, 17], scale_fact=1/3600,
                 bwidth=0.5, targlab_colind=0):
    '''Plot coarse labels for one recording day.
    Note that the colors for the plots are currently
    pre-defined to work for sub-01 day 4.'''
    # Define colors for each label
    act_cols = plt.get_cmap('Reds')(np.linspace(0.15, 0.85, 5))
    if targeted:
        category_colors = np.array(['w', act_cols[targlab_colind]],
                                   dtype=object)
    else:
        category_colors = np.array([[1, 128/255, 178/255],'dimgray',
                                    'lightgreen','lightskyblue'],
                                   dtype=object)

    # Plot each label as a horizontal bar
    fig, ax = plt.subplots(figsize=(20, 2), dpi=150)
    for i in range(len(uni_labs)):
        lab_inds = np.nonzero(uni_labs[i] == clabels['labels'].values)[0]
        lab_starts = clabels.loc[lab_inds, 'start_time'].values
        lab_stops = clabels.loc[lab_inds, 'stop_time'].values
        lab_widths = lab_stops - lab_starts
        rects = ax.barh(np.ones_like(lab_widths), lab_widths*scale_fact,
                        left=lab_starts*scale_fact,
                        height=bwidth, label=uni_labs[i],
                        color=category_colors[i])
    ax.legend(ncol=len(uni_labs), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    # Define x-axis based on if targeted window or not
    if targeted:
        plt.xlim(targ_tlims)
        targ_tlims_int = [int(val) for val in targ_tlims]
        plt.xticks(targ_tlims_int)
        ax.set_xticklabels(['{}:00'.format(targ_tlims_int[0]),
                            '{}:00'.format(targ_tlims_int[-1])])
    else:
        plt.xlim([0, 24])
        plt.xticks([0, 12, 24])
        ax.set_xticklabels(['0:00', '12:00', '0:00'])

    # Remove border lines and show plot
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return fig


def clabel_table_create(common_acts, n_parts=12,
                        data_lp='/data2/users/stepeter/files_nwb/downloads/000055/'):
    '''Create table of coarse label durations across participants.
    Labels to include in the table are specified by common_acts.'''
    vals_all = np.zeros([n_parts, len(common_acts)+1])
    for part_ind in range(n_parts):
        fids = natsort.natsorted(glob.glob(data_lp+'sub-'+str(part_ind+1).zfill(2)+'/*.nwb'))
        for fid in fids:
            io = NWBHDF5IO(fid, mode='r', load_namespaces=False)
            nwb = io.read()

            curr_labels = nwb.intervals['epochs'].to_dataframe()
            durations = (curr_labels.loc[:,'stop_time'].values - curr_labels.loc[:,'start_time'].values)

            # Add up durations of each label
            for s, curr_act in enumerate(common_acts):
                for i, curr_label in enumerate(curr_labels['labels'].tolist()):
                    if curr_act in curr_label.split(', '):
                        vals_all[part_ind, s] += durations[i]/3600

            # Add up total durations of selected labels (avoid double counting)
            for i, curr_label in enumerate(curr_labels['labels'].tolist()):
                in_lab_grp = False
                for sub_lab in curr_label.split(', '):
                    if sub_lab in common_acts:
                        in_lab_grp = True
                vals_all[part_ind, -1] += durations[i]/3600 if in_lab_grp else 0

    # Make final table/dataframe 
    common_acts_col = [val.lstrip('Blocklist (').rstrip(')') for val in common_acts]
    df_all = pd.DataFrame(vals_all.round(1), index=['P'+str(val+1).zfill(2) for val in range(n_parts)],
                          columns=common_acts_col+['Total'])
    return df_all


def identify_elecs(group_names):
    '''Determine surface v. depth ECoG electrodes'''
    is_surf = []
    for label in group_names:
        if 'grid' in label.lower():
            is_surf.append(True)
        elif label.lower() in ['mhd', 'latd', 'lmtd', 'ltpd']:
            is_surf.append(True)  # special cases
        elif (label.lower() == 'ahd') & ('PHD' not in group_names):
            is_surf.append(True)  # special case
        elif 'd' in label.lower():
            is_surf.append(False)
        else:
            is_surf.append(True)
    return np.array(is_surf)


def load_data_characteristics(lp='/data2/users/stepeter/files_nwb/downloads/000055/',
                              nparts=12):
    '''Load data characteristics including the number of
    good and total ECoG electrodes, hemisphere implanted,
    and number of recording days for each participant.'''
    n_elecs_tot, n_elecs_good = [], []
    rec_days, hemis, n_elecs_surf_tot, n_elecs_depth_tot = [], [], [], []
    n_elecs_surf_good, n_elecs_depth_good = [], []
    for part_ind in range(nparts):
        cur_t = 0
        fids = natsort.natsorted(glob.glob(lp+'sub-'+str(part_ind+1).zfill(2)+'/*.nwb'))
        rec_days.append(len(fids))
        for fid in fids[:1]:
            io = NWBHDF5IO(fid, mode='r', load_namespaces=False)
            nwb = io.read()

            # Determine good/total electrodes
            n_elecs_good.append(np.sum(nwb.electrodes['good'][:]))
            n_elecs_tot.append(len(nwb.electrodes['good'][:]))

            # Determine implanted hemisphere
            c_wrist = nwb.processing['behavior'].data_interfaces['ReachEvents'].description[0]
            hemis.append('L' if c_wrist == 'r' else 'R')

            # Determine surface vs. depth electrode count
            is_surf = identify_elecs(nwb.electrodes['group_name'][:])
            n_elecs_surf_tot.append(np.sum(is_surf))
            n_elecs_depth_tot.append(np.sum(1-is_surf))
            n_elecs_surf_good.append(np.sum(nwb.electrodes['good'][is_surf.nonzero()[0]]))
            n_elecs_depth_good.append(np.sum(nwb.electrodes['good'][(1-is_surf).nonzero()[0]]))

    part_nums = [val+1 for val in range(nparts)]
    part_ids = ['P'+str(val).zfill(2) for val in part_nums]
    
    return [rec_days, hemis, n_elecs_surf_tot, n_elecs_surf_good,
            n_elecs_depth_tot, n_elecs_depth_good, part_nums,
            part_ids, n_elecs_good, n_elecs_tot]


def plot_ecog_descript(n_elecs_tot, n_elecs_good, part_ids,
                       lp='/data2/users/stepeter/files_nwb/downloads/000055/',
                       nparts=12, allLH=False, nrows=3,
                       chan_labels='all', width=7, height=3):
    '''Plot ECoG electrode positions and identified noisy
    electrodes side by side.'''
    fig = plt.figure(figsize=(width*3, height*3), dpi=150)
    
    # First subplot: electrode locations
    ncols = nparts//nrows
    gs = gridspec.GridSpec(nrows=nrows, 
                           ncols=ncols, #+2, 
                           figure=fig, 
                           width_ratios= [width/ncols]*ncols, #[width/ncols/2]*ncols+[width/10, 4*width/10],
                           height_ratios= [height/nrows]*nrows,
                           wspace=0, hspace=-.5
                          )
    ax = [None]*(nparts) #+1)

    for part_ind in range(nparts):
        # Load NWB data file
        fids = natsort.natsorted(glob.glob(lp+'sub-'+str(part_ind+1).zfill(2)+'/*.nwb'))
        io = NWBHDF5IO(fids[0], mode='r', load_namespaces=False)
        nwb = io.read()

        # Determine hemisphere to display
        if allLH:
            sides_2_display ='l'
        else:
            average_xpos_sign = np.nanmean(nwb.electrodes['x'][:])
            sides_2_display = 'r' if average_xpos_sign>0 else 'l'

        # Run electrode plotting function
        ax[part_ind] = fig.add_subplot(gs[part_ind//ncols, part_ind%ncols])
        plot_ecog_electrodes_mni_from_nwb_file(nwb,chan_labels,num_grid_chans=64,node_size=50,
                                               colors='silver',alpha=.9,sides_2_display=sides_2_display,
                                               node_edge_colors='k',edge_linewidths=1.5,
                                               ax_in=ax[part_ind],allLH=allLH)
#         ax[part_ind].text(-0.2,0.1,'P'+str(part_ind+1).zfill(2), fontsize=8)
#     fig.text(0.1, 0.91, '(a) ECoG electrode positions', fontsize=10)
    
    # Second subplot: noisy electrodes per participant
#     ax[-1] = fig.add_subplot(gs[:, -1])
#     ax[-1].bar(part_ids,n_elecs_tot,color='lightgrey')
#     ax[-1].bar(part_ids,n_elecs_good,color='dimgrey')
#     ax[-1].spines['right'].set_visible(False)
#     ax[-1].spines['top'].set_visible(False)
#     ax[-1].set_xticklabels(part_ids, rotation=45)
#     ax[-1].legend(['Total','Good'], frameon=False, fontsize=8)
#     ax[-1].tick_params(labelsize=9)
#     ax[-1].set_ylabel('Number of electrodes', fontsize=9, labelpad=0)
#     ax[-1].set_title('(b) Total/good electrodes per participant',
#                     fontsize=10)
    plt.show()
    return fig


def plot_ecog_electrodes_mni_from_nwb_file(nwb_dat,chan_labels='all',num_grid_chans=64,colors=None,node_size=50,
                                           figsize=(16,6),sides_2_display='auto',node_edge_colors=None,
                                           alpha=0.5,edge_linewidths=3,ax_in=None,rem_zero_chans=False,
                                           allLH=False,zero_rem_thresh=.99,elec_col_suppl=None):
    """
    Plots ECoG electrodes from MNI coordinate file (only for specified labels)
    NOTE: If running in Jupyter, use '%matplotlib inline' instead of '%matplotlib notebook'
    """ 
    #Load channel locations
    chan_info = nwb_dat.electrodes.to_dataframe()
    
    #Create dataframe for electrode locations
    if chan_labels== 'all':
        locs = chan_info.loc[:,['x','y','z']]
    elif chan_labels== 'allgood':
        locs = chan_info.loc[:,['x','y','z','good']]
    else:
        locs = chan_info.loc[chan_labels,['x','y','z']]
    if (colors is not None):
        if (locs.shape[0]>len(colors)) & isinstance(colors, list):
            locs = locs.iloc[:len(colors),:]
#     locs.rename(columns={'X':'x','Y':'y','Z':'z'}, inplace=True)
    chan_loc_x = chan_info.loc[:,'x'].values
    
    #Remove NaN electrode locations (no location info)
    nan_drop_inds = np.nonzero(np.isnan(chan_loc_x))[0]
    locs.dropna(axis=0,inplace=True) #remove NaN locations
    if (colors is not None) & isinstance(colors, list):
        colors_new,loc_inds_2_drop = [],[]
        for s,val in enumerate(colors):
            if not (s in nan_drop_inds):
                colors_new.append(val)
            else:
                loc_inds_2_drop.append(s)
        colors = colors_new.copy()
        
        if elec_col_suppl is not None:
            loc_inds_2_drop.reverse() #go from high to low values
            for val in loc_inds_2_drop:
                del elec_col_suppl[val]
    
    if chan_labels=='allgood':
        goodChanInds = chan_info.loc[:,'good',:]
        inds2drop = np.nonzero(locs['good']==0)[0]
        locs.drop(columns=['good'],inplace=True)
        locs.drop(locs.index[inds2drop],inplace=True)
        
        if colors is not None:
            colors_new,loc_inds_2_drop = [],[]
            for s,val in enumerate(colors):
                if not (s in inds2drop):
#                     np.all(s!=inds2drop):
                    colors_new.append(val)
                else:
                    loc_inds_2_drop.append(s)
            colors = colors_new.copy()
            
            if elec_col_suppl is not None:
                loc_inds_2_drop.reverse() #go from high to low values
                for val in loc_inds_2_drop:
                    del elec_col_suppl[val]
    
    if rem_zero_chans:
        #Remove channels with zero values (white colors)
        colors_new,loc_inds_2_drop = [],[]
        for s,val in enumerate(colors):
            if np.mean(val)<zero_rem_thresh:
                colors_new.append(val)
            else:
                loc_inds_2_drop.append(s)
        colors = colors_new.copy()
        locs.drop(locs.index[loc_inds_2_drop],inplace=True)
        
        if elec_col_suppl is not None:
            loc_inds_2_drop.reverse() #go from high to low values
            for val in loc_inds_2_drop:
                del elec_col_suppl[val]
    
    #Decide whether to plot L or R hemisphere based on x coordinates
    if len(sides_2_display)>1:
        N,axes,sides_2_display = _setup_subplot_view(locs,sides_2_display,figsize)
    else:
        N = 1
        axes = ax_in
        if allLH:
            average_xpos_sign = np.mean(np.asarray(locs['x']))
            if average_xpos_sign>0:
                locs['x'] = -locs['x']
            sides_2_display ='l'
                
    if colors is None:
        colors = list()
    
    #Label strips/depths differently for easier visualization (or use defined color list)
    if len(colors)==0:
        for s in range(locs.shape[0]):
            if s>=num_grid_chans:
                colors.append('r')
            else:
                colors.append('b')
    
    if elec_col_suppl is not None:
        colors = elec_col_suppl.copy()
    
    #Rearrange to plot non-grid electrode first
    if num_grid_chans>0: #isinstance(colors, list):
        locs2 = locs.copy()
        locs2['x'] = np.concatenate((locs['x'][num_grid_chans:],locs['x'][:num_grid_chans]),axis=0)
        locs2['y'] = np.concatenate((locs['y'][num_grid_chans:],locs['y'][:num_grid_chans]),axis=0)
        locs2['z'] = np.concatenate((locs['z'][num_grid_chans:],locs['z'][:num_grid_chans]),axis=0)
        
        if isinstance(colors, list):
            colors2 = colors.copy()
            colors2 = colors[num_grid_chans:]+colors[:num_grid_chans]
        else:
            colors2 = colors
    else:
        locs2 = locs.copy()
        if isinstance(colors, list):
            colors2 = colors.copy()
        else:
            colors2 = colors #[colors for i in range(locs2.shape[0])]
    
    #Plot the result
    _plot_electrodes(locs2,node_size,colors2,axes,sides_2_display,
                     N,node_edge_colors,alpha,edge_linewidths)


def _plot_electrodes(locs,node_size,colors,axes,sides_2_display,N,node_edge_colors,
                     alpha,edge_linewidths,marker='o'):
    """
    Handles plotting of electrodes.
    """
    if N==1:
        ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                               node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,
                                            'linewidths':edge_linewidths,'marker': marker},
                               node_size=node_size, node_color=colors,axes=axes,display_mode=sides_2_display)
    elif sides_2_display=='yrz' or sides_2_display=='ylz':
        colspans=[5,6,5] #different sized subplot to make saggital view similar size to other two slices
        current_col=0
        total_colspans=int(np.sum(np.asarray(colspans)))
        for ind,colspan in enumerate(colspans):
            axes[ind]=plt.subplot2grid((1,total_colspans), (0,current_col), colspan=colspan, rowspan=1)
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                                   node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,
                                                'linewidths':edge_linewidths,'marker': marker},
                                   node_size=node_size, node_color=colors,
                                   axes=axes[ind],display_mode=sides_2_display[ind])
            current_col+=colspan
    else:
        for i in range(N):
            ni_plt.plot_connectome(np.eye(locs.shape[0]), locs, output_file=None,
                                   node_kwargs={'alpha': alpha, 'edgecolors': node_edge_colors,
                                                'linewidths':edge_linewidths,'marker': marker},
                                   node_size=node_size, node_color=colors,
                                   axes=axes[i],display_mode=sides_2_display[i])


def plot_ecog_pow(lp, rois_plt, freq_range, sbplt_titles,
                  part_id='P01', n_parts=12, nrows=2, ncols=4,
                  figsize=(7,4)):
    '''Plot ECoG projected spectral power.'''
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    
    # Plot projected power for all participants
    fig, ax = _ecog_pow_group(fig, ax, lp, rois_plt, freq_range, sbplt_titles,
                              n_parts, nrows, ncols, row_ind=0)

    # Plot projected power for 1 participant
    fig, ax = _ecog_pow_single(fig, ax, lp, rois_plt, freq_range, sbplt_titles,
                               n_parts, nrows, ncols, row_ind=1,
                               part_id=part_id)

    fig.tight_layout()
    plt.show()

    
def _ecog_pow_group(fig, ax, lp, rois_plt, freq_range, sbplt_titles,
                    n_parts=12, nrows=2, ncols=4, row_ind=0):
    '''Plot projected power for all participants.'''
    freqs_vals = np.arange(freq_range[0],freq_range[1]+1).tolist()
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.1)
    power, freqs, parts = [], [], []
    n_wins_sbj = []
    for k,roi in enumerate(rois_plt):
        power_roi, freqs_roi, parts_roi = [], [], []
        for j in range(n_parts):
            dat = np.load(lp+'P'+str(j+1).zfill(2)+'_'+roi+'.npy')
            dat = 10*np.log10(dat)
            for i in range(dat.shape[0]):
                power_roi.extend(dat[i,:].tolist())
                freqs_roi.extend(freqs_vals)
                parts_roi.extend(['P'+str(j+1).zfill(2)]*len(freqs_vals))
            if k==0:
                n_wins_sbj.append(dat.shape[0])
        power.extend(power_roi)
        freqs.extend(freqs_roi)
        parts.extend(parts_roi)

        parts_uni = np.unique(np.asarray(parts_roi))[::-1].tolist()
        df_roi = pd.DataFrame({'Power': power_roi, 'Freqs': freqs_roi, 'Parts': parts_roi})
        col = k%ncols
        ax_curr = ax[row_ind,col] if nrows > 1 else ax[col]
        leg = False # 'brief' if k==3 else False
        sns.lineplot(data=df_roi, x="Freqs", y="Power", hue="Parts",
                     ax=ax_curr, ci='sd', legend=leg, palette=['darkgray']*len(parts_uni),
                     hue_order=parts_uni) # palette='Blues'
    #     ax_curr.set_xscale('log')
        ax_curr.set_xlim(freq_range)
        ax_curr.set_ylim([-20,30])
        ax_curr.spines['right'].set_visible(False)
        ax_curr.spines['top'].set_visible(False)
        ax_curr.set_xlim(freq_range)
        ax_curr.set_xticks([freq_range[0]]+np.arange(20,101,20).tolist()+[freq_range[1]])
        ylab = ''  # '' if k%ncols > 0 else 'Power\n(dB)'  # 10log(uV^2)
        xlab = ''  # 'Frequency (Hz)' if k//ncols==(nrows-1) else ''
        ax_curr.set_ylabel(ylab, rotation=0, labelpad=15, fontsize=9)
        ax_curr.set_xlabel(xlab, fontsize=9)
        if k%ncols > 0:
            l_yticks = len(ax_curr.get_yticklabels())
            ax_curr.set_yticks(ax_curr.get_yticks().tolist())
            ax_curr.set_yticklabels(['']*l_yticks)
        ax_curr.tick_params(axis='both', which='major', labelsize=8)
        ax_curr.set_title(sbplt_titles[k], fontsize=9)
    return fig, ax


def _ecog_pow_single(fig, ax, lp, rois_plt, freq_range, sbplt_titles,
                     n_parts=12, nrows=2, ncols=4, row_ind=1, part_id='P01'):
    '''Plot projected power for a single participant.'''
    part_id = 'P01'
    freqs_vals = np.arange(freq_range[0],freq_range[1]+1).tolist()
    power, freqs, parts = [], [], []
    n_wins_sbj = []
    for k,roi in enumerate(rois_plt):
        power_roi, freqs_roi, parts_roi = [], [], []

        dat = np.load(lp+part_id+'_'+roi+'.npy')
        dat = 10*np.log10(dat)
        for i in range(dat.shape[0]):
            power_roi.extend(dat[i,:].tolist())
            freqs_roi.extend(freqs_vals)
            parts_roi.extend([i]*len(freqs_vals))
        if k==0:
            n_wins_sbj.append(dat.shape[0])
        power.extend(power_roi)
        freqs.extend(freqs_roi)
        parts.extend(parts_roi)

        parts_uni = np.unique(np.asarray(parts_roi))[::-1].tolist()
        df_roi = pd.DataFrame({'Power': power_roi, 'Freqs': freqs_roi, 'Parts': parts_roi})
        col = k%ncols
        ax_curr = ax[row_ind,col] if nrows > 1 else ax[col]
        leg = False # 'brief' if k==3 else False
        sns.lineplot(data=df_roi, x="Freqs", y="Power", hue="Parts",
                     ax=ax_curr, ci=None, legend=leg, palette=['darkgray']*len(parts_uni),
                     hue_order=parts_uni, linewidth=0.2) # palette='Blues'
        ax_curr.set_xlim(freq_range)
        ax_curr.set_ylim([-20,30])
        ax_curr.spines['right'].set_visible(False)
        ax_curr.spines['top'].set_visible(False)
        ax_curr.set_xlim(freq_range)
        ax_curr.set_xticks([freq_range[0]]+np.arange(20,101,20).tolist()+[freq_range[1]])
        ylab = ''  # '' if k%ncols > 0 else 'Power\n(dB)'  # 10log(uV^2)
        xlab = ''  # 'Frequency (Hz)' if k//ncols==(nrows-1) else ''
        ax_curr.set_ylabel(ylab, rotation=0, labelpad=15, fontsize=9)
        ax_curr.set_xlabel(xlab, fontsize=9)
        if k%ncols > 0:
            l_yticks = len(ax_curr.get_yticklabels())
            ax_curr.set_yticks(ax_curr.get_yticks().tolist())
            ax_curr.set_yticklabels(['']*l_yticks)
        ax_curr.tick_params(axis='both', which='major', labelsize=8)
        ax_curr.set_title(sbplt_titles[k], fontsize=9)
    return fig, ax


def plot_dlc_recon_errs(fig, ax):
    '''Plots DeepLabCut reconstruction errors on training and heldout
    images. This information is not present in the NWB files.'''
    # DLC reconstruction errors [train set, holdout set]
    sbj_d = {'P01': [1.45, 4.27], 'P02': [1.44, 3.58],
             'P03': [1.58, 6.95], 'P04': [1.63, 6.02],
             'P05': [1.43, 3.42], 'P06': [1.43, 6.63],
             'P07': [1.51, 5.45], 'P08': [1.84, 10.35],
             'P09': [1.4, 4.05], 'P10': [1.48, 7.59],
             'P11': [1.51, 5.45], 'P12': [1.52, 4.73]}

    train_err = [val[0] for key, val in sbj_d.items()]
    test_err = [val[1] for key, val in sbj_d.items()]

    nsbjs = len(train_err)
    sbj_nums = [val+1 for val in range(nsbjs)]
    sbj = ['P'+str(val).zfill(2) for val in sbj_nums]
    
    # Create plot
    
    ax.bar(sbj,train_err,color='dimgrey')
    ax.bar(sbj,test_err,color='lightgrey')
    ax.bar(sbj,train_err,color='dimgrey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels(sbj, rotation=45)
    ax.legend(['Train set','Holdout set'], frameon=False, fontsize=8)
    ax.tick_params(labelsize=9)
    ax.set_ylabel('Reconstruction error (pixels)')
    ax.set_title('(a) Pose estimation model errors',
                 fontsize=10)


def plot_wrist_trajs(fig, ax, lp, base_start=-1.5, base_end=-1,
                     before=3, after=3, fs_video=30, n_parts=12):
    '''Plot contralateral wrist trajectories during move onset events.'''
    df_pose, part_lst = _get_wrist_trajs(lp, base_start, base_end, before, after,
                               fs_video, n_parts)
    
    df_pose_orig = df_pose.copy()
    df_pose = df_pose_orig.loc[df_pose['Contra']=='contra', :]

    # Set custom color palette
    sns.set_palette(sns.color_palette(["gray"]))

    uni_sbj = np.unique(np.asarray(part_lst))

    for j in range(n_parts):
        sns.lineplot(x="Time",y="Displ",data=df_pose[df_pose['Sbj']==uni_sbj[j]],ax=ax,
                     linewidth=1.5,hue='Contra',legend=False,
                     estimator=np.median, ci=95)

    ax.set_ylim([0,60])
    ax.set_xlim([-0.5,1.5])
    ax.set_xticks([-0.5,0,0.5,1,1.5])
    ax.set_ylabel('Displacement (px)', fontsize=9)
    ax.set_xlabel('Time (sec)', fontsize=9)
    sns.set_style("ticks")
    sns.despine()
    ax.axvline(0, linewidth=1.5, color="black", linestyle="--")
    ax.set_title('(b) Contralateral wrist trajectories during move events',
                 fontsize=10)


def _get_wrist_trajs(lp, base_start=-1.5, base_end=-1,
                     before=3, after=3, fs_video=30,
                     n_parts=12):
    '''Load in wrist trajectories around move onset events.'''
    displ_lst, part_lst, time_lst, pose_lst = [], [], [], []
    for pat in range(n_parts):
        fids = natsort.natsorted(glob.glob(lp+'sub-'+str(pat+1).zfill(2)+'/*.nwb'))
        for i, fid in enumerate(fids):
            io = NWBHDF5IO(fid, mode='r', load_namespaces=False)
            nwb_file = io.read()

            # Segment data
            events = nwb_file.processing["behavior"].data_interfaces["ReachEvents"]
            times = events.timestamps[:]
            starts = times - before
            stops = times + after

            # Get event hand label
            contra_arm = events.description
            contra_arm = map(lambda x: x.capitalize(), contra_arm.split("_"))
            contra_arm = list(contra_arm)
            contra_arm = "_".join(contra_arm)
            ipsi_arm = 'R'+contra_arm[1:] if contra_arm[0] == 'L' else 'L'+contra_arm[1:]

            reach_lab = ['contra', 'ipsi']
            for k, reach_arm in enumerate([contra_arm, ipsi_arm]):
                spatial_series = nwb_file.processing["behavior"].data_interfaces["Position"][reach_arm]
                ep_dat = align_by_times(spatial_series, starts, stops)
                ep_dat_mag = np.sqrt(np.square(ep_dat[...,0]) + np.square(ep_dat[...,1]))

                # Interpolate and median filter
                for j in range(ep_dat_mag.shape[0]):
                    df_mag = pd.DataFrame(ep_dat_mag[j,:])
                    df_mag = df_mag.interpolate(method='pad')
                    tmp_val = df_mag.values.copy().flatten() #medfilt(df_mag.values, kernel_size=31)
                    df_mag = pd.DataFrame(tmp_val[::-1])
                    df_mag = df_mag.interpolate(method='pad')
                    ep_dat_mag[j,:] = medfilt(df_mag.values.copy().flatten()[::-1], kernel_size=31)

                zero_ind = timeseries_time_to_ind(spatial_series, before)
                base_start_ind = timeseries_time_to_ind(spatial_series, base_start+before)
                base_end_ind = timeseries_time_to_ind(spatial_series, base_end+before)
                n_tpoints = ep_dat_mag.shape[1]
                t_vals = np.arange(n_tpoints)/fs_video - before

                # Subtract baseline from position data
                for j in range(ep_dat_mag.shape[0]):
                    curr_magnitude = ep_dat_mag[j,:]
                    curr_magnitude = np.abs(curr_magnitude - \
                                            np.mean(curr_magnitude[base_start_ind:base_end_ind]))
                    curr_magnitude[np.isnan(curr_magnitude)] = 0
                    displ_lst.extend(curr_magnitude.tolist())
                    part_lst.extend(['P'+str(pat+1).zfill(2)]*n_tpoints)
                    time_lst.extend(t_vals.tolist())
                    pose_lst.extend([reach_lab[k]]*n_tpoints)

    df_pose = pd.DataFrame({'Displ': displ_lst, 'Sbj': part_lst,
                            'Time': time_lst, 'Contra': pose_lst})
    return df_pose, part_lst
