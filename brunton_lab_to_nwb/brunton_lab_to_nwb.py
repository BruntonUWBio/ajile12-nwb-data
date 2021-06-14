import os
import re
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from h5py import File
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.data_utils import DataChunkIterator
from lazy_ops import DatasetView
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.file import Subject, TimeIntervals
from ndx_events import Events

SPECIAL_CHANNELS = (b'EOGL', b'EOGR', b'ECGL', b'ECGR')


def run_conversion(
        fpath_in='/Volumes/easystore5T/data/Brunton/subj_01_day_4.h5',
        fpath_out='/Volumes/easystore5T/data/Brunton/subj_01_day_4.nwb',
        events_path='C:/Users/micha/Desktop/Brunton Lab Data/event_times.csv',
        r2_path='C:/Users/micha/Desktop/Brunton Lab Data/full_model_r2.npy',
        coarse_events_path='C:/Users/micha/Desktop/Brunton Lab Data/coarse_labels/coarse_labels',
        reach_features_path='C:/Users/micha/Desktop/Brunton Lab Data/behavioral_features.csv',
        elec_loc_labels_path='elec_loc_labels.csv',
        special_chans=SPECIAL_CHANNELS,
        session_description='no description'
):
    print(f"Converting {fpath_in}...")
    fname = os.path.split(os.path.splitext(fpath_in)[0])[1]
    _, subject_id, _, session = fname.split('_')

    file = File(fpath_in, 'r')

    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid.uuid4()),
        session_start_time=datetime.fromtimestamp(file['start_timestamp'][()]),
        subject=Subject(subject_id=subject_id, species="Homo sapiens"),
        session_id=session
    )

    # extract electrode groups
    file_elec_col_names = file['chan_info']['axis1'][:]
    elec_data = file['chan_info']['block0_values']

    re_exp = re.compile("([ a-zA-Z]+)([0-9]+)")

    channel_labels_dset = file['chan_info']['axis0']

    group_names, group_nums = [], []
    for i, bytes_ in enumerate(channel_labels_dset):
        if bytes_ not in special_chans:
            str_ = bytes_.decode()
            res = re_exp.match(str_).groups()
            group_names.append(res[0])
            group_nums.append(int(res[1]))

    is_elec = ~np.isin(channel_labels_dset, special_chans)

    dset = DatasetView(file['dataset']).lazy_transpose()

    # add special channels
    for kwargs in (
            dict(
                name='EOGL',
                description='Electrooculography for tracking saccades - left',
            ),
            dict(
                name='EOGR',
                description='Electrooculography for tracking saccades - right',
            ),
            dict(
                name='ECGL',
                description='Electrooculography for tracking saccades - left',
            ),
            dict(
                name='ECGR',
                description='Electrooculography for tracking saccades - right',
            )
    ):
        if kwargs['name'].encode() in channel_labels_dset:
            nwbfile.add_acquisition(
                TimeSeries(
                    rate=file['f_sample'][()],
                    conversion=np.nan,
                    unit='V',
                    data=dset[:, list(channel_labels_dset).index(kwargs['name'].encode())],
                    **kwargs
                )
            )

    # add electrode groups
    df = pd.read_csv(elec_loc_labels_path)
    df_subject = df[df['subject_ID'] == 'subj' + subject_id]
    electrode_group_descriptions = {row['label']: row['long_name'] for _, row in df_subject.iterrows()}

    groups_map = dict()
    for group_name, group_description in electrode_group_descriptions.items():
        device = nwbfile.create_device(name=group_name)
        groups_map[group_name] = nwbfile.create_electrode_group(
            name=group_name,
            description=group_description,
            device=device,
            location='unknown'
        )

    # add required cols to electrodes table
    for row, group_name in zip(elec_data[:].T, group_names):
        nwbfile.add_electrode(
            x=row[file_elec_col_names == b'X'][0],
            y=row[file_elec_col_names == b'Y'][0],
            z=row[file_elec_col_names == b'Z'][0],
            imp=np.nan,
            location='unknown',
            filtering='250 Hz lowpass',
            group=groups_map[group_name],
        )

    # load r2 values to input into custom cols in electrodes table
    r2 = np.load(r2_path)
    low_freq_r2 = np.ravel(r2[int(subject_id)-1, :len(group_names), 0])

    high_freq_r2 = np.ravel(r2[int(subject_id)-1, :len(group_names), 1])

    # add custom cols to electrodes table
    elecs_dset = file['chan_info']['block0_values']

    def get_data(label):
        return elecs_dset[file_elec_col_names == label, :].ravel()[is_elec]

    [nwbfile.add_electrode_column(**kwargs) for kwargs in (
        dict(
            name='standard_deviation',
            description="standard deviation of each electrode's data for the entire recording period",
            data=get_data(b'SD_channels')
        ),
        dict(
            name='kurtosis',
            description="kurtosis of each electrode's data for the entire recording period",
            data=get_data(b'Kurt_channels')
        ),
        dict(
            name='median_deviation',
            description="median absolute deviation estimator for standard deviation for each electrode",
            data=get_data(b'standardizeDenoms')
        ),
        dict(
            name='good',
            description='good electrodes',
            data=get_data(b'goodChanInds').astype(bool)

        ),
        dict(
            name='low_freq_R2',
            description='R^2 for low frequency band on each electrode',
            data=low_freq_r2
        ),
        dict(
            name='high_freq_R2',
            description='R^2 for high frequency band on each electrode',
            data=high_freq_r2
        )
    )]

    # confirm that electrodes table looks right
    # nwbfile.electrodes.to_dataframe()

    # add ElectricalSeries
    elecs_data = dset.lazy_slice[:, is_elec]
    n_bytes = np.dtype(elecs_data).itemsize

    nwbfile.add_acquisition(
        ElectricalSeries(
            name='ElectricalSeries',
            data=H5DataIO(
                data=DataChunkIterator(
                    data=elecs_data,
                    maxshape=elecs_data.shape,
                    buffer_size=int(5000 * 1e6) // elecs_data.shape[1] * n_bytes
                ),
                compression='gzip'
            ),
            rate=file['f_sample'][()],
            conversion=1e-6,  # data is in uV
            electrodes=nwbfile.create_electrode_table_region(
                region=list(range(len(nwbfile.electrodes))),
                description='all electrodes'
            )
        )
    )

    # add pose data
    pose_dset = file['pose_data']['block0_values']

    nwbfile.create_processing_module(
        name='behavior',
        description='pose data').add(
        Position(
            spatial_series=[
                SpatialSeries(
                    name=file['pose_data']['axis0'][x_ind][:-2].decode(),
                    data=H5DataIO(
                        data=pose_dset[:, [x_ind, y_ind]],
                        compression='gzip'
                    ),
                    reference_frame='unknown',
                    conversion=np.nan,
                    rate=30.
                ) for x_ind, y_ind in zip(
                    range(0, pose_dset.shape[1], 2),
                    range(1, pose_dset.shape[1], 2))
            ]
        )
    )

    # add events
    events = pd.read_csv(events_path)
    mask = (events['Subject'] == int(subject_id)) & (events['Recording day'] == int(session))
    events = events[mask]
    timestamps = events['Event time'].values
    events = events.reset_index()

    events = Events(
        name='ReachEvents',
        description=events['Event type'][0],  # Specifies which arm was used
        timestamps=timestamps,
        resolution=2e-3,  # resolution of the timestamps, i.e., smallest possible difference between timestamps
    )

    # add the Events type to the processing group of the NWB file
    nwbfile.processing['behavior'].add(events)

    # add coarse behavioral labels
    event_fp = f'sub{subject_id}_fullday_{session}'
    full_fp = coarse_events_path + '//' + event_fp + '.npy'
    coarse_events = np.load(full_fp, allow_pickle=True)

    label, data = np.unique(coarse_events, return_inverse=True)
    transition_idx = np.where(np.diff(data) != 0)
    start_t = nwbfile.processing["behavior"].data_interfaces["Position"]['L_Wrist'].starting_time
    rate = nwbfile.processing["behavior"].data_interfaces["Position"]['L_Wrist'].rate
    times = np.divide(transition_idx, rate) + start_t  # 30Hz sampling rate
    max_time = (np.shape(coarse_events)[0] / rate) + start_t
    times = np.hstack([start_t, np.ravel(times), max_time])
    transition_labels = np.hstack([label[data[transition_idx]], label[data[-1]]])

    nwbfile.add_epoch_column(name='labels', description='Coarse behavioral labels')

    for start_time, stop_time, label in zip(times[:-1], times[1:], transition_labels):
        nwbfile.add_epoch(start_time=start_time, stop_time=stop_time, labels=label)

    # add additional reaching features
    reach_features = pd.read_csv(reach_features_path)
    mask = (reach_features['Subject'] == int(subject_id)) & (reach_features['Recording day'] == int(session))
    reach_features = reach_features[mask]

    reaches = TimeIntervals(name='reaches', description='Features of each reach')
    reaches.add_column(name='Reach_magnitude_px', description='Magnitude of reach in pixels')
    reaches.add_column(name='Reach_angle_degrees', description='Reach angle in degrees')
    reaches.add_column(name='Onset_speed_px_per_sec', description='Onset speed in pixels / second)')
    reaches.add_column(name='Speech_ratio', description='rough estimation of whether someone is likely to be speaking '
                                                        'based on a power ratio of audio data; ranges from 0 (no '
                                                        'speech) to 1 (high likelihood of speech)h')
    reaches.add_column(name='Bimanual_ratio', description='ratio of ipsilateral wrist reach magnitude to the sum of '
                                                          'ipsilateral and contralateral wrist magnitudes; ranges from '
                                                          '0 (unimanual/contralateral move only) to 1 (only ipsilateral'
                                                          ' arm moving); 0.5 indicates bimanual movement')
    reaches.add_column(name='Bimanual_overlap', description='The amount of ipsilateral and contralateral wrist temporal'
                                                            'overlap as a fraction of the entire contralateral movement'
                                                            ' duration')
    reaches.add_column(name='Bimanual_class', description='binary feature that classifies each movement event as '
                                                          'unimanual (0) or bimanual (1) based on how close in time a '
                                                          'ipsilateral wrist movement started relative to each '
                                                          'contralateral wrist movement events')
    for row in reach_features.iterrows():
        row_data = row[1]
        start_time = row_data['Time of day (sec)']
        stop_time = start_time + row_data['Reach duration (sec)']
        reaches.add_row(start_time=start_time,
                        stop_time=stop_time,
                        Reach_magnitude_px=row_data['Reach magnitude (px)'],
                        Reach_angle_degrees=row_data['Reach angle (degrees)'],
                        Onset_speed_px_per_sec=row_data['Onset speed (px/sec)'],
                        Speech_ratio=row_data['Speech ratio'],
                        Bimanual_ratio=row_data['Bimanual ratio'],
                        Bimanual_overlap=row_data['Bimanual overlap (sec)'],
                        Bimanual_class=row_data['Bimanual class']
                        )

    nwbfile.add_time_intervals(reaches)

    with NWBHDF5IO(fpath_out, 'w') as io:
        io.write(nwbfile)


def convert_dir(
    in_dir,
    events_path,
    r2_path,
    coarse_events_path,
    reach_features_path,
    elec_loc_labels_path,
    n_jobs=1,
    overwrite: bool = False
):
    all_data_files = [x.stem for x in Path(in_dir).iterdir() if ".h5" in x.suffix]
    nwb_files = [x.stem for x in Path(in_dir).iterdir() if ".nwb" in x.suffix]

    if overwrite:
        in_files = [os.path.join(in_dir, f"{x}.h5") for x in all_data_files]
    else:
        in_files = [os.path.join(in_dir, f"{x}.h5") for x in all_data_files if x not in nwb_files]
    out_files = [os.path.join(in_dir, f"{Path(x).stem}.nwb") for x in in_files]

    Parallel(n_jobs=n_jobs)(
        delayed(run_conversion)(
            fpath_in,
            fpath_out,
            events_path,
            r2_path,
            coarse_events_path,
            reach_features_path,
            elec_loc_labels_path
        )
        for fpath_in, fpath_out in zip(in_files, out_files)
    )
