# ajile12-nwb-data
Package for analyzing long-term naturalistic human intracranial neural recordings and pose.
Includes code to plot figures from data descriptor paper (ADD LINK).

**Fig 2. Coarse behavior labelling** figs/Plot_coarse_labels.ipynb

**Fig 3. ECoG electrode positions and identified noisy electrodes** figs/good_elecs.ipynb and figs/ecog_eleclocs_nwb.ipynb

**Fig 4. ECoG power spectra over time** figs/Plot_pow_spectra.ipynb (requires running *figs/comput_cont_spec.py* first)

**Fig 5. Validation of pose trajectories and movement event selection** figs/DLC_recon_errors.ipynb and NWB_wrist_trajs.ipynb

**Tables 3-4. Coarse activity/blocklist durations** figs/Cont_coarse_labels.ipynb


## Installation
```bash
pip install git+https://github.com/catalystneuro/brunton-lab-to-nwb.git
```

## Usage
```python
from brunton_lab_to_nwb import run_conversion

run_conversion('in/path', 'out/path')
```

## Example uses: 
### Load nwb file:
```python
from pynwb import NWBHDF5IO
io = NWBHDF5IO(r'C:\Users\micha\Desktop\Brunton Lab Data\H5\subj_01_day_3.nwb', mode='r')
nwb = io.read()
```

### See accessible fields in nwb file object:
```python
nwb_file.fields
```

### Get electrode series:
```python
nwb_file.electrodes
```

### Align events
```python
# Get reach events
events = nwb_file.processing["behavior"].data_interfaces["ReachEvents"]
# Get reach position
reach_arm_pos = nwb_file.processing["behavior"].data_interfaces["Position"]["L_Wrist"]
# Set window around event alignment
before = 1.5 # in seconds
after = 1.5 # in seconds
starts = self.events - before
stops = self.events + after
# Get trials aligned by events
trials = align_by_times(reach_arm_pos, starts, stops)
```
