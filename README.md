# ajile12-nwb-data
Package for analyzing long-term naturalistic human intracranial neural recordings and pose.
Includes code to plot figures from data descriptor paper (LINK TBD).

Our data dashboard is available for streaming (*stream_dashboard.ipynb*) and
offline (*offline_dashboard.ipynb*) viewing. All figure/table scripts are also
available for offline (*figs_offline*) and streaming (*figs_stream*). Note that
while streaming is fast for the data dashboard, the figure/table scripts have
not been optimized for streaming and will run slowly.

**Fig 2. Coarse behavior labelling:** figs_offline/Fig_coarse_labels.ipynb

**Fig 3. ECoG electrode positions and technical validation** figs_offline/Fig_pow_spectra.ipynb (requires running *figs_offline/comput_cont_spec.py* first)

**Table 2. Individual participant characteristics** figs_offline/Table_part_characteristics.ipynb

**Tables 3-4. Coarse activity/blocklist durations** figs_offline/Table_coarse_labels.ipynb
