import numpy as np
import plotly.graph_objects as go
import pynwb
from ipywidgets import widgets, ValueWidget
from plotly.colors import DEFAULT_PLOTLY_COLORS


class ShowElectrodesWidget(ValueWidget, widgets.HBox):
    def __init__(self, nwbobj: pynwb.base.DynamicTable, **kwargs):
        super().__init__()

        group_names = nwbobj.group_name[:]
        ugroups, group_pos, counts = np.unique(group_names,
                                               return_inverse=True,
                                               return_counts=True)

        self.fig = go.FigureWidget()
        x = nwbobj.x[:]
        y = nwbobj.y[:]
        z = nwbobj.z[:]
        for i, group in enumerate(ugroups):
            inds = group_names == group
            self.fig.add_trace(
                go.Scatter3d(
                    x=x[inds], y=y[inds], z=z[inds],
                    surfacecolor=np.array(DEFAULT_PLOTLY_COLORS)[i % len(DEFAULT_PLOTLY_COLORS)],
                    mode='markers',
                    name=group
                )
            )

        self.children = [self.fig]
