import numpy as np
import pandas as pd
import bokeh_catplot
import bokeh 
import bokeh.io
from bokeh.themes import Theme
from bokeh.io import output_file, save, output_notebook
from bokeh.plotting import figure

import holoviews as hv
hv.extension('bokeh')

def plot_with_error_bars(means, confs, names, **kwargs):
    """Make a horizontal plot of means/conf ints with error bars."""
    frame_height = kwargs.pop("frame_height", 150)
    frame_width = kwargs.pop("frame_width", 450)

    p = bokeh.plotting.figure(
        y_range=names, frame_height=frame_height, frame_width=frame_width, **kwargs
    )

    p.circle(x=means, y=names)
    for conf, name in zip(confs, names):
        p.line(x=conf, y=[name, name], line_width=2)

    return p

def model_cdf(t, beta_1, beta_2):
    """
    Theoretical curve for double exponential model. 
    """
    if np.isclose(beta_1, beta_2):
        return st.gamma.cdf(a = 2, loc=0, scale=1/beta_1,x = t)
    
    cdf = (1 - np.exp(-beta_1 * t)) / beta_1 - (1 - np.exp(-beta_2 * t)) / beta_2

    return beta_1 * beta_2 * cdf / (beta_2 - beta_1)