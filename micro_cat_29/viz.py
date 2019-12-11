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

def double_exponential_ecdf(beta_2):
    
    # Sample time until the arrival for the first Poisson process
    time_1 = rg.exponential(1/beta_1, size = n_trials)

    # Sample time until the arrival for the second Poisson process
    time_2 = rg.exponential(1/beta_2, size = n_trials)

    # Sum both times to get time to catastrophe
    time_to_cat = time_2 + time_1
    
    # Initialize y-values for the ECDF
    ecdf = np.linspace(0, 1, n_trials )
    
    # Sort the time to catastrophe values
    x_sorted = sorted(time_to_cat)


    # Plot the ECDF with a constant x-axis
    im_plot = hv.Curve((x_sorted, ecdf)).opts(xlabel = 'time to catastrophe', 
                                              ylabel = 'ECDF').opts(xlim=(0, 10), 
                                              height = 300, width = 500)

    
    return im_plot

def double_exponential_panel(beta_2):
    
    time_1 = rg.exponential(1/beta_1, size = n_trials)

    time_2 = rg.exponential(1/beta_2, size = n_trials)

    # Get the total time to catastrophe
    time_to_cat = time_2 + time_1

    # Get y-values for the ECDF 
    ecdf = np.linspace(0, 1, n_trials)
    
    # Sort the time to catastrophe samples
    time_to_cat_sorted = sorted(time_to_cat)
    
    # Plot the ECDF with a constant x-axis
    im_plot = hv.Curve((time_to_cat_sorted, ecdf), label = 'ECDF').opts(
                                              xlabel = 'time to catastrophe', 
                                              ylabel = 'ECDF').opts(
                                              xlim=(0, 10), 
                                              height = 300,
                                              width = 500)
    
    # Compute the theoretical CDF
    cdf = double_exponential_cdf(beta_1, beta_2, time_vector)
    
    # Initialize a hv.Curve object for the theoretical CDF 
    theor_cdf_curve = hv.Curve((time_vector, cdf), label = 'theoretical CDF').opts(
                                              xlabel = 'time to catastrophe', 
                                              ylabel = 'Cumulative density function').opts(
                                              xlim=(0, 10), 
                                              height = 300, width = 500)

    return (theor_cdf_curve*im_plot)

def model_cdf(t, beta_1, beta_2):
    """
    Theoretical curve for double exponential model. 
    """
    if np.isclose(beta_1, beta_2):
        return st.gamma.cdf(a = 2, loc=0, scale=1/beta_1,x = t)
    
    cdf = (1 - np.exp(-beta_1 * t)) / beta_1 - (1 - np.exp(-beta_2 * t)) / beta_2

    return beta_1 * beta_2 * cdf / (beta_2 - beta_1)