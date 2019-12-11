import numpy as np
import pandas as pd

import holoviews as hv
hv.extension('bokeh')

import tinypkg.general_utils as utils


def plot_timecourse(df, variable, value, condition=None, split=None, sort=None, cmap=None, show_all=False,
                    show_points='default', legend=False, height=350, width=500, additional_opts={}):
#   ...

    # Check columns
    utils.check_df_col(df, variable, name='variable')
    utils.check_df_col(df, value, name='value')
    utils.check_df_col(df, condition, name='condition')
    utils.check_df_col(df, split, name='split')
    utils.check_df_col(df, sort, name='sort')

#   ...

    # Check for replicates; aggregate df
    groups = [grouping for grouping in (condition, split) if grouping is not None]
    if groups == []:
        groups = None
    replicates, df = utils.check_replicates(df, variable, value, groups)