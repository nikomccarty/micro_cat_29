import numpy as np
import pandas as pd

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

def diff_means(x, y): 
    """Compute the difference in means, means / std."""
    diff_means = (np.mean(x) - np.mean(y))**2 / (np.std(x) - np.std(y))
    return diff_means

def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]

def draw_perm_reps(x, y, stat_fun, size=1):
    """Generate array of permuation replicates."""
    return np.array([stat_fun(*draw_perm_sample(x, y)) for _ in range(size)])

def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = diff_means(x_perm, y_perm)

    return out
