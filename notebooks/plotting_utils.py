#!/usr/bin/env python3

__all__=['setup_formatting','pull_max','process_data','plot_data','load_data','load_final_data','reject_outliers_fn','process_eval_data']

import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def setup_formatting():
    # figure formatting
    size = 2
    if size==0 : # small
        font_sizes, fig_size = [10,12,13], (8,3)  
    elif size==1 : # medium
        font_sizes, fig_size = [11,13,14], (12,4.5) 
    elif size==2 : # medium single
        font_sizes, fig_size = [11,13,14], (6,4.5) 
    else: # large
        font_sizes, fig_size = [12,14,15], (16,6) 
    plt.rcParams.update({'font.size': font_sizes[0], 'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 
                         'legend.fontsize': font_sizes[0],
                         'xtick.labelsize': font_sizes[1], 'ytick.labelsize': font_sizes[1], 
                         'axes.labelsize': font_sizes[2], 'axes.titlesize': font_sizes[2], 'figure.titlesize': font_sizes[2],'figure.figsize': fig_size})

def pull_max(d,xs,start_idx):
    _rew = None
    new_mean = []
    for m,x in zip(d,xs):
        if _rew is None:
            _rew = m
        else:
            if m > _rew and x > start_idx:
                _rew = m
        new_mean.append(_rew)
    return new_mean

def process_data(env_path, N=60,get_max=False, start_idx=0, penalize_early_stop=False, skip_early_stop=False, max_steps=1000, data_file='reward_data.pkl'):
    # load data sets
    data = []
    min_size = np.inf
    for i,path in enumerate(glob.glob(env_path + 'seed*/')):
        try: 
            data_set = pickle.load(open(path + data_file, 'rb'))
            data_set = np.stack(data_set)
            if data_set[-1,0] < min_size:
                min_size = data_set[-1,0]
            data.append(data_set)
        except:
            pass
        
    # process data sets (skip early termination and pull max values if specified)
    interp_data, raw_data, interp_steps = [], [], []
    xy_idx = [0, 1]
    x_samples = np.linspace(start_idx, min_size, N)
    for data_set in data:
        _x, _y = data_set[:,xy_idx].T 
        try: 
            _steps = data_set[:,3] 
        except: 
            _steps = np.roll(np.roll(data_set[:,0],-1 ) - data_set[:,0],1)
            _steps[0] = _x[0]            
        if skip_early_stop:
            data_idx = (_steps >= max_steps)
            _x, _y = _x[data_idx], _y[data_idx]
        elif penalize_early_stop: 
            penalty = max_steps - _steps
            _y = _y - penalty
        if get_max:
            _y = pull_max(_y,_x,start_idx)
            
        interp_data.append(np.interp(x_samples, _x, _y))
        if not skip_early_stop:
            interp_steps.append(np.interp(x_samples, _x, _steps))
        raw_data.append([_x, _y])
            
    if not skip_early_stop:
        # save processed data to log
        mean = np.mean(interp_steps, axis=0)
        std  = np.std(interp_steps, axis=0)
        steps_log = {'x' : x_samples, 'mean' : mean, 'std' : std, 
                   'max' : np.max(interp_steps, axis=0),
                   'min' : np.min(interp_steps, axis=0),
                   '-std' : mean-std, '+std' : mean+std, 
                    'data' : interp_steps}
    else: 
        steps_log = {}
    
    # save processed data to log
    mean = np.mean(interp_data, axis=0)
    std  = np.std(interp_data, axis=0)
    data_log = {'x' : x_samples, 'mean' : mean, 'std' : std, 
               'max' : np.max(interp_data, axis=0),
               'min' : np.min(interp_data, axis=0),
               '-std' : mean-std, '+std' : mean+std, 
                'data' : interp_data, 'raw_data' : raw_data,
               'steps_log': steps_log, 'debug': np.sum(np.vstack(interp_data)[:,-1]<-900)}
    
    return data_log

def plot_data(ax, log, label, color, fill=True, raw=True, zorder=0, linestyle='solid',linewidth=1):
    all_data = np.array(log['data']).T
    if not fill:
        ax.plot(log['x'],all_data,alpha=0.3,color=color,zorder=zorder,linestyle=linestyle)
        if raw: 
            for data in log['raw_data']:
                ax.plot(data[0],data[1],alpha=0.3,color=color,zorder=zorder,linestyle=linestyle)
    ax.plot(log['x'], log['mean'], label=label,color=color,zorder=zorder,linestyle=linestyle,linewidth=linewidth)
    if fill: 
        ax.fill_between(log['x'], 
                        log['-std'], log['+std'],
                        alpha=0.2,color=color,zorder=zorder)
        
        
from matplotlib.patches import Patch
def load_data(env_path, data_file='reward_data.pkl'):
    # load data sets
    data = []
    min_size = 100000000
    for i,path in enumerate(glob.glob(env_path + 'seed*/')):
        try: 
            data_set = pickle.load(open(path + data_file, 'rb'))
            data.append(np.stack(data_set))
            if len(data_set) < min_size:
                min_size = len(data_set)
        except:
            pass
    y_data = []
    xy_idx = [0, 1]
    for data_set in data:
        _x, _y = data_set[:,xy_idx].T 
        y_data.append(_y[-min_size:])
    y_data = np.array(y_data,dtype=np.float)
    y_data = y_data[~np.isnan(y_data)]
    
    # save processed data to log
    x_samples = 1.
    mean = np.mean(y_data)
    std  = np.std(y_data)
    data_log = {'x' : x_samples, 'mean' : mean, 'std' : std, 
               'max' : np.max(y_data),
               'min' : np.min(y_data),
               '-std' : mean-std, '+std' : mean+std,'data':y_data}
    
    return data_log


def load_final_data(env_path, data_file='buffer_data.pkl'):
    # load data sets
    data = []
    seed = []
    for i,path in enumerate(sorted(glob.glob(env_path + 'seed*/'))):
        try: 
            data_set = pickle.load(open(path + data_file, 'rb'))
            data.append(np.stack(data_set)[-1][0])
            seed.append(path.split('seed_')[-1].split('/')[0])
        except:
            pass
    flip = seed.pop(1)
    seed.insert(0,flip)
    flip = data.pop(1)
    data.insert(0,flip)
    return seed,data


def reject_outliers_fn(x, iq_range=50):
    pcnt = (100 - iq_range) / 2
    iqr = np.subtract(*np.percentile(x, [100-pcnt, pcnt]))
    median = np.median(x)
    return np.abs(x - median) <= iqr

def process_eval_data(env_path, penalize_early_stop=False, skip_early_stop=True, reject_outliers=True, max_steps=1000, data_file='reward_data.pkl'):
    # load data sets
    data = []
    for i,path in enumerate(glob.glob(env_path + 'seed*/')):
        # if '513' not in path:
        try: 
            data_set = pickle.load(open(path + data_file, 'rb'))
            data.append(np.stack(data_set))
        except:
            pass

    y_data = np.empty(0)
    xy_idx = [0, 1]
    for data_set in data:
        _, _y = data_set[:,xy_idx].T 
        _steps = data_set[:,3]
        if skip_early_stop:
            data_idx = (_steps >= max_steps)
            _y = _y[data_idx]
        elif penalize_early_stop: 
            penalty = (max_steps - _steps)
            _y = _y - penalty
        y_data = np.hstack([y_data,_y])

                
    y_data = y_data[~np.isnan(y_data)]
    if reject_outliers: 
        locs = reject_outliers_fn(y_data,80) 
        y_data  = y_data[locs]

    # save processed data to log
    x_samples = 1.
    mean = np.mean(y_data)
    std  = np.std(y_data)
    data_log = {'x' : x_samples, 'mean' : mean, 'std' : std, 
               '-std' : mean-std, '+std' : mean+std,'data':y_data}
    
    return data_log