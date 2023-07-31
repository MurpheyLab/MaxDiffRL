#!/usr/bin/env python3

import datetime
import time
from termcolor import cprint
import yaml
import torch
import os

def get_duration(start_time,print_update=True):
    duration_str = str(datetime.timedelta(seconds=(time.time()-start_time)))
    if print_update: cprint('runtime: '+duration_str,'magenta')
    return duration_str

import torch
from torch.distributions import Normal
from typing import List

def _batch_mv(mat: torch.Tensor,vec: torch.Tensor):
    '''assumes batch dim=0'''
    return (mat @ vec.unsqueeze(-1)).squeeze(-1)

def jit_prob_dist(
    prob_dist: torch.distributions.distribution.Distribution,
) -> torch.nn.Module:
    class JITProbDist(torch.nn.Module):
        @torch.jit.ignore
        def single_sample(self):
            return prob_dist.sample()
        @torch.jit.ignore
        def sample(self,x:List[int]):
            return prob_dist.sample(x)
        @torch.jit.ignore
        def log_prob(self,x):
            return prob_dist.log_prob(x)
        def extra_repr(self):
            return prob_dist.__repr__()
    return JITProbDist()

import numpy as np

def obs_to_np(time_step,subset=True):
    if subset:
        states = ['origin','torso_velocity','torso_upright',
                  'imu','force_torque','rangefinder']
        obs = [time_step.observation[key] for key in states]
        ego_state = time_step.observation['egocentric_state']
        obs.append(ego_state)
    else:
        obs = list(time_step.observation.values())
        # obs[2] = obs[2].reshape(1,)
    obs = np.hstack(obs)
    return obs

def save_config(args,config,env_name):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
    dir_name = 'seed_{}/'.format(str(args.seed))
    if args.pointmass:
        mod = '_beta'+ '{:0.0e}'.format(args.beta).replace('+','').replace('-','_')
    else:
        mod = ''
    path = args.base_dir + args.method + '/' + env_name + mod +  config['name_mod'] + '/' + dir_name
    if os.path.exists(path) == False:
        os.makedirs(path)
    # save config yaml
    with open(path+'/config.yaml', 'w') as f:
        yaml.safe_dump(config,f)

    # save config
    # cprint(torch.__config__.parallel_info(),'blue')
    cprint(path,'red')
    with open(path + "/../config.txt","a") as f:
        f.write('\n'+ date_str )
        f.write('\nArgparse\n')
        f.write('\t' + str(args))
        f.write('\nConfig\n')
        for key,value in config.items():
            f.write('\t' + str(key) + '\t' + str(value) + '\n')
        f.write('\Torch nConfig\n')
        f.write(torch.__config__.parallel_info())
        f.close()
    with open(path + "/rewards.txt","a") as f:
        f.write('{}\t{}\t{}\t{}\n'.format('ep_num','episode_reward','step','ep_time'))
    return path
