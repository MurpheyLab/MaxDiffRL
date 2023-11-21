#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import yaml
from termcolor import cprint
from datetime import datetime

import torch
import numpy as np
import random
import pickle
import gym
import argparse
import matplotlib.pyplot as plt

# local imports
from envs import build_env


parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='SwimmerEnv_v3', help="PointMass2D_DoubleIntEnv,SwimmerEnv_v3, AntEnv_v3, etc.")
parser.add_argument('--method', type=str, default='maxdiff', help='maxdiff, mppi, or sac_orig')
parser.add_argument('--seed', type=int, default=13, help='any positive integer')
parser.add_argument('--render', dest='render', action='store_true',help='show visualization while running')
parser.add_argument('--no_render', dest='render', action='store_false',help='run offline / without showing plots')
parser.set_defaults(render=True)
parser.add_argument('--cpu', dest='cpu', action='store_true',help='only use CPU')
parser.add_argument('--no_cpu', dest='cpu', action='store_false',help='try to use GPU if available')
parser.set_defaults(cpu=False)
parser.add_argument('--record', dest='record', action='store_true',help='record video of environment')
parser.set_defaults(record=False)
parser.add_argument('--mod', type=str, default='_H40_alpha100',help="end of file name for specfic config after beta (e.g. '_H40' or '_Light_H40')")
parser.add_argument('--iters',   type=int,   default='1',help="how many test iterations to run")
parser.add_argument('--frame', type=int, default=-1,help='which saved model to use')
parser.add_argument('--physics_model_eval', type=str, default='orig',help="[gym envs only] specify alternate xml file for enviroment (e.g. 'light' or 'orig' for swimmer, can be same or different from training)")
parser.add_argument('--base_dir',   type=str,   default='./data/',help="directory where enviroment folder with data")
parser.add_argument('--start_mode', type=str, default='one_corner',help="[PointMass envs only] one_corner, four_corners, circle10, random")
parser.add_argument('--beta', type=float, default=0.01, help='[PointMass envs only] weights pointmass A matrix (e.g. 1.0, 0.1, 0.01, 0.001) ')
args = parser.parse_args()
cprint(args,'cyan') # other good colors are red, green, yellow, magenta

args.v3 = 'v3' in args.env
args.pointmass = 'PointMass' in args.env
args.done_util = True

save_states = False

# added to stop rendering when exiting
from signal import signal, SIGINT
from sys import exit

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected.')
    args.render = False
    print('Exiting gracefully')
    exit(0)

if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    if args.pointmass:
        pm_mod = '_beta'+ '{:0.0e}'.format(args.beta).replace('+','').replace('-','_')
        args.mod = pm_mod + args.mod

    # load config
    state_dict_path = args.base_dir + args.method + '/' + args.env + args.mod + '/seed_{}/'.format(args.seed)
    print(state_dict_path)

    base_method = args.method[:3]
    config_path = state_dict_path + 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'H_sequence' in config.keys():
        if 'horizon' in config['H_sequence'].keys():
            config['planner']['horizon'] = config['H_sequence']['horizon'][-1]
    if 'alpha_sequence' in config.keys():
        if 'alpha' in config['alpha_sequence'].keys():
            config['planner']['alpha'] = config['alpha_sequence']['alpha'][-1]

    if  not(args.pointmass) and  args.record and args.render:
        cprint('cannot record while rendering, disabling rendering','red')
        args.render = False
        cprint('only recording 1 iteration','red')
        args.iters = 1

    # set seeds / torch config
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set torch config
    device ='cpu'
    if not args.cpu:
        if torch.cuda.is_available():
            torch.set_num_threads(1)
            device  = 'cuda:0'
            print('Using GPU Accel')
        else:
            args.cpu = True

    # initialize environment
    args.mod_weight = args.physics_model_eval
    env, env_name, action_dim, state_dim, traj, viewer = build_env(args,config,device)
    cprint(env,'green')
    print(action_dim,state_dim)

    # set up video recording
    if args.frame == -1:
        test_frame = 'final'
    else:
        test_frame = args.frame

    if not(args.pointmass) and args.record:
        video_path = '{}/vid/{}/'.format(args.base_dir,args.env)
        if base_method == 'max':
            test_path = video_path+'{}-mod{}-alpha_{}-seed_{}-frame_{}'.format(args.method,args.mod_weight,int(config['planner']['alpha']),args.seed,test_frame)
        else:
            test_path = video_path+'{}-mod{}-seed_{}-frame_{}'.format(args.method,args.mod_weight,args.seed,test_frame)
        if os.path.exists(video_path) == False:
            os.makedirs(video_path)
        env = gym.wrappers.Monitor(env, test_path, force=True)

    # load models / policies / controllers
    if base_method == 'sac':
        from sac_lib import PolicyNetwork
        policy_net = PolicyNetwork(state_dim, action_dim, config['hidden_dim'],device=device).to(device)
        policy_net.load_state_dict(torch.load(state_dict_path+'policy_{}.pt'.format(test_frame), map_location=device))
    else:
        from mpc_lib import Model
        model_kwargs = {'model_layers':config['model_layers'],'model_AF':config['model_activation_fun'],
                        'reward_layers':config['reward_layers'],'reward_AF':config['reward_activation_fun']}
        model = Model(state_dim, action_dim, **model_kwargs).to(device)
        model.load_state_dict(torch.load(state_dict_path+'model_{}.pt'.format(test_frame), map_location=device))
        if base_method == 'mpp':
            from mpc_lib import PathIntegral
            planner = PathIntegral(model,device=device,**config['planner'])
        elif base_method == 'max':
            from mpc_lib import MaxDiff
            planner = MaxDiff(model,device=device,**config['planner'])

    # main simulation loop
    max_frames  = config['max_frames']
    max_steps   = config['max_steps']

    states = []
    for ep_num in range(args.iters):
        ep_states = []
        state = env.reset()
        if save_states: ep_states.append(state)
        if not(base_method == 'sac'):
            planner.reset()

        episode_reward = 0
        done = False
        for step in range(max_steps):
            if base_method == 'sac' :
                action = policy_net.get_action(state.copy())
            else:
                action = planner(state.copy())
            state, reward, done, _ = env.step(action.copy())
            if not args.pointmass:
                if args.render:
                    try:
                        env.render(mode="human")
                    except TypeError as err:
                        env.render()
                if args.record and step % 100 == 0:
                    print(f"{step}/{max_steps}")
            if save_states or args.pointmass: ep_states.append(state)
            episode_reward += reward

            if done:
                break
        if save_states: states.append(np.stack(ep_states))
        print(step, episode_reward, state[0])
        if args.pointmass:
            viewer.render(ep_states,ep_num)

    env.close()

    # clean up recording files
    if not(args.pointmass) and args.record:
        import shutil
        path = os.path.abspath(test_path)
        size, max_size, max_file = 0,0,''
        # walking through files in folder
        for folder, subfolders, files in os.walk(path):
            # checking the size of each file
            for file in files:
                size = os.stat(os.path.join(folder, file)).st_size
                # updating maximum size
                if size>max_size:
                    max_size = size
                    max_file = os.path.join(folder, file)
        shutil.move(max_file,test_path+'.mp4')
        shutil.rmtree(test_path)

    # save states
    if save_states:
        states = np.stack(states)
        print(states.shape)

        save_path = state_dict_path + '../eval_{}/'.format(args.env+args.mod)
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        pickle.dump(states, open(save_path+'state_data_seed{}'.format(args.seed)+ '.pkl', 'wb'))

    if args.pointmass:
        plt.pause(1) # make sure all iters finish rendering
        i = input('press any key to quit')
