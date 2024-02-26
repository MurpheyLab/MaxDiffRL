#!/usr/bin/env python3

import yaml
from termcolor import cprint
import time
from utils import get_duration

import torch
import numpy as np
import random
import pickle

# local imports
from envs import build_env

import argparse

parser = argparse.ArgumentParser()
# these params are the learned model/policy to load
parser.add_argument('--env',   type=str,   default='SwimmerEnv_v3', help="PointMass2D_DoubleIntEnv,SwimmerEnv_v3, AntEnv_v3, etc.")
parser.add_argument('--method', type=str, default='maxdiff', help='maxdiff, mppi, or sac_orig')
parser.add_argument('--seed', type=int, default=13, help='any positive integer')
parser.add_argument('--done_util', dest='done_util', action='store_true', help='ends epoch with done signal from environment')
parser.add_argument('--no_done_util', dest='done_util', action='store_false', help='ignores done signal from environment and runs for max_steps')
parser.set_defaults(done_util=True)
parser.add_argument('--render', dest='render', action='store_true',help='render each epoch in figure window as running')
parser.add_argument('--no_render', dest='render', action='store_false',help='run offline / without showing plots')
parser.set_defaults(render=False)
parser.add_argument('--cpu', dest='cpu', action='store_true',help='only use CPU')
parser.add_argument('--no_cpu', dest='cpu', action='store_false',help='try to use GPU if available')
parser.set_defaults(cpu=False)
parser.add_argument('--mod', type=str, default='_H40_alpha100',help="end of file name for specfic config after beta (e.g. '_H40' or '_H40_alpha5')")
parser.add_argument('--iters',   type=int,   default='10',help="how many test iterations to run")
parser.add_argument('--final_only', dest='final_only', action='store_true',help='only test final saved model')
parser.add_argument('--all_frames', dest='final_only', action='store_false',help='save checkpoints and final saved model')
parser.set_defaults(final_only=True)
# this specifies the xml to load
parser.add_argument('--physics_model_eval', type=str, default='orig',help="specify alternate xml file for testing (gym envs only, can be same or different from training)")
parser.add_argument('--base_dir',   type=str,   default='./data/',help="directory where enviroment folder with data")
parser.add_argument('--start_mode', type=str, default='one_corner',help="(PointMass envs only) one_corner, four_corners, circle10, random")
parser.add_argument('--beta', type=float, default=0.01, help='(PointMass envs only) weights pointmass control matrix (e.g. 1.0, 0.1, 0.01, 0.001) ')

args = parser.parse_args()
cprint(args,'cyan') 
args.v3 = 'v3' in args.env
args.pointmass = 'PointMass' in args.env

# added to stop rendering when exiting
from signal import signal, SIGINT
from sys import exit

if args.pointmass:
    def end_test():
        env.close()
        try:
            print('saving data set')
            pickle.dump(rewards, open(state_dict_path + args.start_mode + '_final_eval_reward_data' + '.pkl', 'wb'))
        except NameError:
            print('no rewards to save, closing simulation')
        fig_path = state_dict_path + args.start_mode + "_eval_" + "final_fig" 
        if args.render:
            viewer.save(fig_path)
        else:
            try:
                traj.save_fig(fig_path + '.svg')
            except:
                traj.save_buff(fig_path + '.pkl')

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

    # load models / policies / controllers
    if base_method == 'sac':
        from sac_lib import PolicyNetwork
        policy_net = PolicyNetwork(state_dim, action_dim, config['hidden_dim'],device=device).to(device)
    else:
        from mpc_lib import Model
        model_kwargs = {'model_layers':config['model_layers'],'model_AF':config['model_activation_fun'],
                        'reward_layers':config['reward_layers'],'reward_AF':config['reward_activation_fun']}
        model = Model(state_dim, action_dim, **model_kwargs).to(device)
        if base_method == 'mpp':
            from mpc_lib import PathIntegral
            planner = PathIntegral(model,device=device,**config['planner'])
        elif base_method == 'max':
            from mpc_lib import MaxDiff
            planner = MaxDiff(model,device=device,**config['planner'])

    start_time = time.time()
    # main simulation loop
    max_steps   = config['max_steps']
    rewards     = []

    if args.final_only:
        test_frames = ['final']
    else:
        test_frames = [ 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]

    for test_frame in test_frames:
        # load model/policy for particular frame
        if base_method == 'mpp':
            model.load_state_dict(torch.load(state_dict_path+'model_{}.pt'.format(test_frame), map_location=device))
        elif base_method == 'sac':
            policy_net.load_state_dict(torch.load(state_dict_path+'policy_{}.pt'.format(test_frame), map_location=device))
        elif base_method == 'max':
            model.load_state_dict(torch.load(state_dict_path+'model_{}.pt'.format(test_frame), map_location=device))
        else:
            raise ValueError('method not found')

        # test for fixed number of iters
        for ep_num in range(args.iters):
            state = env.reset()
            if base_method == 'sac' :
                action = policy_net.get_action(state.copy())
            else:
                planner.reset()
                action = planner(state.copy())

            episode_reward = 0
            states = []
            for step in range(max_steps):
                if base_method == 'sac' :
                    action = policy_net.get_action(state.copy())
                else:
                    action = planner(state.copy())
                state, reward, done, _ = env.step(action.copy())
                if args.pointmass:
                    states.append(state)
                else:
                    if args.render:
                        try:
                            env.render(mode="human")
                        except TypeError as err:
                            env.render()
                episode_reward += reward

                if args.done_util:
                    if done:
                        break
            if args.pointmass:
                if args.render:
                    viewer.render(states,ep_num)
                traj.push(states,ep_num)
            step += 1
            if ep_num % (args.iters//10) == 0:
                get_duration(start_time)
            print(test_frame,f'{ep_num}/{args.iters}', episode_reward, step, state[0])
            rewards.append([test_frame, episode_reward,ep_num, step])
    env.close()
    print('saving reward log')
    if args.pointmass:
        end_test()
    else:
        pickle.dump(rewards, open(state_dict_path + '/' + args.physics_model_eval + '_eval_reward_log.pkl', 'wb'))
