#!/usr/bin/env python3

import sys
import os
import yaml
from termcolor import cprint
from datetime import datetime
import time

import torch
import numpy as np
import random
import pickle
import argparse

# local imports
from utils import get_duration, save_config

parser = argparse.ArgumentParser()
parser.add_argument('--env',   type=str,   default='SwimmerEnv_v3', help="PointMass2D_DoubleIntEnv,SwimmerEnv_v3, AntEnv_v3, etc.")
parser.add_argument('--method', type=str, default='maxdiff', help='maxdiff, mppi, or sac_orig')
parser.add_argument('--seed', type=int, default=666, help='any positive integer')
parser.add_argument('--log', dest='log', action='store_true',help='save data for experiment')
parser.add_argument('--no_log', dest='log', action='store_false',help='run test without saving')
parser.set_defaults(log=True)
parser.add_argument('--render', dest='render', action='store_true',help='show visualization while running')
parser.add_argument('--no_render', dest='render', action='store_false',help='run offline / without showing plots')
parser.set_defaults(render=False)
parser.add_argument('--cpu', dest='cpu', action='store_true',help='only use CPU')
parser.add_argument('--no_cpu', dest='cpu', action='store_false',help='try to use GPU if available')
parser.set_defaults(cpu=False)
parser.add_argument('--mod_weight', type=str, default='None',help="[gym envs only] load alternate xml file for enviroment (e.g. 'light' or 'orig' for swimmer)")
parser.add_argument('--frames_before_learning', type=int, default=0,help="if specified, number of frames to collect before starting to learn (otherwise, batch size is used)")
parser.add_argument('--random_actions', type=int, default=0,help="if specified, number random frames to collect before starting to use the policy")
parser.add_argument('--base_dir',   type=str,   default='./results/',help="where to save the data (if log=True)")
parser.add_argument('--singleshot', dest='singleshot', action='store_true',help="don't reset for each epoch and run all steps from initial condition")
parser.set_defaults(singleshot=False)
parser.add_argument('--start_mode', type=str, default='one_corner',help="[PointMass envs only] one_corner, four_corners, circle10, random")
parser.add_argument('--beta', type=float, default=0.01, help='[PointMass envs only] weights pointmass A matrix (e.g. 1.0, 0.1, 0.01, 0.001) ')

args = parser.parse_args()
cprint(args,'cyan') # other good colors are red, green, yellow, magenta

args.v3 = 'v3' in args.env
args.pointmass = 'PointMass' in args.env
args.done_util = True

# added to save when exiting
from signal import signal, SIGINT
from sys import exit

def end_test():
    env.close()
    if args.log:
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        pickle.dump(eval_rewards, open(path + 'eval_reward_data' + '.pkl', 'wb'))
        if base_method == 'sac':
            torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
        else:
            torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
            pickle.dump(model_optim.log, open(path + 'optim_data'+ '.pkl', 'wb'))

        # save duration
        end = datetime.now()
        date_str = end.strftime("%Y-%m-%d_%H-%M-%S/")
        duration_str = get_duration(start_time)

        # save config
        with open(path + "/../config.txt","a") as f:
            f.write('End Time\n')
            f.write('\t'+ date_str + '\n')
            f.write('Duration\n')
            f.write('\t'+ duration_str + '\n')
            f.close()

        # save final steps
        if args.pointmass:
            fig_saved = False
            try:
                if args.render:
                    if args.singleshot:
                        viewer.save(path + "/final_fig_viewer.svg")
                        fig_saved = True
                    viewer.close()
            except:
                pass
            if not fig_saved:
                try:
                    traj.save_fig(path + "/final_fig.svg")
                except:
                    traj.save_buff(path + "/final_fig.pkl")
        else:
            buff = replay_buffer.get_final_samples(10000)
            pickle.dump(buff, open(path + 'buffer_data'+ '.pkl', 'wb'))



def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected.')
    end_test()
    print('Exiting gracefully')
    exit(0)

# eval function
def eval():
    state = env.reset()
    if not(base_method == 'sac' ):
        planner.reset()

    episode_reward = 0.
    states = []
    for step in range(max_steps):
        if base_method == 'sac' :
            action = policy_net.get_action(state,eval=True)
        else:
            action = planner(state,eval=True)
        state, reward, done, _ = env.step(action.copy())
        episode_reward += reward
        if args.pointmass:
            states.append(state)
        elif args.render:
            env.render(mode="human")
        if args.done_util:
            if done:
                break
    step += 1
    cprint('eval: {} {}'.format(episode_reward, step),'cyan')
    return states, episode_reward, step


if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    # load config
    base_method = args.method[:3]
    if args.singleshot:
        mod = '_singleshot'
    else:
        mod=''
    if base_method == 'sac':
        config_path = f'./config/sac{mod}.yaml'
    elif base_method == 'mpp':
        config_path = f'./config/mppi{mod}.yaml'
    elif base_method == 'max':
        config_path = f'./config/maxdiff{mod}.yaml'
    else:
        raise ValueError('config file not found for env')


    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict['default']
        if args.env in list(config_dict.keys()):
            config.update(config_dict[args.env])
        else:
            raise ValueError('env not found config file')

    if args.singleshot:
        args.done_util = False
        ss_count = 0

    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_flush_denormal(True)

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
    from envs import build_env
    env, env_name, action_dim, state_dim, traj, viewer = build_env(args,config,device)
    cprint(env,'green')
    print('actions states',action_dim,state_dim)

    # load models / policies / controllers
    from envs import ReplayBuffer
    if args.pointmass:
        eval_freq = 10
        replay_buffer_size = int(3e5)
    else:
        eval_freq = 5
        replay_buffer_size = int(1e6)
    replay_buffer = ReplayBuffer(replay_buffer_size,state_dim,action_dim)
    replay_buffer.seed(args.seed)

    if base_method == 'sac':
        from sac_lib import PolicyNetwork, SoftActorCritic
        policy_net = PolicyNetwork(state_dim, action_dim, config['hidden_dim'],device=device).to(device)
        sac = SoftActorCritic(policy=policy_net,
                              state_dim=state_dim,
                              action_dim=action_dim,
                              replay_buffer=replay_buffer,
                              hidden_dim=config['hidden_dim_sac'],
                              policy_lr=config['policy_lr'],
                              value_lr=config['value_lr'],
                              soft_q_lr=config['soft_q_lr'],
                              device=device)
    else:
        from mpc_lib import Model, ModelOptimizer
        model_kwargs = {'model_layers':config['model_layers'],'model_AF':config['model_activation_fun'],
                        'reward_layers':config['reward_layers'],'reward_AF':config['reward_activation_fun']}
        model = Model(state_dim, action_dim,**model_kwargs).to(device)
        #### jit model for planner (samples)
        with torch.no_grad():
            inputs = (torch.rand(config['planner']['samples'],state_dim,device=device),torch.rand( config['planner']['samples'],action_dim,device=device))
            jit_model_plan = torch.jit.trace(model,inputs) # set up traced model
            primed = jit_model_plan(*inputs) # prime model
            # print(jit_model_plan.graph)
        #### jit model for optimizer (batch size)
        inputs = (torch.rand(config['batch_size'],state_dim,device=device),torch.rand( config['batch_size'],action_dim,device=device))
        jit_model_opt = torch.jit.trace(model,inputs) # set up traced model
        primed = jit_model_opt(*inputs) # prime model
        model_optim = ModelOptimizer(jit_model_opt, replay_buffer, lr=config['model_lr'],device=device)
        if base_method == 'mpp':
            from mpc_lib import PathIntegral
            planner = PathIntegral(jit_model_plan,device=device,**config['planner'])
        elif base_method == 'max':
            from mpc_lib import MaxDiff
            planner = MaxDiff(jit_model_plan,device=device,**config['planner'])

    update_H = False
    if 'H_sequence' in config.keys():
        update_H = True
        H_seq_idx = 0
        ready_to_update_H = False
        def update_H_fn(H_seq_idx):
            cprint('updating horizon to {}'.format(config['H_sequence']['horizon'][H_seq_idx]),'magenta')
            planner.update_horizon(config['H_sequence']['horizon'][H_seq_idx])
            H_seq_idx += 1
            update_H = False if H_seq_idx >= len(config['H_sequence']['steps']) else True
            return update_H, H_seq_idx

    update_alpha = False
    if 'alpha_sequence' in config.keys():
        update_alpha = True
        alpha_seq_idx = 0
        ready_to_update_alpha = False
        def update_alpha_fn(alpha_seq_idx):
            new_alpha = config['alpha_sequence']['alpha'][alpha_seq_idx]
            cprint('updating alpha to {}'.format(new_alpha),'magenta')
            logdet_method = 'abs' if alpha_seq_idx == 0 else None
            planner.update_alpha(new_alpha,logdet_method=logdet_method)
            alpha_seq_idx += 1
            update_alpha = False if alpha_seq_idx >= len(config['alpha_sequence']['steps']) else True
            return update_alpha, alpha_seq_idx

    update_reward = False
    if 'reward_sequence' in config.keys():
        update_reward = True
        reward_seq_idx = 0
        ready_to_update_reward = False
        def update_reward_fn(reward_seq_idx):
            reward_scale = config['reward_sequence']['reward'][reward_seq_idx]
            cprint('updating reward scale to {}'.format(reward_scale),'magenta')
            reward_seq_idx += 1
            update_reward = False if reward_seq_idx >= len(config['reward_sequence']['steps']) else True
            return update_reward, reward_seq_idx

    # set up logs
    start_time = time.time()
    if args.log:
        path = save_config(args,config,env_name)

    # main simulation loop
    max_frames   = config['max_frames']
    max_steps    = config['max_steps'] if not args.singleshot else max_frames
    reward_scale = config['reward_scale']
    batch_size   = config['batch_size']

    frame_idx    = 0
    rewards      = []
    eval_rewards = []

    # pretrain params
    frames_before_learning = max(batch_size,args.frames_before_learning)
    RANDOM_FRAMES = args.random_actions
    if args.random_actions > 0:
        def get_random_action():
            return np.random.random(action_dim) * 2 - 1

    ep_num = 0
    while (frame_idx < max_frames):
        ep_start_time = time.time()
        state = env.reset()
        if frame_idx < RANDOM_FRAMES:
            action = get_random_action()
            if frame_idx == RANDOM_FRAMES-1:
                cprint('finished collecting random actions','green')
        else:
            if base_method == 'sac' :
                action = policy_net.get_action(state.copy())
            else:
                planner.reset()
                action = planner(state.copy())

        episode_reward = 0
        done = False
        states = []
        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action.copy())

            # get next action
            if frame_idx < RANDOM_FRAMES:
                next_action = get_random_action()
            else:
                if base_method == 'sac':
                    next_action = policy_net.get_action(next_state.copy())
                else:
                    next_action = planner(next_state.copy())

            # error handling
            if np.isnan(next_action).any():
                cprint('nan in action, resetting simulation','magenta')
                break

            # add to buffer
            replay_buffer.push(state, action, reward_scale * reward, next_state, next_action, done)

            # train
            if len(replay_buffer) > frames_before_learning:
                if base_method == 'sac':
                    if 'PointMass' in args.env:
                        sac.update(batch_size,soft_tau=0.01,debug=(frame_idx%250==0))
                    else:
                        sac.update(batch_size,debug=(frame_idx%250==0))
                else:
                    model_optim.update_model(batch_size, mini_iter=config['model_iter'],
                    debug=(frame_idx%250==0),calc_eig=False)

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if args.pointmass:
                states.append(state)
            elif args.render:
                env.render(mode="human")

            if update_H and (frame_idx % config['H_sequence']['steps'][H_seq_idx] == 0):
                if args.pointmass:
                    traj.args.horizon = config['H_sequence']['horizon'][H_seq_idx]
                    if args.render:
                        viewer.horizon = config['H_sequence']['horizon'][H_seq_idx]
                        viewer.update_title()
                update_H, H_seq_idx = update_H_fn(H_seq_idx)
            if update_alpha and (frame_idx % config['alpha_sequence']['steps'][alpha_seq_idx] == 0):
                if args.pointmass:
                    traj.args.alpha = config['alpha_sequence']['alpha'][alpha_seq_idx]
                    if args.render:
                        viewer.alpha = config['alpha_sequence']['alpha'][alpha_seq_idx]
                        viewer.update_title()
                update_alpha, alpha_seq_idx = update_alpha_fn(alpha_seq_idx)
            if update_reward and (frame_idx % config['reward_sequence']['steps'][reward_seq_idx] == 0):
                update_reward, reward_seq_idx = update_reward_fn(reward_seq_idx)

            if frame_idx % (max_frames//10) == 0:
                get_duration(start_time)
                if args.log:
                    print('saving model and reward log')
                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    pickle.dump(eval_rewards, open(path + 'eval_reward_data' + '.pkl', 'wb'))
                    if base_method == 'sac':
                        torch.save(policy_net.state_dict(), path + 'policy_' + str(frame_idx) + '.pt')
                    else:
                        torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')
            if args.done_util:
                if done:
                    break
            if args.singleshot:
                if frame_idx % config['max_steps'] == 0:
                    ep_time = time.time()-ep_start_time
                    ep_start_time = time.time()
                    print('frame : {}/{}, \t {:.2f} seconds'.format(frame_idx, max_frames, ep_time))
                    print('ep rew', ep_num, episode_reward, frame_idx)
                    rewards.append([frame_idx, episode_reward,ep_num])
                    episode_reward = 0
                    if args.pointmass:
                        if args.render:
                            viewer.render(states,f'ss{ss_count}',ncol=int(np.floor(ss_count/15))+1)
                            ss_count += 1
                        traj.push(states,ep_num)
                        states = []
        if not args.singleshot:
            if args.pointmass:
                if args.render:
                    viewer.render(states,ep_num)
                traj.push(states,ep_num)
            step += 1
            ep_time = time.time()-ep_start_time
            print('frame : {}/{}, \t {:.2f} seconds'.format(frame_idx, max_frames, ep_time))
            print('ep rew', ep_num, episode_reward, frame_idx, step)
            rewards.append([frame_idx, episode_reward,ep_num])
            ep_num += 1
            if (ep_num % eval_freq == 0) and (frame_idx > frames_before_learning):
                eval_states, eval_rew, eval_steps = eval()
                eval_rewards.append([frame_idx, eval_rew, ep_num, eval_steps])
                if args.pointmass:
                    plot_name = 'eval'
                    if args.render:
                        viewer.render(eval_states,plot_name)
                    traj.push(eval_states,plot_name)

            with open(path + "/rewards.txt","a") as f:
                f.write('{}\t{}\t{}\t{}\n'.format(ep_num,episode_reward,step,ep_time))
    end_test()
